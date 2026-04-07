import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import Dict, List, Optional
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.expandable_resnet import ExpandableResNet
from buffers.replay_buffer import ReplayBuffer
from utils.metrics import MetricsTracker, AMCATester, evaluate_model, compute_model_size
from utils.losses import ClassBalancedLoss


def check_model_health(model: nn.Module, name: str = "Model") -> bool:
    """Check if model has NaN or Inf weights."""
    has_nan = False
    has_inf = False
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"[WARNING] {name}: NaN detected in {param_name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"[WARNING] {name}: Inf detected in {param_name}")
            has_inf = True
    if has_nan or has_inf:
        print(f"[ERROR] {name} weights are corrupted!")
        return False
    return True


class ContinualLearner:
    """
    Continual learning trainer
    """
    
    def __init__(
        self,
        model: ExpandableResNet,
        replay_buffer: ReplayBuffer,
        config: Dict,
        device: str = 'cuda',
        use_amca: bool = False
    ):
        self.device = device
        
        # Use Channels Last memory format
        if self.device == 'cuda':
            self.model = model.to(device, memory_format=torch.channels_last)
        else:
            self.model = model.to(device)
            
        self.replay_buffer = replay_buffer
        self.config = config
        self.use_amca = use_amca
        
        # Initialize AMP Scaler
        self.use_amp = config['training'].get('use_amp', False) and (device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp, init_scale=2048.0)
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) enabled.")

        self.use_torch_compile = config['training'].get('use_torch_compile', True)
        self.train_model = self.model

        # Initialize metrics tracker
        self.metrics = MetricsTracker(
            num_tasks=config['benchmark']['num_tasks']
        )
        
        # Initialize AMCA tester for CLAD-C
        if use_amca:
            num_classes = config['benchmark'].get('total_classes', 7)
            self.amca_tester = AMCATester(num_classes=num_classes)
        else:
            self.amca_tester = None
        
        # Training config
        self.epochs_per_task = config['training']['epochs_per_task']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training']['optimizer']
        self.weight_decay = config['training']['weight_decay']
        self.eval_frequency = config['training'].get('eval_frequency', 5)

        expansion_cfg = config.get('network_expansion', {})
        self.expansion_enabled = expansion_cfg.get('enabled', True)
        self.expansion_strategy = str(expansion_cfg.get('strategy', 'all_layers')).strip().lower()
        self.expansion_layer_order = self._resolve_expansion_layers(self.expansion_strategy)

        base_params = int(self.model.get_num_parameters())
        max_ratio = float(config.get('benchmark', {}).get('max_params_ratio', 1.05))
        self.max_params = int(base_params * max_ratio)
        
        self.optimizer = None
        self.criterion = None
        
        self.task_expansions = {}
        self.expansion_history = []

    def _resolve_expansion_layers(self, strategy: str) -> List[str]:
        """Resolve configured strategy into an expansion layer schedule."""
        if strategy == 'all_layers':
            return ["layer1", "layer2", "layer3", "layer4"]

        if strategy == 'layer2_layer3_split':
            return ["layer2", "layer3"]

        print(
            f"[WARNING] Unknown network_expansion.strategy='{strategy}'. "
            "Falling back to 'all_layers'."
        )
        self.expansion_strategy = 'all_layers'
        return ["layer1", "layer2", "layer3", "layer4"]

    def _get_task_expansion_targets(self, task_id: int) -> List[str]:
        """Return the layer list to expand for a task-start event."""
        del task_id

        if not self.expansion_layer_order:
            return []

        if self.expansion_strategy == 'all_layers':
            return self._get_all_layers_budget_aware_targets()

        # layer2_layer3_split: expand both layers every task (50/50 within task).
        return list(self.expansion_layer_order)

    def _get_available_param_budget(self) -> int:
        """Return remaining parameter budget before reaching max_params."""
        current_params = int(self.model.get_num_parameters())
        return max(0, int(self.max_params - current_params))

    def _get_all_layers_budget_aware_targets(self) -> List[str]:
        """
        Select all_layers expansion targets using available parameter budget.

        If budget can cover one expansion pass over all eligible layers, expand all.
        Otherwise, allocate budget as evenly as possible across layers.
        """
        layers = list(self.expansion_layer_order)
        if not layers:
            return []

        available_budget = self._get_available_param_budget()
        if available_budget <= 0:
            print("No parameter budget left for network expansion.")
            return []

        layer_costs = {
            layer_name: int(self._estimate_expand_layer_params_delta(layer_name))
            for layer_name in layers
        }
        eligible_layers = [layer_name for layer_name in layers if layer_costs[layer_name] > 0]
        if not eligible_layers:
            return []

        total_required = sum(layer_costs[layer_name] for layer_name in eligible_layers)
        if available_budget >= total_required:
            return eligible_layers

        # Budget-limited case: keep allocation as even as possible across layers.
        remaining_budget = float(available_budget)
        pending = list(eligible_layers)
        selected = []

        while pending and remaining_budget > 0:
            per_layer_share = remaining_budget / len(pending)
            picked = []

            for layer_name in pending:
                layer_cost = layer_costs[layer_name]
                if layer_cost <= per_layer_share and layer_cost <= remaining_budget:
                    selected.append(layer_name)
                    remaining_budget -= layer_cost
                    picked.append(layer_name)

            if picked:
                pending = [layer_name for layer_name in pending if layer_name not in picked]
                continue

            affordable = [
                layer_name for layer_name in pending
                if layer_costs[layer_name] <= remaining_budget
            ]
            if not affordable:
                break

            # Deterministic fallback when strict equal-share assignment is impossible.
            affordable.sort(key=lambda layer_name: layer_costs[layer_name])
            chosen = affordable[0]
            selected.append(chosen)
            remaining_budget -= layer_costs[chosen]
            pending.remove(chosen)

        selected_set = set(selected)
        ordered_selected = [layer_name for layer_name in layers if layer_name in selected_set]

        print(
            f"Budget-aware all_layers selection: available={available_budget:,}, "
            f"required={total_required:,}, selected={ordered_selected}"
        )
        return ordered_selected

    def _expand_network_at_task_start(self, task_id: int):
        """Apply deterministic network expansion once at the start of each task."""
        executed = []

        if not self.expansion_enabled:
            print("Task-start network expansion disabled by config")
            self.task_expansions[task_id] = executed
            return

        print(f"\n--- Task-Start Network Expansion (Task {task_id}) ---")
        targets = self._get_task_expansion_targets(task_id)
        if not targets:
            print("No expansion targets configured")
            self.task_expansions[task_id] = executed
            return

        for layer_name in targets:
            if self._try_expand_layer(layer_name):
                executed.append(layer_name)
            else:
                print(f"No eligible expansion executed for {layer_name} at task {task_id}.")

        self.task_expansions[task_id] = executed
        for layer_name in executed:
            self.expansion_history.append((task_id, f"expand_{layer_name}"))

        if executed:
            print(f"Expanded at task start: {executed}")
        else:
            print("No expansion executed at task start")

    def _prepare_training_model_for_task(self, task_id: int):
        """Compile a training-only model once per task after expansion."""
        self.train_model = self.model

        if not self.use_torch_compile:
            print("torch.compile disabled; using eager model for training")
            return

        try:
            self.train_model = torch.compile(self.model)
            print(f"Compiled training model for task {task_id}")
        except Exception as e:
            print(f"Could not compile training model for task {task_id}: {e}")
            self.train_model = self.model

    def _try_expand_layer(self, layer_name: str) -> bool:
        """Attempt to expand one layer while respecting the parameter budget."""
        try:
            current_params = self.model.get_num_parameters()
            projected_delta = self._estimate_expand_layer_params_delta(layer_name)

            if projected_delta <= 0:
                print(f"No eligible blocks to expand in {layer_name}.")
                return False

            projected_params = current_params + projected_delta
            if projected_params > self.max_params:
                print(
                    f"Skipping {layer_name} expansion: projected params "
                    f"{projected_params:,} would exceed limit {self.max_params:,}."
                )
                return False

            before_trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.model.expand_layer(layer_name)
            after_trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            if after_trainable <= before_trainable:
                print(f"Expansion request for {layer_name} did not add trainable parameters.")
                return False
            return True
        except Exception as e:
            print(f"Error expanding {layer_name}: {e}")
            return False
        
    def _create_optimizer(self, learning_rate: Optional[float] = None):
        """Create optimizer for current trainable parameters."""
        lr = self.learning_rate if learning_rate is None else learning_rate
        use_fused = (self.device == 'cuda')
        extra_args = {}
        if use_fused:
            extra_args['fused'] = True

        if self.optimizer_name.lower() != 'adamw':
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer_name}'. Only 'adamw' is supported."
            )

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.weight_decay,
            eps=1e-8,
            **extra_args
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.epochs_per_task,
            eta_min=self.learning_rate * 0.01
        )

    def train_task(self, task_id: int, train_loader: DataLoader, test_loaders: List[DataLoader]):
        """Train on a single task."""
        print(f"\n{'='*60}")
        print(f"Training Task {task_id}")
        print(f"{'='*60}\n")
        
        if not check_model_health(self.model, f"Task {task_id} start"):
            print(f"[CRITICAL] Model corrupted before Task {task_id}! Training may fail.")
        
        task_start_time = time.time()
        
        if self.model.fc is None:
            num_classes = self._get_num_classes(train_loader)
            self.model.add_task_head(num_classes)
            print(f"Created shared classifier with {num_classes} classes")

        # Deterministic expansion is applied exactly once per task, before optimizer creation.
        self._expand_network_at_task_start(task_id)
        self._prepare_training_model_for_task(task_id)
        
        self._create_optimizer()
        
        # --- Class Balanced Loss Setup ---
        print("Calculating class counts for Class Balanced Loss")
        if hasattr(train_loader.dataset, 'targets'):
            all_labels = train_loader.dataset.targets
            if isinstance(all_labels, list) and len(all_labels) > 0 and torch.is_tensor(all_labels[0]):
                all_labels = [l.item() for l in all_labels]
            
            num_classes = self.config['benchmark'].get('total_classes', 7)
            class_counts = [all_labels.count(i) for i in range(num_classes)]
        else:
            class_counts_dict = {}
            for batch_idx, (_, labels) in enumerate(train_loader):
                labels = labels.cpu().numpy()
                for label in labels:
                    class_counts_dict[label] = class_counts_dict.get(label, 0) + 1
                if batch_idx >= 100: break
            num_classes = self.config['benchmark'].get('total_classes', 7)
            class_counts = [class_counts_dict.get(i, 0) for i in range(num_classes)]
        
        safe_counts = [c if c > 0 else 1 for c in class_counts]
        self.criterion = ClassBalancedLoss(
            class_counts=safe_counts, 
            beta=self.config['training'].get('cb_beta', 0.9999)
        ).to(self.device)
        
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights[0] = 0.0
            
        # Reuse one loader for all evals in this task
        all_test_datasets = []
        for tid in range(task_id + 1):
            if tid < len(test_loaders) and test_loaders[tid] is not None:
                all_test_datasets.append(test_loaders[tid].dataset)
        
        cached_test_loader = None
        cached_location_loaders = None
        if all_test_datasets:
            combined_test_dataset = ConcatDataset(all_test_datasets)
            data_loading_config = self.config.get('data_loading', {})
            num_workers = min(data_loading_config.get('num_workers', 4), 8)
            
            cached_test_loader = DataLoader(
                combined_test_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=data_loading_config.get('pin_memory', True),
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False
            )

            cached_location_loaders = self._build_location_loaders(
                combined_test_dataset=combined_test_dataset,
                batch_size=128,
                num_workers=num_workers,
                pin_memory=data_loading_config.get('pin_memory', True)
            )

        # --- Training Loop ---
        replay_candidates = {'images': [], 'labels': [], 'logits': []} # Logits added for re-use
        
        for epoch in range(self.epochs_per_task):
            is_last_epoch = (epoch == self.epochs_per_task - 1)
            collection_dict = replay_candidates if is_last_epoch else None
            
            self._train_epoch(task_id, train_loader, epoch, collection_dict)
            
            # Evaluations
            if (epoch + 1) % self.eval_frequency == 0 or is_last_epoch:
                print(f"\nEpoch {epoch + 1}/{self.epochs_per_task} - Evaluating")
                self._evaluate_and_update_metrics(
                    task_id,
                    cached_test_loader,
                    location_loaders=cached_location_loaders,
                    run_amca=False
                )
            
        # --- Replay Buffer Update ---
        replay_enabled = self.config.get('replay_buffer', {}).get('enabled', True)
        if replay_enabled:
            if len(replay_candidates['images']) > 0:
                self._update_replay_buffer_from_memory(task_id, replay_candidates)
            else:
                self._update_replay_buffer(task_id, train_loader)

        # Automatically distill replay memory at each chronological task boundary.
        self._auto_distill_replay_buffer_end_of_task(task_id)
        
        # Final Full Evaluation
        print(f"\nFinal evaluation after task {task_id}:")
        self._evaluate_and_update_metrics(
            task_id,
            cached_test_loader,
            location_loaders=cached_location_loaders,
            run_amca=True
        )

        # Record full task compute time (training + replay + final evaluation)
        task_elapsed_time = time.time() - task_start_time
        self.metrics.record_task_time(task_id, task_elapsed_time)
        self.metrics.update_compute_cost(task_elapsed_time)
        print(f"\nTask {task_id} compute time: {task_elapsed_time:.2f} seconds\n")
        
        self.metrics.print_summary(task_id)
        if self.use_amca and self.amca_tester is not None:
            self.amca_tester.print_summary()
    
    def _train_epoch(self, task_id: int, train_loader: DataLoader, epoch: int, collection_dict: Optional[Dict] = None):
        """Train for one epoch with Batched Replay and Logit Capture."""
        from tqdm import tqdm
        
        self.train_model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        
        training_config = self.config.get('training', {})
        use_gradient_accumulation = training_config.get('use_gradient_accumulation', False)
        accumulation_steps = training_config.get('accumulation_steps', 1) if use_gradient_accumulation else 1
        
        self.optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch + 1}", 
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        
        collected_count = 0
        samples_per_class_limit = self.config['replay_buffer'].get('samples_per_class', 200)
        collection_limit = samples_per_class_limit * 7 

        for batch_idx, (images, labels) in pbar:
            try:
                if self.device == 'cuda':
                    images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    images = images.to(self.device, non_blocking=True)
                
                labels = labels.to(self.device, non_blocking=True)
                
                # Logic to capture logits for replay reuse
                capture_outputs = False
                if collection_dict is not None and collected_count < collection_limit:
                    if batch_idx < 20: 
                        capture_outputs = True

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.train_model(images)
                
                if capture_outputs:
                    collection_dict['images'].append(images.detach().cpu())
                    collection_dict['labels'].append(labels.detach().cpu())
                    collection_dict['logits'].append(outputs.detach().cpu())
                    collected_count += images.size(0)

                outputs_float = outputs.float()
                
                if self.criterion is not None:
                    loss = self.criterion(outputs_float, labels)
                else:
                    loss = F.cross_entropy(outputs_float, labels)
            
                replay_enabled = self.config.get('replay_buffer', {}).get('enabled', True)
                if replay_enabled and len(self.replay_buffer) > 0:
                    replay_batch_size = self.config['replay_buffer'].get('replay_batch_size', 32)
                    replay_batch = self.replay_buffer.get_batch(
                        batch_size=min(replay_batch_size, len(self.replay_buffer))
                    )
                    
                    if replay_batch is not None:
                        if self.device == 'cuda':
                            r_imgs = replay_batch['images'].to(self.device, non_blocking=True, memory_format=torch.channels_last)
                        else:
                            r_imgs = replay_batch['images'].to(self.device, non_blocking=True)
                        
                        r_lbls = replay_batch['labels'].to(self.device, non_blocking=True)
                        
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            r_out = self.train_model(r_imgs)
                            
                        loss += 0.5 * F.cross_entropy(r_out.float(), r_lbls)
                        
                        if replay_batch['soft_targets'] is not None:
                            soft_targets = replay_batch['soft_targets'].to(self.device, non_blocking=True).float()
                            
                            # Logits already computed in r_out usually, but we check structure differences.
                            # Re-using r_out for distillation if it matches soft_target shape
                            if r_out.size(1) == soft_targets.size(1):
                                r_out_dist = r_out
                            else:
                                with torch.amp.autocast('cuda', enabled=self.use_amp):
                                    r_out_dist = self.train_model(r_imgs)
                                
                            log_probs = F.log_softmax(r_out_dist.float() / self.replay_buffer.distillation_temp, dim=1)
                            loss += 0.3 * F.kl_div(log_probs, soft_targets, reduction='batchmean')

                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
            except RuntimeError as e:
                if "cuDNN" in str(e) or "CUDA" in str(e):
                    torch.cuda.empty_cache()
                    print(f"\nSkipped batch {batch_idx} due to CUDA error: {e}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise
            
            loss_val = loss.item() * accumulation_steps if torch.is_tensor(loss) else 0.0
            total_loss += loss_val
            _, predicted = outputs_float.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        if (batch_idx + 1) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if hasattr(self, 'scheduler'):
            self.scheduler.step()

    def _update_replay_buffer_from_memory(self, task_id: int, collected_data: Dict):
        """Optimized update using pre-computed logits."""
        if not collected_data['images']:
            print("No samples collected for replay buffer!")
            return

        print(f"\nUpdating replay buffer for task {task_id} (from memory)")
        
        all_images = torch.cat(collected_data['images'], dim=0)
        all_labels = torch.cat(collected_data['labels'], dim=0)
        
        # Use pre-computed logits
        if 'logits' in collected_data and len(collected_data['logits']) > 0:
            all_logits = torch.cat(collected_data['logits'], dim=0)
            soft_targets = F.softmax(all_logits.float() / self.replay_buffer.distillation_temp, dim=1)
        else:
            print("Computing logits (fallback)")
            self.model.eval()
            soft_targets_list = []
            batch_size = 128
            
            with torch.no_grad():
                for i in range(0, len(all_images), batch_size):
                    batch_imgs = all_images[i:i+batch_size]
                    if self.device == 'cuda':
                        batch_imgs = batch_imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        batch_imgs = batch_imgs.to(self.device)
                    
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        outputs = self.model(batch_imgs)
                        soft = F.softmax(outputs.float() / self.replay_buffer.distillation_temp, dim=1)
                    
                    soft_targets_list.append(soft.cpu())
            soft_targets = torch.cat(soft_targets_list, dim=0)
        
        self.replay_buffer.add_samples(all_images, all_labels, task_id, soft_targets)
        print(f"Buffer updated.")

    def _update_replay_buffer(self, task_id: int, train_loader: DataLoader):
        """Fallback method."""
        print(f"\n[Fallback] Updating replay buffer by re-scanning dataset")
        all_images = []
        all_labels = []
        for i, (images, labels) in enumerate(train_loader):
            all_images.append(images)
            all_labels.append(labels)
            if i >= 10: break
            
        if all_images:
            imgs = torch.cat(all_images)
            lbls = torch.cat(all_labels)
            self._update_replay_buffer_from_memory(task_id, {'images': [imgs], 'labels': [lbls], 'logits': []})

    def _auto_distill_replay_buffer_end_of_task(self, task_id: int):
        """Apply replay-buffer distillation at the end of each task if enabled."""
        replay_cfg = self.config.get('replay_buffer', {})
        if not replay_cfg.get('enabled', True):
            return
        if not replay_cfg.get('auto_distill_end_of_task', True):
            return
        if len(self.replay_buffer) == 0:
            print(f"Replay buffer empty after task {task_id}; skipping auto-distillation")
            return

        compression_ratio = replay_cfg.get('end_of_task_distill_compression_ratio', 0.9)
        target_size = replay_cfg.get('end_of_task_distill_target_size', None)

        try:
            if target_size is None:
                ratio_for_call = float(compression_ratio)
                ratio_for_call = min(max(ratio_for_call, 0.01), 1.0)
                self.replay_buffer.distill(
                    self.model,
                    compression_ratio=ratio_for_call,
                    device=self.device
                )
            else:
                self.replay_buffer.distill(
                    self.model,
                    target_size=target_size,
                    device=self.device
                )
            print(f"Auto-distilled replay buffer at end of task {task_id}")
        except Exception as e:
            print(f"[WARNING] Auto replay-buffer distillation failed after task {task_id}: {e}")

    def _resolve_location_for_dataset_index(self, dataset, idx: int):
        """Resolve location string for an index from wrapped/concat/subset datasets."""
        if hasattr(dataset, 'get_location'):
            return dataset.get_location(idx)

        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            parent_idx = dataset.indices[idx]
            return self._resolve_location_for_dataset_index(dataset.dataset, parent_idx)

        if hasattr(dataset, 'datasets'):
            sample_idx = idx
            for d in dataset.datasets:
                if sample_idx < len(d):
                    return self._resolve_location_for_dataset_index(d, sample_idx)
                sample_idx -= len(d)

        return None

    def _build_location_loaders(self, combined_test_dataset, batch_size: int, num_workers: int, pin_memory: bool):
        """Create per-location dataloaders from the same official validation split."""
        location_indices = {
            'Citystreet': [],
            'Countryroad': [],
            'Highway': []
        }

        for idx in range(len(combined_test_dataset)):
            location = self._resolve_location_for_dataset_index(combined_test_dataset, idx)
            if location in location_indices:
                location_indices[location].append(idx)

        location_loaders = {}
        for location, indices in location_indices.items():
            if not indices:
                continue
            subset = Subset(combined_test_dataset, indices)
            location_loaders[location] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False
            )
        return location_loaders

    def _evaluate_and_update_metrics(
        self,
        task_id: int,
        test_loader: DataLoader,
        location_loaders: Optional[Dict[str, DataLoader]] = None,
        run_amca: bool = True
    ):
        # Uses persistent loader
        if test_loader is None: return

        try:
            # Standard Evaluation
            overall_accuracy, per_class_acc = evaluate_model(
                self.model, test_loader, task_id=task_id, device=self.device, return_per_class=True
            )
            self.metrics.update_accuracy(task_id, overall_accuracy, per_class_acc)

            # Per-location evaluation (validation split, grouped by location)
            if location_loaders:
                for location, loader in location_loaders.items():
                    loc_acc, loc_per_class = evaluate_model(
                        self.model, loader, task_id=task_id, device=self.device, return_per_class=True
                    )
                    self.metrics.update_location_accuracy(
                        task_id=task_id,
                        location=location,
                        accuracy=loc_acc,
                        class_accuracies=loc_per_class
                    )
            
            # AMCA Logic
            if self.use_amca and self.amca_tester:
                if run_amca:
                    try:
                        amca_results = self.amca_tester.evaluate(self.model, test_loader, device=self.device)
                        print(f"MCA: {amca_results['mean_class_accuracy']:.4f}, AMCA: {self.amca_tester.compute_amca():.4f}")
                    except Exception as e:
                        print(f"AMCA error: {e}")
                        self.amca_tester.mean_class_accuracies.append(0.0)
                else:
                    if per_class_acc:
                        mca = sum(per_class_acc.values()) / len(per_class_acc)
                        print(f"Current MCA: {mca:.4f}")
                    else:
                        print("MCA: N/A")
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
        
        model_stats = compute_model_size(self.model)
        buffer_memory = self.replay_buffer.get_memory_size()
        self.metrics.update_memory(model_stats['total_mb'] + buffer_memory)
        
        self.metrics.update_num_parameters(int(model_stats['num_params']))
    def _estimate_expand_layer_params_delta(self, layer_name: str) -> int:
        """
        Estimate parameter increase for one expand_layer action.
        Keeps the same eligibility criteria as ExpandableBottleneck.add_adapter().
        """
        if not hasattr(self.model, layer_name):
            return 0

        layer = getattr(self.model, layer_name)
        delta = 0

        for block in layer:
            if not (hasattr(block, 'conv1') and hasattr(block, 'conv3') and hasattr(block, 'stride')):
                continue

            in_channels = block.conv1.in_channels
            out_channels = block.conv3.out_channels
            stride = block.stride
            if isinstance(stride, tuple):
                stride = stride[0]

            if in_channels == out_channels and stride == 1:
                hidden = max(1, in_channels // 16)
                delta += (in_channels * hidden) + (hidden * in_channels)

        return int(delta)

    def _get_num_classes(self, dataloader: DataLoader) -> int:
        labels = set()
        for _, batch_labels in dataloader:
            labels.update(batch_labels.tolist())
            if len(labels) >= 100: break
        return len(labels)

    def save_checkpoint(self, path: str, task_id: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'task_id': task_id,
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'config': self.config,
            'task_expansions': self.task_expansions,
            'expansion_history': self.expansion_history
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint['metrics']
        self.task_expansions = checkpoint.get('task_expansions', {})
        self.expansion_history = checkpoint.get('expansion_history', [])

def create_trainer(config: Dict, model, replay_buffer, device='cuda', use_amca=False):
    return ContinualLearner(
        model=model,
        replay_buffer=replay_buffer,
        config=config,
        device=device,
        use_amca=use_amca
    )
