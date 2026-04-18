import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as torch_prune
import copy
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import Dict, List, Optional
import time
import os
import sys

try:
    import torch_pruning as tp
except ImportError:
    tp = None

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

        pruning_cfg = config.get('network_pruning', {})
        self.pruning_enabled = pruning_cfg.get('enabled', False)
        self.pruning_amount = float(pruning_cfg.get('amount', 0.0))
        self.pruning_scope = str(pruning_cfg.get('scope', 'whole_model')).strip().lower()
        self.pruning_method = str(pruning_cfg.get('method', 'structural')).strip().lower()
        self.pruning_interval_tasks = max(1, int(pruning_cfg.get('interval_tasks', 1)))
        self.pruning_importance = str(pruning_cfg.get('importance', 'magnitude')).strip().lower()
        self.pruning_use_replay_for_taylor = bool(pruning_cfg.get('use_replay_for_taylor', True))
        self.pruning_taylor_replay_batches = max(0, int(pruning_cfg.get('taylor_replay_batches', 2)))
        self.pruning_taylor_train_batches = max(0, int(pruning_cfg.get('taylor_train_batches', 1)))
        self.pruning_taylor_replay_batch_size = max(
            1,
            int(
                pruning_cfg.get(
                    'taylor_replay_batch_size',
                    config.get('replay_buffer', {}).get('replay_batch_size', 32)
                )
            )
        )
        self.pruning_warmup_tasks = max(0, int(pruning_cfg.get('warmup_tasks', 2)))
        self.pruning_budget_driven = bool(pruning_cfg.get('budget_driven', True))
        self.pruning_min_amount = float(pruning_cfg.get('min_amount', 0.0))
        self.pruning_max_amount = float(pruning_cfg.get('max_amount', self.pruning_amount))
        self.pruning_example_input_shape = tuple(
            pruning_cfg.get('example_input_shape', [1, 3, 224, 224])
        )
        self.rebuild_optimizer_after_prune = bool(
            pruning_cfg.get('rebuild_optimizer_after_prune', True)
        )
        self.recompile_after_prune = bool(
            pruning_cfg.get('recompile_after_prune', True)
        )

        if (
            self.use_torch_compile
            and self.pruning_enabled
            and self.pruning_scope in {'whole_model', 'hybrid'}
            and self.pruning_method == 'structural'
        ):
            try:
                import torch._dynamo as dynamo
                dynamo.config.force_parameter_static_shapes = False
                print(
                    "Enabled dynamic parameter shapes for torch.compile "
                    "(structural pruning is active)."
                )
            except Exception as e:
                print(
                    "[WARNING] Could not enable dynamic parameter shapes for torch.compile: "
                    f"{e}"
                )

        self.protect_current_task_adapters = bool(
            pruning_cfg.get('protect_current_task_adapters', True)
        )
        self.protect_classifier = bool(pruning_cfg.get('protect_classifier', True))
        self.protect_downsample = bool(pruning_cfg.get('protect_downsample', True))

        rollback_cfg = pruning_cfg.get('rollback', {})
        self.prune_rollback_enabled = bool(rollback_cfg.get('enabled', False))
        self.prune_rollback_max_mca_drop = float(rollback_cfg.get('max_mca_drop', 0.01))
        self.prune_rollback_eval_max_batches = max(
            1,
            int(rollback_cfg.get('eval_max_batches', 20))
        )

        recovery_cfg = pruning_cfg.get('recovery', {})
        self.prune_recovery_enabled = bool(recovery_cfg.get('enabled', True))
        self.prune_recovery_epochs = max(0, int(recovery_cfg.get('epochs', 4)))
        self.prune_recovery_lr_scale = float(recovery_cfg.get('learning_rate_scale', 0.2))

        self.pruning_amount = min(max(self.pruning_amount, 0.0), 1.0)
        self.pruning_min_amount = min(max(self.pruning_min_amount, 0.0), 1.0)
        self.pruning_max_amount = min(max(self.pruning_max_amount, 0.0), 1.0)
        if self.pruning_max_amount < self.pruning_min_amount:
            self.pruning_max_amount = self.pruning_min_amount
        if self.pruning_importance not in {'magnitude', 'taylor'}:
            print(
                f"[WARNING] Unsupported network_pruning.importance='{self.pruning_importance}'. "
                "Falling back to 'magnitude'."
            )
            self.pruning_importance = 'magnitude'
        self.prune_rollback_max_mca_drop = max(0.0, self.prune_rollback_max_mca_drop)
        self.prune_recovery_lr_scale = max(self.prune_recovery_lr_scale, 0.0)
        if len(self.pruning_example_input_shape) != 4:
            self.pruning_example_input_shape = (1, 3, 224, 224)

        base_params = int(self._get_effective_model_params())
        max_ratio = float(config.get('benchmark', {}).get('max_params_ratio', 1.05))
        self.max_params = int(base_params * max_ratio)
        
        self.optimizer = None
        self.criterion = None
        
        self.task_expansions = {}
        self.expansion_history = []
        self.pruning_history = []
        self.current_task_new_adapter_ids = set()
        self.last_taylor_gradient_stats = {
            'replay_batches': 0,
            'train_batches': 0,
            'total_batches': 0
        }
        self.last_good_model_snapshot: Optional[nn.Module] = None
        self.last_good_task_id: Optional[int] = None

    def _is_pruning_task(self, task_id: int) -> bool:
        """Return whether pruning should run for this task id under warmup/interval rules."""
        if not self.pruning_enabled:
            return False
        if task_id < self.pruning_warmup_tasks:
            return False
        offset = task_id - self.pruning_warmup_tasks
        return (offset % self.pruning_interval_tasks) == 0

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

    def _get_effective_model_params(self) -> int:
        """
        Count active model parameters for budget checks.

        When pruning masks are present, masked-out parameters are excluded.
        """
        if hasattr(self.model, 'get_effective_num_parameters'):
            return int(self.model.get_effective_num_parameters())
        if hasattr(self.model, 'get_num_parameters'):
            return int(self.model.get_num_parameters())
        return int(sum(p.numel() for p in self.model.parameters()))

    def _get_available_param_budget(self) -> int:
        """Return remaining parameter budget before reaching max_params."""
        current_params = self._get_effective_model_params()
        return max(0, int(self.max_params - current_params))

    def _estimate_budget_needed_for_next_task_expansion(self) -> int:
        """
        Estimate how much free budget is needed to make the next expansion step feasible.

        - all_layers: enough for one full pass over eligible layers.
        - layer2_layer3_split: enough for both layer2 and layer3.
        """
        if not self.expansion_enabled:
            return 0

        layers = list(self.expansion_layer_order)
        if not layers:
            return 0

        costs = []
        for layer_name in layers:
            delta = int(self._estimate_expand_layer_params_delta(layer_name))
            if delta > 0:
                costs.append(delta)

        if not costs:
            return 0

        if self.expansion_strategy in {'all_layers', 'layer2_layer3_split'}:
            return int(sum(costs))

        return int(sum(costs))

    def _resolve_budget_driven_prune_amount(
        self,
        prunable_effective_params: int,
        task_id: int
    ) -> float:
        """Resolve pruning amount, optionally driven by next-task budget requirements."""
        configured_amount = min(max(float(self.pruning_amount), 0.0), 1.0)
        if configured_amount <= 0.0:
            return 0.0

        if prunable_effective_params <= 0:
            return 0.0

        if not self.pruning_budget_driven:
            return configured_amount

        target_budget = self._estimate_budget_needed_for_next_task_expansion()
        if target_budget <= 0:
            print(
                f"Budget-driven pruning skipped at task {task_id}: "
                "no next-task expansion pressure detected"
            )
            return 0.0

        available_budget = self._get_available_param_budget()
        missing_budget = max(0, target_budget - available_budget)
        if missing_budget <= 0:
            print(
                f"Budget-driven pruning skipped at task {task_id}: "
                f"available budget {available_budget:,} already meets target {target_budget:,}"
            )
            return 0.0

        required_fraction = float(missing_budget) / float(max(1, prunable_effective_params))
        resolved_amount = max(self.pruning_min_amount, required_fraction)
        resolved_amount = min(self.pruning_max_amount, resolved_amount)
        resolved_amount = min(max(resolved_amount, 0.0), 1.0)

        estimated_freed = int(prunable_effective_params * resolved_amount)
        if estimated_freed < missing_budget:
            print(
                f"Budget-driven pruning at task {task_id} is capped by max_amount={self.pruning_max_amount:.4f}: "
                f"needed~{missing_budget:,}, estimated_freed~{estimated_freed:,}"
            )
        else:
            print(
                f"Budget-driven pruning at task {task_id}: "
                f"needed~{missing_budget:,}, prunable={prunable_effective_params:,}, "
                f"resolved_amount={resolved_amount:.4f}"
            )

        return resolved_amount

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
        before_adapter_ids = {id(adapter) for _, _, adapter in self._iter_adapters()}
        self.current_task_new_adapter_ids = set()

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

        after_adapter_ids = {id(adapter) for _, _, adapter in self._iter_adapters()}
        self.current_task_new_adapter_ids = after_adapter_ids - before_adapter_ids

        if executed:
            print(f"Expanded at task start: {executed}")
        else:
            print("No expansion executed at task start")

    def _iter_adapters(self):
        """Yield (layer_name, block_index, adapter_module) for all adapters in model."""
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if not hasattr(self.model, layer_name):
                continue
            layer = getattr(self.model, layer_name)
            for block_idx, block in enumerate(layer):
                adapters = getattr(block, 'adapters', None)
                if adapters is None:
                    continue
                for adapter in adapters:
                    yield layer_name, block_idx, adapter

    def _get_pruning_example_input(self, train_loader: Optional[DataLoader] = None) -> Optional[torch.Tensor]:
        """Build example input for graph-based structural pruning."""
        if train_loader is not None:
            try:
                first_batch = next(iter(train_loader))
                images = None

                if isinstance(first_batch, (tuple, list)) and len(first_batch) > 0:
                    if torch.is_tensor(first_batch[0]):
                        images = first_batch[0]
                elif isinstance(first_batch, dict):
                    candidate = first_batch.get('images', None)
                    if torch.is_tensor(candidate):
                        images = candidate

                if images is not None and images.ndim >= 4 and images.size(0) > 0:
                    example = images[:1].to(self.device, non_blocking=True)
                    if self.device == 'cuda':
                        example = example.contiguous(memory_format=torch.channels_last)
                    return example
            except Exception as e:
                print(f"[WARNING] Could not derive pruning example input from loader: {e}")

        try:
            shape = tuple(int(v) for v in self.pruning_example_input_shape)
            example = torch.randn(*shape, device=self.device)
            if self.device == 'cuda':
                example = example.contiguous(memory_format=torch.channels_last)
            return example
        except Exception as e:
            print(f"[WARNING] Could not create fallback pruning example input: {e}")
            return None

    def _collect_structural_pruning_ignored_layers(self) -> List[nn.Module]:
        """Collect strict protection list for structural pruning."""
        ignored_layers = []
        ignored_ids = set()

        if self.protect_classifier:
            classifier_module = getattr(self.model, 'fc', None)
            if classifier_module is not None and id(classifier_module) not in ignored_ids:
                ignored_layers.append(classifier_module)
                ignored_ids.add(id(classifier_module))

        for module_name, module in self.model.named_modules():
            should_ignore = False

            if self.protect_classifier and module_name == 'fc':
                should_ignore = True

            if self.protect_downsample and 'downsample' in module_name:
                should_ignore = True

            if should_ignore and id(module) not in ignored_ids:
                ignored_layers.append(module)
                ignored_ids.add(id(module))

        if self.protect_current_task_adapters:
            for _, _, adapter in self._iter_adapters():
                if id(adapter) not in self.current_task_new_adapter_ids:
                    continue
                for m in adapter.modules():
                    if not isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                        continue
                    if id(m) in ignored_ids:
                        continue
                    ignored_layers.append(m)
                    ignored_ids.add(id(m))

        return ignored_layers

    def _compute_mca_from_loader(
        self,
        dataloader: Optional[DataLoader],
        max_batches: Optional[int] = None
    ) -> Optional[float]:
        """Compute MCA on a dataloader without mutating AMCA history state."""
        if dataloader is None:
            return None

        if max_batches is None:
            max_batches = self.prune_rollback_eval_max_batches
        max_batches = max(1, int(max_batches))

        num_classes = int(self.config.get('benchmark', {}).get('total_classes', 7))
        max_cls = max(1, num_classes)
        class_correct = torch.zeros(max_cls, device=self.device, dtype=torch.long)
        class_total = torch.zeros(max_cls, device=self.device, dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    continue

                images, labels = batch[0], batch[1]
                if self.device == 'cuda':
                    images = images.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )
                else:
                    images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                predicted = outputs.argmax(dim=1)

                class_total += torch.bincount(labels, minlength=max_cls)[:max_cls]
                correct_mask = (predicted == labels)
                class_correct += torch.bincount(labels[correct_mask], minlength=max_cls)[:max_cls]

        valid = class_total > 0
        if valid.any():
            mca = (class_correct.float()[valid] / class_total.float()[valid]).mean().item()
            return float(mca)
        return 0.0

    def _snapshot_model_for_rollback(self) -> Optional[nn.Module]:
        """Create a CPU snapshot for prune rollback checks."""
        if not self.prune_rollback_enabled:
            return None
        try:
            return copy.deepcopy(self.model).cpu()
        except Exception as e:
            print(f"[WARNING] Could not snapshot model for rollback: {e}")
            return None

    def _restore_model_from_snapshot(self, snapshot_model: Optional[nn.Module], task_id: int) -> bool:
        """Restore model from rollback snapshot and refresh training state."""
        if snapshot_model is None:
            return False

        try:
            restored_model = snapshot_model.to(self.device)

            self.model = restored_model
            self.train_model = self.model

            self._refresh_training_state_after_prune(
                task_id=task_id,
                learning_rate=self.learning_rate,
                scheduler_tmax=self.epochs_per_task
            )
            return True
        except Exception as e:
            print(f"[WARNING] Could not restore rollback snapshot: {e}")
            return False

    def _model_has_nonfinite_parameters(self) -> bool:
        """Return True if any model parameter contains NaN or Inf."""
        for parameter in self.model.parameters():
            if not torch.isfinite(parameter).all():
                return True
        return False

    def _snapshot_last_good_model(self, task_id: int) -> bool:
        """Persist a CPU snapshot of a healthy model at task boundary."""
        try:
            self.last_good_model_snapshot = copy.deepcopy(self.model).cpu()
            self.last_good_task_id = int(task_id)
            return True
        except Exception as e:
            print(f"[WARNING] Could not snapshot last-good model at task {task_id}: {e}")
            return False

    def _restore_last_good_model(self, task_id: int) -> bool:
        """Restore model from the most recent healthy task snapshot."""
        if self.last_good_model_snapshot is None:
            return False

        try:
            restored = copy.deepcopy(self.last_good_model_snapshot).to(self.device)
            if self.device == 'cuda':
                restored = restored.to(memory_format=torch.channels_last)
            self.model = restored
            self.train_model = self.model
            print(
                f"Restored last-good model snapshot from task {self.last_good_task_id} "
                f"before starting task {task_id}"
            )
            return True
        except Exception as e:
            print(f"[WARNING] Could not restore last-good model snapshot: {e}")
            return False

    def _prepare_taylor_gradients_for_pruning(
        self,
        train_loader: Optional[DataLoader] = None
    ) -> bool:
        """Accumulate first-order gradients used by Taylor importance."""
        expected_num_classes = int(self.config.get('benchmark', {}).get('total_classes', 7))
        self._ensure_classifier_head(expected_num_classes)

        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        replay_batches_used = 0
        train_batches_used = 0

        if (
            self.pruning_use_replay_for_taylor
            and self.pruning_taylor_replay_batches > 0
            and len(self.replay_buffer) > 0
        ):
            replay_batch_size = min(self.pruning_taylor_replay_batch_size, len(self.replay_buffer))
            for _ in range(self.pruning_taylor_replay_batches):
                replay_batch = self.replay_buffer.get_batch(batch_size=replay_batch_size)
                if replay_batch is None:
                    break

                if self.device == 'cuda':
                    replay_images = replay_batch['images'].to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )
                else:
                    replay_images = replay_batch['images'].to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True
                    )

                replay_labels = replay_batch['labels'].to(self.device, non_blocking=True)
                replay_logits = self.model(replay_images)

                num_output_classes = int(replay_logits.size(1))
                valid_label_mask = (replay_labels >= 0) & (replay_labels < num_output_classes)
                if not bool(valid_label_mask.all().item()):
                    invalid_count = int((~valid_label_mask).sum().item())
                    valid_count = int(valid_label_mask.sum().item())
                    print(
                        "[WARNING] Taylor replay batch contains out-of-range labels; "
                        f"dropping {invalid_count} samples (n_classes={num_output_classes})."
                    )
                    if valid_count <= 0:
                        continue
                    replay_images = replay_images[valid_label_mask]
                    replay_labels = replay_labels[valid_label_mask]
                    replay_logits = replay_logits[valid_label_mask]

                replay_loss = F.cross_entropy(replay_logits.float(), replay_labels)

                if replay_batch.get('soft_targets', None) is not None:
                    soft_targets = replay_batch['soft_targets'].to(self.device, non_blocking=True).float()
                    if soft_targets.size(0) == valid_label_mask.size(0):
                        soft_targets = soft_targets[valid_label_mask]
                    if soft_targets.ndim == 2 and soft_targets.size(1) == replay_logits.size(1):
                        temp = max(float(self.replay_buffer.distillation_temp), 1e-6)
                        replay_log_probs = F.log_softmax(
                            replay_logits.float() / temp,
                            dim=1
                        )
                        replay_loss += 0.5 * (temp ** 2) * F.kl_div(
                            replay_log_probs,
                            soft_targets,
                            reduction='batchmean'
                        )

                replay_loss.backward()
                replay_batches_used += 1

        if train_loader is not None and self.pruning_taylor_train_batches > 0:
            for batch_idx, (images, labels) in enumerate(train_loader):
                if train_batches_used >= self.pruning_taylor_train_batches:
                    break

                if self.device == 'cuda':
                    images = images.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )
                else:
                    images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)

                num_output_classes = int(logits.size(1))
                valid_label_mask = (labels >= 0) & (labels < num_output_classes)
                if not bool(valid_label_mask.all().item()):
                    invalid_count = int((~valid_label_mask).sum().item())
                    valid_count = int(valid_label_mask.sum().item())
                    print(
                        "[WARNING] Taylor train batch contains out-of-range labels; "
                        f"dropping {invalid_count} samples (n_classes={num_output_classes})."
                    )
                    if valid_count <= 0:
                        continue
                    logits = logits[valid_label_mask]
                    labels = labels[valid_label_mask]

                loss = F.cross_entropy(logits.float(), labels)
                loss.backward()
                train_batches_used += 1

                if train_batches_used >= self.pruning_taylor_train_batches:
                    break

        total_batches = replay_batches_used + train_batches_used
        self.last_taylor_gradient_stats = {
            'replay_batches': int(replay_batches_used),
            'train_batches': int(train_batches_used),
            'total_batches': int(total_batches)
        }

        if total_batches <= 0:
            self.model.zero_grad(set_to_none=True)
            return False

        scale = 1.0 / float(total_batches)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
            else:
                # Ensure every parameter has a gradient tensor so Taylor-based
                # importance never receives None gradients from torch-pruning.
                parameter.grad = torch.zeros_like(parameter)

        print(
            "Prepared Taylor gradients for structural pruning: "
            f"replay_batches={replay_batches_used}, "
            f"train_batches={train_batches_used}"
        )
        return True

    def _resolve_structural_importance(
        self,
        train_loader: Optional[DataLoader] = None
    ):
        """Resolve structural pruning importance criterion from config."""
        if tp is None:
            raise RuntimeError("torch-pruning is required for structural importance resolution")

        if self.pruning_importance != 'taylor':
            self.last_taylor_gradient_stats = {
                'replay_batches': 0,
                'train_batches': 0,
                'total_batches': 0
            }
            return tp.importance.MagnitudeImportance(p=1), 'magnitude'

        taylor_ready = self._prepare_taylor_gradients_for_pruning(train_loader=train_loader)
        if not taylor_ready:
            print(
                "[WARNING] Taylor importance requested but no gradients were prepared. "
                "Falling back to magnitude importance."
            )
            return tp.importance.MagnitudeImportance(p=1), 'magnitude'

        taylor_cls = getattr(tp.importance, 'GroupTaylorImportance', None)
        if taylor_cls is None:
            taylor_cls = getattr(tp.importance, 'TaylorImportance', None)

        if taylor_cls is None:
            print(
                "[WARNING] torch-pruning Taylor importance class not found. "
                "Falling back to magnitude importance."
            )
            return tp.importance.MagnitudeImportance(p=1), 'magnitude'

        try:
            return taylor_cls(), 'taylor'
        except Exception as e:
            print(
                "[WARNING] Could not initialize Taylor importance "
                f"({e}); falling back to magnitude importance."
            )
            return tp.importance.MagnitudeImportance(p=1), 'magnitude'

    def _maybe_rollback_post_prune(
        self,
        task_id: int,
        prune_applied: bool,
        eval_loader: Optional[DataLoader],
        pre_prune_mca: Optional[float],
        snapshot_model: Optional[nn.Module]
    ) -> bool:
        """Rollback pruning if post-prune MCA drop exceeds configured threshold."""
        if not prune_applied:
            return False
        if not self.prune_rollback_enabled:
            return False
        if eval_loader is None:
            print(f"Skipping prune rollback check at task {task_id}: no evaluation loader")
            return False
        if pre_prune_mca is None:
            print(f"Skipping prune rollback check at task {task_id}: pre-prune MCA unavailable")
            return False

        try:
            post_prune_mca = self._compute_mca_from_loader(
                eval_loader,
                max_batches=self.prune_rollback_eval_max_batches
            )
        except Exception as e:
            print(f"[WARNING] Could not compute post-prune MCA for rollback check: {e}")
            return False

        if post_prune_mca is None:
            return False

        mca_drop = float(pre_prune_mca - post_prune_mca)
        rollback_needed = mca_drop > self.prune_rollback_max_mca_drop

        if self.pruning_history and self.pruning_history[-1].get('task_id') == int(task_id):
            self.pruning_history[-1]['rollback_pre_mca'] = float(pre_prune_mca)
            self.pruning_history[-1]['rollback_post_mca'] = float(post_prune_mca)
            self.pruning_history[-1]['rollback_mca_drop'] = float(mca_drop)
            self.pruning_history[-1]['rollback_threshold'] = float(self.prune_rollback_max_mca_drop)

        if not rollback_needed:
            print(
                f"Prune rollback check passed at task {task_id}: "
                f"MCA drop={mca_drop:.4f} <= threshold={self.prune_rollback_max_mca_drop:.4f}"
            )
            if self.pruning_history and self.pruning_history[-1].get('task_id') == int(task_id):
                self.pruning_history[-1]['rolled_back'] = False
            return False

        restored = self._restore_model_from_snapshot(snapshot_model=snapshot_model, task_id=task_id)
        if not restored:
            print(
                f"[WARNING] Prune rollback triggered at task {task_id} but restore failed; "
                "keeping pruned model."
            )
            if self.pruning_history and self.pruning_history[-1].get('task_id') == int(task_id):
                self.pruning_history[-1]['rolled_back'] = False
                self.pruning_history[-1]['rollback_restore_failed'] = True
            return False

        print(
            f"Rolled back pruning at task {task_id}: "
            f"MCA drop={mca_drop:.4f} > threshold={self.prune_rollback_max_mca_drop:.4f}"
        )
        if self.pruning_history and self.pruning_history[-1].get('task_id') == int(task_id):
            self.pruning_history[-1]['rolled_back'] = True

        return True

    def _refresh_training_state_after_prune(
        self,
        task_id: int,
        learning_rate: Optional[float] = None,
        scheduler_tmax: Optional[int] = None
    ):
        """Refresh optimizer and compiled training graph after structural pruning."""
        expected_num_classes = int(self.config.get('benchmark', {}).get('total_classes', 7))
        self._ensure_classifier_head(expected_num_classes)

        if self.rebuild_optimizer_after_prune:
            self._create_optimizer(
                learning_rate=learning_rate,
                scheduler_tmax=scheduler_tmax
            )
            print(f"Rebuilt optimizer after pruning for task {task_id}")

        self.train_model = self.model
        if self.recompile_after_prune and self.use_torch_compile:
            try:
                try:
                    import torch._dynamo as dynamo
                    dynamo.reset()
                except Exception:
                    pass
                self.train_model = torch.compile(self.model)
                print(f"Recompiled training model after pruning for task {task_id}")
            except Exception as e:
                print(f"Could not recompile training model after pruning for task {task_id}: {e}")
                self.train_model = self.model

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
            current_params = self._get_effective_model_params()
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
        
    def _create_optimizer(
        self,
        learning_rate: Optional[float] = None,
        scheduler_tmax: Optional[int] = None
    ):
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

        t_max = self.epochs_per_task if scheduler_tmax is None else max(1, int(scheduler_tmax))
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=t_max,
            eta_min=lr * 0.01
        )

    def _ensure_classifier_head(self, expected_num_classes: int):
        """Ensure classifier head exists and matches expected class count."""
        expected_num_classes = max(1, int(expected_num_classes))
        current_fc = getattr(self.model, 'fc', None)

        if current_fc is None:
            self.model.add_task_head(expected_num_classes)
            print(f"Created shared classifier with {expected_num_classes} classes")
            return

        current_out = int(getattr(current_fc, 'out_features', expected_num_classes))
        if current_out == expected_num_classes:
            return

        device = next(self.model.parameters()).device
        current_in = int(getattr(current_fc, 'in_features', getattr(self.model, 'feature_dim', 2048)))
        new_fc = nn.Linear(current_in, expected_num_classes).to(device)

        try:
            with torch.no_grad():
                copy_out = min(current_out, expected_num_classes)
                copy_in = min(
                    int(getattr(current_fc, 'in_features', new_fc.in_features)),
                    int(new_fc.in_features)
                )
                if hasattr(current_fc, 'weight') and current_fc.weight is not None:
                    new_fc.weight[:copy_out, :copy_in].copy_(current_fc.weight[:copy_out, :copy_in])
                if (
                    hasattr(current_fc, 'bias')
                    and current_fc.bias is not None
                    and new_fc.bias is not None
                ):
                    new_fc.bias[:copy_out].copy_(current_fc.bias[:copy_out])
        except Exception as e:
            print(f"[WARNING] Could not transfer classifier weights while resizing head: {e}")

        self.model.fc = new_fc
        print(
            f"[WARNING] Resized classifier head from {current_out} to "
            f"{expected_num_classes} outputs to match config/classes."
        )

    def train_task(self, task_id: int, train_loader: DataLoader, test_loaders: List[DataLoader]):
        """Train on a single task."""
        print(f"\n{'='*60}")
        print(f"Training Task {task_id}")
        print(f"{'='*60}\n")
        
        if not check_model_health(self.model, f"Task {task_id} start"):
            restored = self._restore_last_good_model(task_id)
            if not restored:
                raise RuntimeError(
                    f"Model corrupted before Task {task_id}, and no healthy snapshot is available. "
                    "Stopping to avoid poisoning replay and metrics."
                )
            if not check_model_health(self.model, f"Task {task_id} restored-start"):
                raise RuntimeError(
                    f"Model remains corrupted after restore attempt before Task {task_id}."
                )
        
        task_start_time = time.time()

        expected_num_classes = int(self.config.get('benchmark', {}).get('total_classes', 7))
        self._ensure_classifier_head(expected_num_classes)

        # Deterministic expansion is applied exactly once per task, before optimizer creation.
        self._expand_network_at_task_start(task_id)

        frozen_historical_adapters = 0
        trainable_new_adapters = 0
        for _, _, adapter in self._iter_adapters():
            is_new_adapter = id(adapter) in self.current_task_new_adapter_ids
            for param in adapter.parameters():
                param.requires_grad = bool(is_new_adapter)
            if is_new_adapter:
                trainable_new_adapters += 1
            else:
                frozen_historical_adapters += 1

        print(
            "Adapter freeze policy applied: "
            f"trainable_new={trainable_new_adapters}, "
            f"frozen_historical={frozen_historical_adapters}"
        )

        self._prepare_training_model_for_task(task_id)
        
        self._create_optimizer()
        
        # --- Class Balanced Loss Setup ---
        print("Calculating class counts for Class Balanced Loss")
        num_classes = int(
            getattr(self.model.fc, 'out_features', self.config['benchmark'].get('total_classes', 7))
        )
        max_label_observed = -1

        if hasattr(train_loader.dataset, 'targets'):
            all_labels = train_loader.dataset.targets
            if isinstance(all_labels, list) and len(all_labels) > 0 and torch.is_tensor(all_labels[0]):
                all_labels = [l.item() for l in all_labels]

            all_labels = [int(v) for v in all_labels]
            if all_labels:
                max_label_observed = max(all_labels)
            class_counts = [all_labels.count(i) for i in range(num_classes)]
        else:
            class_counts_dict = {}
            for batch_idx, (_, labels) in enumerate(train_loader):
                labels = labels.cpu().numpy()
                for label in labels:
                    label_int = int(label)
                    max_label_observed = max(max_label_observed, label_int)
                    class_counts_dict[label_int] = class_counts_dict.get(label_int, 0) + 1
                if batch_idx >= 100: break
            class_counts = [class_counts_dict.get(i, 0) for i in range(num_classes)]

        if max_label_observed >= num_classes:
            raise RuntimeError(
                f"Label range mismatch at task {task_id}: observed max label {max_label_observed}, "
                f"but classifier outputs only {num_classes} classes."
            )
        
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

        pre_prune_mca = None
        rollback_snapshot = None
        if self.prune_rollback_enabled and self._is_pruning_task(task_id):
            pre_prune_mca = self._compute_mca_from_loader(
                cached_test_loader,
                max_batches=self.prune_rollback_eval_max_batches
            )
            rollback_snapshot = self._snapshot_model_for_rollback()
            if pre_prune_mca is not None:
                print(f"Pre-prune MCA estimate (task {task_id}): {pre_prune_mca:.4f}")

        prune_applied = self._auto_prune_network_end_of_task(task_id, train_loader=train_loader)
        self._post_prune_recovery(task_id, train_loader, prune_applied)

        rolled_back = self._maybe_rollback_post_prune(
            task_id=task_id,
            prune_applied=prune_applied,
            eval_loader=cached_test_loader,
            pre_prune_mca=pre_prune_mca,
            snapshot_model=rollback_snapshot
        )
        if rolled_back:
            prune_applied = False
        
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

        if self._model_has_nonfinite_parameters():
            print(
                f"[WARNING] Task {task_id} ended with non-finite parameters; "
                "last-good snapshot not updated."
            )
        else:
            self._snapshot_last_good_model(task_id)
    
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
        max_nonfinite_batches = max(
            1,
            int(training_config.get('max_nonfinite_batches_per_epoch', 5))
        )
        nonfinite_batches = 0
        has_pending_grad = False
        
        self.optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch + 1}", 
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        
        collected_count = 0
        samples_per_class_limit = self.config['replay_buffer'].get('samples_per_class', 200)
        num_classes = int(self.config['benchmark'].get('total_classes', 7))
        collection_limit = samples_per_class_limit * num_classes
        class_capture_counts = {class_id: 0 for class_id in range(num_classes)}

        for batch_idx, (images, labels) in pbar:
            try:
                if self.device == 'cuda':
                    images = images.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )
                else:
                    images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                
                labels = labels.to(self.device, non_blocking=True)
                
                # Logic to capture logits for replay reuse
                capture_indices = None
                if collection_dict is not None and collected_count < collection_limit:
                    selected_indices = []
                    labels_cpu = labels.detach().cpu()
                    unique_batch_labels = labels_cpu.unique(sorted=False)

                    for class_tensor in unique_batch_labels:
                        class_id = int(class_tensor.item())
                        if class_id < 0:
                            continue
                        if class_id not in class_capture_counts:
                            continue

                        remaining_for_class = samples_per_class_limit - class_capture_counts[class_id]
                        if remaining_for_class <= 0:
                            continue

                        class_positions = torch.nonzero(labels_cpu == class_id, as_tuple=True)[0]
                        if class_positions.numel() == 0:
                            continue

                        take_count = min(int(class_positions.numel()), int(remaining_for_class))
                        selected_indices.append(class_positions[:take_count])

                    if selected_indices:
                        capture_indices = torch.cat(selected_indices, dim=0)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.train_model(images)
                
                if capture_indices is not None and capture_indices.numel() > 0:
                    selected_images = images.index_select(0, capture_indices.to(images.device)).detach().cpu()
                    selected_labels = labels.index_select(0, capture_indices.to(labels.device)).detach().cpu()
                    selected_logits = outputs.index_select(0, capture_indices.to(outputs.device)).detach().cpu()

                    collection_dict['images'].append(selected_images)
                    collection_dict['labels'].append(selected_labels)
                    collection_dict['logits'].append(selected_logits)

                    for selected_label in selected_labels.tolist():
                        label_int = int(selected_label)
                        if label_int in class_capture_counts:
                            class_capture_counts[label_int] += 1

                    collected_count += int(selected_labels.numel())

                outputs_float = outputs.float()

                num_output_classes = int(outputs_float.size(1))
                if self.criterion is not None and hasattr(self.criterion, 'weights'):
                    weight_count = int(self.criterion.weights.numel())
                    if weight_count != num_output_classes:
                        raise RuntimeError(
                            f"Class weight mismatch at task {task_id}: criterion has {weight_count} weights "
                            f"but model outputs {num_output_classes} classes."
                        )
                if labels.numel() > 0:
                    max_label_in_batch = int(labels.max().item())
                    if max_label_in_batch >= num_output_classes:
                        raise RuntimeError(
                            f"Label out of range at task {task_id}: max label {max_label_in_batch}, "
                            f"but model outputs {num_output_classes} classes."
                        )
                
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
                            r_imgs = replay_batch['images'].to(
                                self.device,
                                dtype=torch.float32,
                                non_blocking=True,
                                memory_format=torch.channels_last
                            )
                        else:
                            r_imgs = replay_batch['images'].to(
                                self.device,
                                dtype=torch.float32,
                                non_blocking=True
                            )
                        
                        r_lbls = replay_batch['labels'].to(self.device, non_blocking=True)
                        
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            r_out = self.train_model(r_imgs)

                        if self.criterion is not None:
                            loss += 0.5 * self.criterion(r_out.float(), r_lbls)
                        else:
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

                            temp = max(float(self.replay_buffer.distillation_temp), 1e-6)
                            log_probs = F.log_softmax(r_out_dist.float() / temp, dim=1)
                            loss += 0.3 * (temp ** 2) * F.kl_div(log_probs, soft_targets, reduction='batchmean')

                if not torch.isfinite(loss).all():
                    nonfinite_batches += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    has_pending_grad = False
                    print(
                        f"\n[WARNING] Non-finite loss at task {task_id}, epoch {epoch + 1}, "
                        f"batch {batch_idx}; skipped ({nonfinite_batches}/{max_nonfinite_batches})"
                    )
                    if nonfinite_batches >= max_nonfinite_batches:
                        raise RuntimeError(
                            f"Too many non-finite losses in task {task_id}, epoch {epoch + 1}. "
                            "Stopping to prevent model corruption."
                        )
                    continue

                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                has_pending_grad = True
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if has_pending_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        has_pending_grad = False
                    
            except RuntimeError as e:
                if "cuDNN" in str(e) or "CUDA" in str(e):
                    torch.cuda.empty_cache()
                    print(f"\nSkipped batch {batch_idx} due to CUDA error: {e}")
                    self.optimizer.zero_grad(set_to_none=True)
                    has_pending_grad = False
                    continue
                else:
                    raise
            
            loss_val = loss.item() * accumulation_steps if torch.is_tensor(loss) else 0.0
            total_loss += loss_val
            _, predicted = outputs_float.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        if (batch_idx + 1) % accumulation_steps != 0 and has_pending_grad:
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
            finite_logits_mask = torch.isfinite(all_logits).all(dim=1)
            finite_count = int(finite_logits_mask.sum().item())
            if finite_count <= 0:
                print(
                    f"[WARNING] Replay update skipped for task {task_id}: "
                    "all cached logits are non-finite."
                )
                return
            if finite_count < all_logits.size(0):
                dropped = int(all_logits.size(0) - finite_count)
                print(
                    f"[WARNING] Dropping {dropped} non-finite replay candidates "
                    f"for task {task_id}."
                )
                all_images = all_images[finite_logits_mask]
                all_labels = all_labels[finite_logits_mask]
                all_logits = all_logits[finite_logits_mask]
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

            finite_soft_mask = torch.isfinite(soft_targets).all(dim=1)
            finite_count = int(finite_soft_mask.sum().item())
            if finite_count <= 0:
                print(
                    f"[WARNING] Replay update skipped for task {task_id}: "
                    "all computed soft targets are non-finite."
                )
                return
            if finite_count < soft_targets.size(0):
                dropped = int(soft_targets.size(0) - finite_count)
                print(
                    f"[WARNING] Dropping {dropped} non-finite fallback replay candidates "
                    f"for task {task_id}."
                )
                all_images = all_images[finite_soft_mask]
                all_labels = all_labels[finite_soft_mask]
                soft_targets = soft_targets[finite_soft_mask]
        
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
        if self._model_has_nonfinite_parameters():
            print(
                f"[WARNING] Skipping auto-distillation after task {task_id}: "
                "model has non-finite parameters."
            )
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

    def _post_prune_recovery(self, task_id: int, train_loader: DataLoader, prune_applied: bool):
        """Run a short recovery fine-tuning phase after pruning to stabilize accuracy."""
        if not prune_applied:
            return

        recovery_lr = max(self.learning_rate * self.prune_recovery_lr_scale, 1e-8)
        recovery_tmax = max(1, self.prune_recovery_epochs)

        self._refresh_training_state_after_prune(
            task_id=task_id,
            learning_rate=recovery_lr,
            scheduler_tmax=recovery_tmax
        )

        if not self.prune_recovery_enabled:
            return
        if self.prune_recovery_epochs <= 0:
            return

        print(f"\n--- Post-Prune Recovery (Task {task_id}) ---")
        print(
            f"Recovery epochs={self.prune_recovery_epochs}, "
            f"learning_rate={recovery_lr:.6g}"
        )

        for recovery_epoch in range(self.prune_recovery_epochs):
            display_epoch = self.epochs_per_task + recovery_epoch
            self._train_epoch(
                task_id=task_id,
                train_loader=train_loader,
                epoch=display_epoch,
                collection_dict=None
            )

        print("Post-prune recovery complete")

    def _collect_prunable_adapter_candidates(self) -> List[Dict]:
        """Collect removable adapter candidates with optional AMCA-aware Taylor scoring."""
        candidates = []
        use_taylor_scoring = (self.pruning_importance == 'taylor')

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if not hasattr(self.model, layer_name):
                continue
            layer = getattr(self.model, layer_name)
            for block_idx, block in enumerate(layer):
                adapters = getattr(block, 'adapters', None)
                if adapters is None or len(adapters) == 0:
                    continue

                for adapter_idx, adapter in enumerate(adapters):
                    adapter_id = id(adapter)
                    is_new_adapter = adapter_id in self.current_task_new_adapter_ids
                    if self.pruning_scope == 'older_adapters_only' and is_new_adapter:
                        continue

                    adapter_params = int(sum(p.numel() for p in adapter.parameters()))
                    if adapter_params <= 0:
                        continue

                    # Lower score means lower importance and thus higher prune priority.
                    score = 0.0
                    for p in adapter.parameters():
                        if use_taylor_scoring and p.grad is not None:
                            score += float((p.detach() * p.grad.detach()).abs().sum().item())
                        else:
                            score += float(p.detach().abs().sum().item())

                    candidates.append({
                        'layer_name': layer_name,
                        'block_idx': int(block_idx),
                        'adapter_idx': int(adapter_idx),
                        'adapter_params': adapter_params,
                        'score': score
                    })

        return candidates

    def _collect_prunable_dense_parameters(self) -> List[tuple]:
        """
        Collect Conv/Linear weights across the whole model for magnitude pruning.

        Current-task adapters are excluded so freshly introduced task adapters
        remain untouched at this task boundary.
        """
        protected_module_ids = set()
        if self.protect_current_task_adapters:
            for _, _, adapter in self._iter_adapters():
                if id(adapter) not in self.current_task_new_adapter_ids:
                    continue
                for m in adapter.modules():
                    protected_module_ids.add(id(m))

        modules_to_prune = []
        classifier_module = getattr(self.model, 'fc', None) if self.protect_classifier else None
        for module_name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if not hasattr(module, 'weight'):
                continue

            if classifier_module is not None and module is classifier_module:
                continue

            weight = getattr(module, 'weight', None)
            if weight is None:
                continue
            if hasattr(weight, 'requires_grad') and not bool(weight.requires_grad):
                # Skip frozen tensors; Taylor importance cannot score parameters
                # that do not receive gradients.
                continue

            if id(module) in protected_module_ids:
                continue

            if self.protect_classifier and module_name == 'fc':
                continue

            if self.protect_downsample and 'downsample' in module_name:
                continue

            modules_to_prune.append((module, 'weight'))

        return modules_to_prune

    def _get_effective_entries_for_parameter(self, module: nn.Module, param_name: str) -> int:
        """Count active entries for one potentially pruned parameter tensor."""
        orig_name = f"{param_name}_orig"
        mask_name = f"{param_name}_mask"

        if hasattr(module, orig_name) and hasattr(module, mask_name):
            mask = getattr(module, mask_name)
            return int(mask.sum().item())

        if hasattr(module, param_name):
            tensor = getattr(module, param_name)
            if isinstance(tensor, nn.Parameter):
                return int(tensor.numel())
            if torch.is_tensor(tensor):
                return int(tensor.numel())

        return 0

    def _count_prunable_dense_effective_params(self, modules_to_prune: List[tuple]) -> int:
        """Count active entries across all dense parameters eligible for pruning."""
        return int(
            sum(
                self._get_effective_entries_for_parameter(module, param_name)
                for module, param_name in modules_to_prune
            )
        )

    def _prepare_modules_for_unstructured_pruning(self, modules_to_prune: List[tuple]):
        """
        Ensure prunable tensors are contiguous so torch prune internals that use
        view() do not fail on channels-last/non-standard strides.
        """
        for module, param_name in modules_to_prune:
            for attr_name in (param_name, f"{param_name}_orig"):
                if not hasattr(module, attr_name):
                    continue

                attr = getattr(module, attr_name)
                if isinstance(attr, nn.Parameter):
                    attr.data = attr.data.contiguous()
                    continue

                if torch.is_tensor(attr):
                    setattr(module, attr_name, attr.contiguous())

    def _remove_pruning_reparam_if_present(
        self,
        modules_to_prune: List[tuple],
        keep_existing_reparam: Optional[set] = None
    ):
        """Best-effort cleanup of pruning reparameterization wrappers."""
        if keep_existing_reparam is None:
            keep_existing_reparam = set()

        for module, param_name in modules_to_prune:
            key = (id(module), param_name)
            if key in keep_existing_reparam:
                continue
            if not hasattr(module, f"{param_name}_orig"):
                continue
            try:
                torch_prune.remove(module, param_name)
            except Exception:
                # Keep cleanup best-effort; original prune exception is reported upstream.
                pass

    def _auto_structural_prune_whole_model(
        self,
        task_id: int,
        train_loader: Optional[DataLoader] = None
    ) -> bool:
        """Apply physical channel/filter pruning for whole-model mode."""
        if tp is None:
            print(
                "[WARNING] Structural pruning requested but 'torch_pruning' is not available. "
                "Install torch-pruning or use network_pruning.method='unstructured'."
            )
            return False

        modules_to_prune = self._collect_prunable_dense_parameters()
        if not modules_to_prune:
            print(
                f"No eligible whole-model parameters found at task {task_id} "
                "(current-task adapters are protected)."
            )
            return False

        prunable_effective = self._count_prunable_dense_effective_params(modules_to_prune)
        if prunable_effective <= 0:
            print(f"No active dense parameters available to prune at task {task_id}")
            return False

        prune_amount = self._resolve_budget_driven_prune_amount(
            prunable_effective_params=prunable_effective,
            task_id=task_id
        )
        if prune_amount <= 0.0:
            return False

        example_inputs = self._get_pruning_example_input(train_loader=train_loader)
        if example_inputs is None:
            print(f"[WARNING] No valid example input available for structural pruning at task {task_id}")
            return False

        ignored_layers = self._collect_structural_pruning_ignored_layers()
        before_effective = self._get_effective_model_params()
        before_total = int(self.model.get_num_parameters()) if hasattr(self.model, 'get_num_parameters') else int(sum(p.numel() for p in self.model.parameters()))
        pre_prune_snapshot = None
        try:
            pre_prune_snapshot = copy.deepcopy(self.model).cpu()
        except Exception as e:
            print(f"[WARNING] Could not snapshot model before structural pruning at task {task_id}: {e}")

        try:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(example_inputs)

            importance, importance_name = self._resolve_structural_importance(
                train_loader=train_loader
            )
            pruner = tp.pruner.MagnitudePruner(
                self.model,
                example_inputs=example_inputs,
                importance=importance,
                pruning_ratio=float(prune_amount),
                ignored_layers=ignored_layers,
            )

            try:
                pruner.step()
            except Exception as prune_error:
                if importance_name == 'taylor':
                    print(
                        f"[WARNING] Taylor structural pruning failed at task {task_id}: {prune_error}. "
                        "Retrying once with magnitude importance."
                    )

                    if pre_prune_snapshot is None:
                        raise

                    restored = copy.deepcopy(pre_prune_snapshot).to(self.device)
                    if self.device == 'cuda':
                        restored = restored.to(memory_format=torch.channels_last)
                    self.model = restored
                    self.train_model = self.model

                    self.model.eval()
                    with torch.no_grad():
                        _ = self.model(example_inputs)

                    fallback_importance = tp.importance.MagnitudeImportance(p=1)
                    pruner = tp.pruner.MagnitudePruner(
                        self.model,
                        example_inputs=example_inputs,
                        importance=fallback_importance,
                        pruning_ratio=float(prune_amount),
                        ignored_layers=ignored_layers,
                    )
                    pruner.step()
                    importance_name = 'magnitude_fallback'
                else:
                    raise

            if self.device == 'cuda':
                torch.cuda.empty_cache()

            after_effective = self._get_effective_model_params()
            after_total = int(self.model.get_num_parameters()) if hasattr(self.model, 'get_num_parameters') else int(sum(p.numel() for p in self.model.parameters()))
            pruned_now = max(0, before_effective - after_effective)
            total_removed = max(0, before_total - after_total)

            self.pruning_history.append({
                'task_id': int(task_id),
                'scope': self.pruning_scope,
                'method': 'structural',
                'amount': float(prune_amount),
                'importance': importance_name,
                'effective_params_before': int(before_effective),
                'effective_params_after': int(after_effective),
                'pruned_params': int(pruned_now),
                'total_params_before': int(before_total),
                'total_params_after': int(after_total),
                'total_removed_params': int(total_removed),
                'ignored_layers': int(len(ignored_layers)),
                'example_input_shape': list(example_inputs.shape),
                'taylor_replay_batches': int(self.last_taylor_gradient_stats.get('replay_batches', 0)),
                'taylor_train_batches': int(self.last_taylor_gradient_stats.get('train_batches', 0))
            })

            self.model.zero_grad(set_to_none=True)

            print(
                f"Auto-structural-pruned whole model at end of task {task_id}: "
                f"amount={prune_amount:.4f}, ignored_layers={len(ignored_layers)}, "
                f"importance={importance_name}, "
                f"effective params {before_effective:,} -> {after_effective:,}, "
                f"total params {before_total:,} -> {after_total:,} "
                f"(freed {pruned_now:,} effective / {total_removed:,} physical)"
            )
            return True
        except Exception as e:
            if pre_prune_snapshot is not None:
                try:
                    restored = pre_prune_snapshot.to(self.device)
                    if self.device == 'cuda':
                        restored = restored.to(memory_format=torch.channels_last)
                    self.model = restored
                    self.train_model = self.model
                    print(
                        f"[WARNING] Restored pre-prune snapshot after structural pruning failure "
                        f"at task {task_id}."
                    )
                except Exception as restore_error:
                    print(
                        f"[WARNING] Could not restore pre-prune snapshot at task {task_id}: "
                        f"{restore_error}"
                    )
            print(f"[WARNING] Structural whole-model pruning failed after task {task_id}: {e}")
            return False

    def _auto_prune_network_end_of_task(
        self,
        task_id: int,
        train_loader: Optional[DataLoader] = None
    ) -> bool:
        """Apply network pruning after replay-buffer distillation at each task boundary."""
        if not self.pruning_enabled:
            return False

        if task_id < self.pruning_warmup_tasks:
            print(
                f"Skipping pruning at task {task_id}: warmup active "
                f"(warmup_tasks={self.pruning_warmup_tasks})"
            )
            return False

        if not self._is_pruning_task(task_id):
            print(
                f"Skipping pruning at task {task_id}: interval active "
                f"(interval_tasks={self.pruning_interval_tasks})"
            )
            return False

        if self.pruning_scope == 'hybrid':
            print(f"\n--- Executing HYBRID pruning at task {task_id} ---")

            original_scope = self.pruning_scope
            original_method = self.pruning_method
            original_amount = self.pruning_amount
            original_budget_driven = self.pruning_budget_driven

            whole_model_success = False
            adapter_success = False
            try:
                # Step 1: Gentle global squeeze to counter cumulative expansion.
                self.pruning_scope = 'whole_model'
                self.pruning_method = 'structural'
                self.pruning_budget_driven = False
                self.pruning_amount = 0.015
                whole_model_success = self._auto_structural_prune_whole_model(
                    task_id=task_id,
                    train_loader=train_loader
                )

                expected_num_classes = int(self.config.get('benchmark', {}).get('total_classes', 7))
                self._ensure_classifier_head(expected_num_classes)

                # Step 2: Aggressive cleanup of stale historical adapters.
                self.pruning_scope = 'older_adapters_only'
                self.pruning_amount = 0.30
                adapter_success = self._auto_prune_network_end_of_task(
                    task_id=task_id,
                    train_loader=train_loader
                )
            finally:
                self.pruning_scope = original_scope
                self.pruning_method = original_method
                self.pruning_amount = original_amount
                self.pruning_budget_driven = original_budget_driven

            return bool(whole_model_success or adapter_success)

        if self.pruning_scope == 'whole_model' and self.pruning_method == 'structural':
            return self._auto_structural_prune_whole_model(
                task_id=task_id,
                train_loader=train_loader
            )

        if self.pruning_scope == 'whole_model':
            modules_to_prune = self._collect_prunable_dense_parameters()
            if not modules_to_prune:
                print(
                    f"No eligible whole-model parameters found at task {task_id} "
                    "(current-task adapters are protected)."
                )
                return False

            prunable_effective = self._count_prunable_dense_effective_params(modules_to_prune)
            if prunable_effective <= 0:
                print(f"No active dense parameters available to prune at task {task_id}")
                return False

            prune_amount = self._resolve_budget_driven_prune_amount(
                prunable_effective_params=prunable_effective,
                task_id=task_id
            )
            if prune_amount <= 0.0:
                return False

            before_effective = self._get_effective_model_params()
            preexisting_reparam = set()
            try:
                self._prepare_modules_for_unstructured_pruning(modules_to_prune)
                preexisting_reparam = {
                    (id(module), param_name)
                    for module, param_name in modules_to_prune
                    if hasattr(module, f"{param_name}_orig")
                }

                torch_prune.global_unstructured(
                    modules_to_prune,
                    pruning_method=torch_prune.L1Unstructured,
                    amount=prune_amount
                )

                # Keep pruning masks so zeroed entries remain inactive across tasks.
                masked_modules = sum(
                    1 for module, param_name in modules_to_prune
                    if hasattr(module, f"{param_name}_orig")
                )

                after_effective = self._get_effective_model_params()
                pruned_now = max(0, before_effective - after_effective)

                self.pruning_history.append({
                    'task_id': int(task_id),
                    'scope': self.pruning_scope,
                    'method': 'unstructured',
                    'amount': float(prune_amount),
                    'effective_params_before': int(before_effective),
                    'effective_params_after': int(after_effective),
                    'pruned_params': int(pruned_now),
                    'pruned_modules': int(len(modules_to_prune)),
                    'masked_modules': int(masked_modules)
                })

                print(
                    f"Auto-pruned whole model at end of task {task_id}: "
                    f"amount={prune_amount:.4f}, modules={len(modules_to_prune)}, "
                    f"effective params {before_effective:,} -> {after_effective:,} "
                    f"(freed {pruned_now:,})"
                )
                return True
            except Exception as e:
                self._remove_pruning_reparam_if_present(
                    modules_to_prune,
                    keep_existing_reparam=preexisting_reparam
                )
                print(f"[WARNING] Whole-model pruning failed after task {task_id}: {e}")
                return False

        if self.pruning_scope not in {'older_adapters_only', 'all_adapters'}:
            print(
                f"[WARNING] Unsupported pruning scope '{self.pruning_scope}'. "
                "Falling back to 'whole_model'."
            )
            self.pruning_scope = 'whole_model'
            return self._auto_prune_network_end_of_task(task_id, train_loader=train_loader)

        adapter_grad_state = []
        try:
            if self.pruning_importance == 'taylor' and self.pruning_scope == 'older_adapters_only':
                print("Populating Replay gradients to evaluate Adapter AMCA utility...")

                # Adapter pruning happens after training where historical adapters are
                # often frozen. Temporarily enable grads so Taylor can score them.
                for _, _, adapter in self._iter_adapters():
                    is_new_adapter = id(adapter) in self.current_task_new_adapter_ids
                    if self.pruning_scope == 'older_adapters_only' and is_new_adapter:
                        continue
                    for p in adapter.parameters():
                        adapter_grad_state.append((p, bool(p.requires_grad)))
                        p.requires_grad = True

                taylor_ready = self._prepare_taylor_gradients_for_pruning(train_loader=train_loader)
                if not taylor_ready:
                    print(
                        "[WARNING] Could not prepare Taylor gradients for adapter pruning; "
                        "falling back to magnitude scoring."
                    )

            candidates = self._collect_prunable_adapter_candidates()
            if not candidates:
                print(
                    f"No removable adapters found for scope '{self.pruning_scope}' at task {task_id} "
                    "(current-task adapters are protected)."
                )
                return False

            before_effective = self._get_effective_model_params()
            total_candidate_params = int(sum(c['adapter_params'] for c in candidates))
            if total_candidate_params <= 0:
                print(f"No removable adapter parameters available at task {task_id}")
                return False

            prune_amount = self._resolve_budget_driven_prune_amount(
                prunable_effective_params=total_candidate_params,
                task_id=task_id
            )
            if prune_amount <= 0.0:
                return False

            target_prune_params = int(total_candidate_params * prune_amount)
            if prune_amount > 0.0:
                target_prune_params = max(1, target_prune_params)

            candidates_sorted = sorted(candidates, key=lambda c: c['score'])
            selected = []
            removed_param_budget = 0
            for candidate in candidates_sorted:
                if removed_param_budget >= target_prune_params:
                    break
                selected.append(candidate)
                removed_param_budget += int(candidate['adapter_params'])

            if not selected:
                print(f"No adapters selected for pruning at task {task_id}")
                return False

            try:
                indices_by_block = {}
                for item in selected:
                    key = (item['layer_name'], item['block_idx'])
                    if key not in indices_by_block:
                        indices_by_block[key] = []
                    indices_by_block[key].append(int(item['adapter_idx']))

                removed_params = 0
                removed_adapters = 0

                for (layer_name, block_idx), adapter_indices in indices_by_block.items():
                    layer = getattr(self.model, layer_name)
                    block = layer[block_idx]
                    adapters = getattr(block, 'adapters', None)
                    if adapters is None or len(adapters) == 0:
                        continue

                    keep_adapters = []
                    remove_set = set(adapter_indices)
                    for idx, adapter in enumerate(adapters):
                        if idx in remove_set:
                            removed_adapters += 1
                            removed_params += int(sum(p.numel() for p in adapter.parameters()))
                        else:
                            keep_adapters.append(adapter)

                    block.adapters = nn.ModuleList(keep_adapters)

                after_effective = self._get_effective_model_params()
                pruned_now = max(0, before_effective - after_effective)

                # Keep tracking set aligned after structural deletions.
                self.current_task_new_adapter_ids = {
                    adapter_id for adapter_id in self.current_task_new_adapter_ids
                    if any(id(adapter) == adapter_id for _, _, adapter in self._iter_adapters())
                }

                self.pruning_history.append({
                    'task_id': int(task_id),
                    'scope': self.pruning_scope,
                    'method': 'adapter_delete',
                    'amount': float(prune_amount),
                    'importance': ('taylor' if self.pruning_importance == 'taylor' else 'magnitude'),
                    'effective_params_before': int(before_effective),
                    'effective_params_after': int(after_effective),
                    'pruned_params': int(pruned_now),
                    'removed_params': int(removed_params),
                    'removed_adapters': int(removed_adapters),
                    'candidate_params': int(total_candidate_params),
                    'target_prune_params': int(target_prune_params)
                })

                print(
                    f"Auto-pruned at end of task {task_id}: "
                    f"scope={self.pruning_scope}, amount={prune_amount:.4f}, "
                    f"effective params {before_effective:,} -> {after_effective:,} "
                    f"(freed {pruned_now:,}, removed_adapters={removed_adapters}, "
                    f"removed_params={removed_params:,})"
                )
                return True
            except Exception as e:
                print(f"[WARNING] Auto network pruning failed after task {task_id}: {e}")
                return False
        finally:
            for param, original_requires_grad in adapter_grad_state:
                param.requires_grad = original_requires_grad
            self.model.zero_grad(set_to_none=True)

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

        self.metrics.update_num_parameters(self._get_effective_model_params())
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
        configured = int(self.config.get('benchmark', {}).get('total_classes', 0))
        if configured > 0:
            return configured

        max_label = -1
        for _, batch_labels in dataloader:
            if torch.is_tensor(batch_labels) and batch_labels.numel() > 0:
                max_label = max(max_label, int(batch_labels.max().item()))
            if max_label >= 100:
                break

        return max_label + 1 if max_label >= 0 else 1

    def save_checkpoint(self, path: str, task_id: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'task_id': task_id,
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'config': self.config,
            'task_expansions': self.task_expansions,
            'expansion_history': self.expansion_history,
            'pruning_history': self.pruning_history
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint['metrics']
        self.task_expansions = checkpoint.get('task_expansions', {})
        self.expansion_history = checkpoint.get('expansion_history', [])
        self.pruning_history = checkpoint.get('pruning_history', [])

def create_trainer(config: Dict, model, replay_buffer, device='cuda', use_amca=False):
    return ContinualLearner(
        model=model,
        replay_buffer=replay_buffer,
        config=config,
        device=device,
        use_amca=use_amca
    )
