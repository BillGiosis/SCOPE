import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Optional
import time
import os

try:
    import torch_pruning as tp
except ImportError:
    tp = None

from models.expandable_resnet import ExpandableResNet
from buffers.replay_buffer import ReplayBuffer
from utils.metrics import (
    MetricsTracker,
    AMCATester,
    evaluate_model,
    evaluate_model_with_groups,
    compute_model_size
)
from utils.losses import ClassBalancedLoss
from trainers.reallocation_policy import (
    PROTECT_CLASSIFIER,
    PROTECT_CURRENT_TASK_ADAPTERS,
    PROTECT_DOWNSAMPLE,
    RECOMPILE_AFTER_REALLOCATION,
)


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
        
        if self.device == 'cuda':
            self.model = model.to(device, memory_format=torch.channels_last)
        else:
            self.model = model.to(device)
            
        self.replay_buffer = replay_buffer
        self.config = config
        self.use_amca = use_amca
        
        self.use_amp = config['training'].get('use_amp', False) and (device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp, init_scale=2048.0)
        
        self.use_torch_compile = config['training'].get('use_torch_compile', True)
        self.train_model = self.model

        self.metrics = MetricsTracker(
            num_tasks=config['benchmark']['num_tasks']
        )
        self.metric_groups = self._resolve_metric_groups(config.get('benchmark', {}).get('metric_groups', {}))
        
        if use_amca:
            num_classes = config['benchmark'].get('total_classes', 7)
            self.amca_tester = AMCATester(num_classes=num_classes)
        else:
            self.amca_tester = None
        
        self.epochs_per_task = config['training']['epochs_per_task']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training']['optimizer']
        self.weight_decay = config['training']['weight_decay']

        expansion_cfg = config.get('network_expansion', {})
        self.expansion_enabled = expansion_cfg.get('enabled', True)
        self.expansion_strategy = str(expansion_cfg.get('strategy', 'all_layers')).strip().lower()
        self.expansion_layer_order = self._resolve_expansion_layers(self.expansion_strategy)

        reallocation_cfg = config.get('parameter_reallocation', {})
        self.reallocation_enabled = reallocation_cfg.get('enabled', False)
        self.reallocation_amount = float(reallocation_cfg.get('amount', 0.0))
        self.reallocation_scope = str(reallocation_cfg.get('scope', 'whole_model')).strip().lower()
        self.reallocation_importance = str(reallocation_cfg.get('importance', 'magnitude')).strip().lower()
        self.reallocation_use_replay_for_taylor = bool(reallocation_cfg.get('use_replay_for_taylor', True))
        self.reallocation_taylor_replay_batches = max(0, int(reallocation_cfg.get('taylor_replay_batches', 2)))
        self.reallocation_taylor_train_batches = max(0, int(reallocation_cfg.get('taylor_train_batches', 1)))
        self.reallocation_taylor_replay_batch_size = max(
            1,
            int(
                reallocation_cfg.get(
                    'taylor_replay_batch_size',
                    config.get('replay_buffer', {}).get('replay_batch_size', 32)
                )
            )
        )
        self.reallocation_warmup_tasks = max(0, int(reallocation_cfg.get('warmup_tasks', 2)))
        hybrid_cfg = reallocation_cfg.get('hybrid', {})
        self.hybrid_whole_model_amount = float(hybrid_cfg.get('whole_model_amount', 0.01))
        self.hybrid_adapter_amount = float(hybrid_cfg.get('adapter_amount', 0.20))
        self.hybrid_adapter_scope = str(
            hybrid_cfg.get('adapter_scope', 'older_adapters_only')
        ).strip().lower()
        if (
            self.use_torch_compile
            and self.reallocation_enabled
            and self.reallocation_scope in {'whole_model', 'hybrid'}
        ):
            try:
                import torch._dynamo as dynamo
                dynamo.config.force_parameter_static_shapes = False
                print(
                    "Enabled dynamic parameter shapes for torch.compile "
                    "(structural reallocation is active)."
                )
            except Exception as e:
                print(
                    "[WARNING] Could not enable dynamic parameter shapes for torch.compile: "
                    f"{e}"
                )

        recovery_cfg = reallocation_cfg.get('recovery', {})
        self.reallocation_recovery_enabled = bool(recovery_cfg.get('enabled', True))
        self.reallocation_recovery_epochs = max(0, int(recovery_cfg.get('epochs', 4)))
        self.reallocation_recovery_lr_scale = float(recovery_cfg.get('learning_rate_scale', 0.2))

        self.reallocation_amount = min(max(self.reallocation_amount, 0.0), 1.0)
        self.hybrid_whole_model_amount = min(max(self.hybrid_whole_model_amount, 0.0), 1.0)
        self.hybrid_adapter_amount = min(max(self.hybrid_adapter_amount, 0.0), 1.0)
        if self.hybrid_adapter_scope not in {'older_adapters_only', 'all_adapters'}:
            print(
                f"[WARNING] Unsupported parameter_reallocation.hybrid.adapter_scope='{self.hybrid_adapter_scope}'. "
                "Falling back to 'older_adapters_only'."
            )
            self.hybrid_adapter_scope = 'older_adapters_only'
        if self.reallocation_importance not in {'magnitude', 'taylor'}:
            print(
                f"[WARNING] Unsupported parameter_reallocation.importance='{self.reallocation_importance}'. "
                "Falling back to 'magnitude'."
            )
            self.reallocation_importance = 'magnitude'
        self.reallocation_recovery_lr_scale = max(self.reallocation_recovery_lr_scale, 0.0)

        base_params = int(self._get_effective_model_params())
        max_ratio = float(config.get('benchmark', {}).get('max_params_ratio', 1.05))
        self.max_params = int(base_params * max_ratio)
        
        self.optimizer = None
        self.criterion = None
        
        self.task_expansions = {}
        self.expansion_history = []
        self.reallocation_history = []
        self.current_task_new_adapter_ids = set()
        self.last_taylor_gradient_stats = {
            'replay_batches': 0,
            'train_batches': 0,
            'total_batches': 0
        }
        self.last_good_model_snapshot: Optional[nn.Module] = None
        self.last_good_task_id: Optional[int] = None
        self._cached_eval_loaders: Dict[int, DataLoader] = {}
        self._cached_group_ids: Dict[int, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _task_label(task_id: int) -> int:
        """Return the user-facing 1-indexed task number."""
        return int(task_id) + 1

    def _is_reallocation_task(self, task_id: int) -> bool:
        """Return whether reallocation should run for this task id after warmup."""
        if not self.reallocation_enabled:
            return False
        return task_id >= self.reallocation_warmup_tasks

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

        return list(self.expansion_layer_order)

    def _get_effective_model_params(self) -> int:
        """
        Count active model parameters for budget checks.

        When reallocation masks are present, masked-out parameters are excluded.
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

    def _resolve_reallocation_amount(
        self,
        reallocatable_effective_params: int,
        task_id: int
    ) -> float:
        """Resolve reallocation amount with fixed (non-budget-driven) behavior."""
        del task_id

        configured_amount = min(max(float(self.reallocation_amount), 0.0), 1.0)
        if configured_amount <= 0.0:
            return 0.0

        if reallocatable_effective_params <= 0:
            return 0.0

        return configured_amount

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

        print(f"\n--- Task-Start Network Expansion (Task {self._task_label(task_id)}) ---")
        targets = self._get_task_expansion_targets(task_id)
        if not targets:
            print("No expansion targets configured")
            self.task_expansions[task_id] = executed
            return

        for layer_name in targets:
            if self._try_expand_layer(layer_name):
                executed.append(layer_name)
            else:
                print(f"No eligible expansion executed for {layer_name} at task {self._task_label(task_id)}.")

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

    def _apply_adapter_training_modes(self):
        """Keep frozen historical adapters, including their BN buffers, fixed."""
        for _, _, adapter in self._iter_adapters():
            is_new_adapter = id(adapter) in self.current_task_new_adapter_ids
            adapter.train(bool(is_new_adapter))

    def _get_reallocation_example_input(self, train_loader: Optional[DataLoader] = None) -> Optional[torch.Tensor]:
        """Build example input for graph-based structural reallocation."""
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
                print(f"[WARNING] Could not derive reallocation example input from loader: {e}")

        try:
            # Normal path uses one sample from train_loader; fallback matches
            # the fixed model input size used by the current benchmarks.
            shape = (1, 3, 224, 224)
            example = torch.randn(*shape, device=self.device)
            if self.device == 'cuda':
                example = example.contiguous(memory_format=torch.channels_last)
            return example
        except Exception as e:
            print(f"[WARNING] Could not create fallback reallocation example input: {e}")
            return None

    def _collect_structural_reallocation_ignored_layers(self) -> List[nn.Module]:
        """Collect strict protection list for structural reallocation."""
        ignored_layers = []
        ignored_ids = set()

        if PROTECT_CLASSIFIER:
            classifier_module = getattr(self.model, 'fc', None)
            if classifier_module is not None and id(classifier_module) not in ignored_ids:
                ignored_layers.append(classifier_module)
                ignored_ids.add(id(classifier_module))

        for module_name, module in self.model.named_modules():
            should_ignore = False

            if PROTECT_CLASSIFIER and module_name == 'fc':
                should_ignore = True

            if PROTECT_DOWNSAMPLE and 'downsample' in module_name:
                should_ignore = True

            if should_ignore and id(module) not in ignored_ids:
                ignored_layers.append(module)
                ignored_ids.add(id(module))

        if PROTECT_CURRENT_TASK_ADAPTERS:
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
            print(f"[WARNING] Could not snapshot last-good model at task {self._task_label(task_id)}: {e}")
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
                f"Restored last-good model snapshot from task {self._task_label(self.last_good_task_id)} "
                f"before starting task {self._task_label(task_id)}"
            )
            return True
        except Exception as e:
            print(f"[WARNING] Could not restore last-good model snapshot: {e}")
            return False

    def _prepare_taylor_gradients_for_reallocation(
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
        use_distillation_loss = bool(
            self.config.get('replay_buffer', {}).get('use_distillation_loss', True)
        )

        if (
            self.reallocation_use_replay_for_taylor
            and self.reallocation_taylor_replay_batches > 0
            and len(self.replay_buffer) > 0
        ):
            replay_batch_size = min(self.reallocation_taylor_replay_batch_size, len(self.replay_buffer))
            for _ in range(self.reallocation_taylor_replay_batches):
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

                if use_distillation_loss and replay_batch.get('soft_targets', None) is not None:
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

        if train_loader is not None and self.reallocation_taylor_train_batches > 0:
            for batch_idx, (images, labels) in enumerate(train_loader):
                if train_batches_used >= self.reallocation_taylor_train_batches:
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

                if train_batches_used >= self.reallocation_taylor_train_batches:
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
                # importance never receives None gradients from the structural dependency.
                parameter.grad = torch.zeros_like(parameter)

        print(
            "Prepared Taylor gradients for structural reallocation: "
            f"replay_batches={replay_batches_used}, "
            f"train_batches={train_batches_used}"
        )
        return True

    def _resolve_structural_importance(
        self,
        train_loader: Optional[DataLoader] = None
    ):
        """Resolve structural reallocation importance criterion from config."""
        if tp is None:
            raise RuntimeError("The structural reallocation dependency is required for importance resolution")

        if self.reallocation_importance != 'taylor':
            self.last_taylor_gradient_stats = {
                'replay_batches': 0,
                'train_batches': 0,
                'total_batches': 0
            }
            return tp.importance.MagnitudeImportance(p=1), 'magnitude'

        taylor_ready = self._prepare_taylor_gradients_for_reallocation(train_loader=train_loader)
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
                "[WARNING] Structural reallocation Taylor importance class not found. "
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

    def _refresh_training_state_after_reallocation(
        self,
        task_id: int,
        learning_rate: Optional[float] = None,
        scheduler_tmax: Optional[int] = None
    ):
        """Refresh optimizer and compiled training graph after structural reallocation."""
        expected_num_classes = int(self.config.get('benchmark', {}).get('total_classes', 7))
        self._ensure_classifier_head(expected_num_classes)

        self._create_optimizer(
            learning_rate=learning_rate,
            scheduler_tmax=scheduler_tmax
        )
        print(f"Rebuilt optimizer after reallocation for task {self._task_label(task_id)}")

        self.train_model = self.model
        if RECOMPILE_AFTER_REALLOCATION and self.use_torch_compile:
            try:
                try:
                    import torch._dynamo as dynamo
                    dynamo.reset()
                except Exception:
                    pass
                self.train_model = torch.compile(self.model)
                print(f"Recompiled training model after reallocation for task {self._task_label(task_id)}")
            except Exception as e:
                print(f"Could not recompile training model after reallocation for task {self._task_label(task_id)}: {e}")
                self.train_model = self.model

    def _prepare_training_model_for_task(self, task_id: int):
        """Compile a training-only model once per task after expansion."""
        self.train_model = self.model

        if not self.use_torch_compile:
            print("torch.compile disabled; using eager model for training")
            return

        try:
            self.train_model = torch.compile(self.model)
            print(f"Compiled training model for task {self._task_label(task_id)}")
        except Exception as e:
            print(f"Could not compile training model for task {self._task_label(task_id)}: {e}")
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
        optimizer_name = self.optimizer_name.lower()
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if optimizer_name == 'adamw':
            use_fused = (self.device == 'cuda')
            extra_args = {}
            if use_fused:
                extra_args['fused'] = True

            self.optimizer = optim.AdamW(
                trainable_parameters,
                lr=lr,
                weight_decay=self.weight_decay,
                eps=1e-8,
                **extra_args
            )
        elif optimizer_name == 'sgd':
            momentum = float(self.config.get('training', {}).get('momentum', 0.9))
            nesterov = bool(self.config.get('training', {}).get('nesterov', False))
            self.optimizer = optim.SGD(
                trainable_parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=self.weight_decay,
                nesterov=nesterov
            )
        else:
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer_name}'. Supported optimizers: adamw, sgd."
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
        print(f"Training Task {self._task_label(task_id)}")
        print(f"{'='*60}\n")
        
        if not check_model_health(self.model, f"Task {self._task_label(task_id)} start"):
            restored = self._restore_last_good_model(task_id)
            if not restored:
                raise RuntimeError(
                    f"Model corrupted before Task {self._task_label(task_id)}, and no healthy snapshot is available. "
                    "Stopping to avoid poisoning replay and metrics."
                )
            if not check_model_health(self.model, f"Task {self._task_label(task_id)} restored-start"):
                raise RuntimeError(
                    f"Model remains corrupted after restore attempt before Task {self._task_label(task_id)}."
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
        self._apply_adapter_training_modes()

        self._prepare_training_model_for_task(task_id)
        
        self._create_optimizer()
        
        training_cfg = self.config.get('training', {})
        use_class_balanced_loss = bool(training_cfg.get('use_class_balanced_loss', True))
        if use_class_balanced_loss:
            print("Calculating class counts for Class Balanced Loss")
        else:
            print("Class Balanced Loss disabled; using Cross Entropy Loss")
        num_classes = int(
            getattr(self.model.fc, 'out_features', self.config['benchmark'].get('total_classes', 7))
        )
        max_label_observed = -1

        if hasattr(train_loader.dataset, 'targets'):
            all_labels = train_loader.dataset.targets

            if torch.is_tensor(all_labels):
                label_tensor = all_labels.detach().to(dtype=torch.long).cpu()
            else:
                if isinstance(all_labels, list) and len(all_labels) > 0 and torch.is_tensor(all_labels[0]):
                    all_labels = [int(l.item()) for l in all_labels]
                label_tensor = torch.as_tensor(all_labels, dtype=torch.long)

            if label_tensor.numel() > 0:
                max_label_observed = int(label_tensor.max().item())
                non_negative_labels = label_tensor[label_tensor >= 0]
                if non_negative_labels.numel() > 0:
                    bincount = torch.bincount(non_negative_labels, minlength=num_classes)
                    class_counts = bincount[:num_classes].tolist()
                else:
                    class_counts = [0 for _ in range(num_classes)]
            else:
                class_counts = [0 for _ in range(num_classes)]
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
                f"Label range mismatch at task {self._task_label(task_id)}: observed max label {max_label_observed}, "
                f"but classifier outputs only {num_classes} classes."
            )
        
        benchmark_name = str(self.config.get('benchmark', {}).get('name', '')).strip().lower()
        if use_class_balanced_loss:
            self.criterion = ClassBalancedLoss(
                class_counts=class_counts,
                beta=training_cfg.get('cb_beta', 0.9999)
            ).to(self.device)

            if benchmark_name == 'clad-c' and hasattr(self.criterion, 'weights') and self.criterion.weights.numel() > 0:
                self.criterion.weights[0] = 0.0
        else:
            self.criterion = None
            
        all_test_datasets = []
        for tid in range(task_id + 1):
            if tid < len(test_loaders) and test_loaders[tid] is not None:
                all_test_datasets.append(test_loaders[tid].dataset)
        
        cached_test_loader = None
        cached_group_ids = {}
        if all_test_datasets:
            if len(all_test_datasets) == 1:
                combined_test_dataset = all_test_datasets[0]
            else:
                combined_test_dataset = ConcatDataset(all_test_datasets)

            data_loading_config = self.config.get('data_loading', {})
            num_workers = min(data_loading_config.get('num_workers', 4), 8)

            cache_key = id(combined_test_dataset)
            if cache_key not in self._cached_eval_loaders:
                self._cached_eval_loaders[cache_key] = DataLoader(
                    combined_test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=data_loading_config.get('pin_memory', True),
                    prefetch_factor=2 if num_workers > 0 else None,
                    persistent_workers=True if num_workers > 0 else False
                )
            cached_test_loader = self._cached_eval_loaders[cache_key]

            if cache_key not in self._cached_group_ids:
                self._cached_group_ids[cache_key] = {}
            for group_name, group_names in self.metric_groups.items():
                if group_name not in self._cached_group_ids[cache_key]:
                    self._cached_group_ids[cache_key][group_name] = self._build_group_id_tensor(
                        combined_test_dataset=combined_test_dataset,
                        group_name=group_name,
                        group_names=group_names
                    )
                cached_group_ids[group_name] = self._cached_group_ids[cache_key][group_name]

        replay_candidates = {'images': [], 'labels': [], 'logits': []}
        last_eval_snapshot = None
        from tqdm import tqdm

        task_total_steps = max(1, int(self.epochs_per_task))
        task_progress_bar = tqdm(
            total=task_total_steps,
            desc=f"Task {self._task_label(task_id)} Training",
            unit='epoch',
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'
        )
        
        try:
            for epoch in range(self.epochs_per_task):
                is_last_epoch = (epoch == self.epochs_per_task - 1)
                collection_dict = replay_candidates if is_last_epoch else None

                epoch_loss, epoch_acc = self._train_epoch(
                    task_id,
                    train_loader,
                    epoch,
                    collection_dict,
                    task_progress_bar=task_progress_bar,
                )
                task_progress_bar.set_postfix(
                    {
                        'loss': f'{epoch_loss:.4f}',
                        'acc': f'{epoch_acc:.2f}%'
                    },
                    refresh=True
                )
                task_progress_bar.update(1)
                
                # Evaluate only at task end. When reallocation can mutate the model
                # after the epoch, defer metric recording until the final model
                # state is known.
                if is_last_epoch and not self._is_reallocation_task(task_id):
                    print(f"\nEpoch {epoch + 1}/{self.epochs_per_task} - Evaluating")
                    last_eval_snapshot = self._evaluate_and_update_metrics(
                        task_id,
                        cached_test_loader,
                        group_id_tensors=cached_group_ids,
                        run_amca=False
                    )
        finally:
            task_progress_bar.close()
            
        replay_enabled = self.config.get('replay_buffer', {}).get('enabled', True)
        if replay_enabled:
            if len(replay_candidates['images']) > 0:
                self._update_replay_buffer_from_memory(task_id, replay_candidates)
            else:
                self._update_replay_buffer(task_id, train_loader)

        self._auto_distill_replay_buffer_end_of_task(task_id)

        reallocation_applied = self._auto_parameter_reallocation_end_of_task(task_id, train_loader=train_loader)
        self._post_reallocation_recovery(task_id, train_loader, reallocation_applied)

        if not reallocation_applied and last_eval_snapshot is not None:
            print(
                f"\nFinal evaluation after task {self._task_label(task_id)}: "
                "reusing latest epoch metrics (no post-train model change)."
            )
            if self.use_amca and self.amca_tester:
                try:
                    per_class_acc = last_eval_snapshot.get('per_class_acc', {})
                    overall_accuracy = float(last_eval_snapshot.get('overall_accuracy', 0.0))
                    if per_class_acc:
                        amca_results = self.amca_tester.record_from_per_class_dict(
                            per_class_acc,
                            overall_accuracy=overall_accuracy
                        )
                    else:
                        amca_results = self.amca_tester.evaluate(self.model, cached_test_loader, device=self.device)
                    print(
                        f"MCA: {amca_results['mean_class_accuracy']:.4f}, "
                        f"AMCA: {self.amca_tester.compute_amca():.4f}"
                    )
                except Exception as e:
                    print(f"AMCA error: {e}")
                    self.amca_tester.mean_class_accuracies.append(0.0)
        else:
            print(f"\nFinal evaluation after task {self._task_label(task_id)}:")
            self._evaluate_and_update_metrics(
                task_id,
                cached_test_loader,
                group_id_tensors=cached_group_ids,
                run_amca=True
            )

        task_elapsed_time = time.time() - task_start_time
        self.metrics.record_task_time(task_id, task_elapsed_time)
        self.metrics.update_compute_cost(task_elapsed_time)
        print(f"\nTask {self._task_label(task_id)} compute time: {task_elapsed_time:.2f} seconds\n")
        
        self.metrics.print_summary(task_id)
        if self.use_amca and self.amca_tester is not None:
            self.amca_tester.print_summary()

        if self._model_has_nonfinite_parameters():
            print(
                f"[WARNING] Task {self._task_label(task_id)} ended with non-finite parameters; "
                "last-good snapshot not updated."
            )
        else:
            self._snapshot_last_good_model(task_id)
    
    def _train_epoch(
        self,
        task_id: int,
        train_loader: DataLoader,
        epoch: int,
        collection_dict: Optional[Dict] = None,
        task_progress_bar=None,
    ) -> tuple:
        """Train for one epoch with Batched Replay and Logit Capture."""
        from tqdm import tqdm
        
        self.train_model.train()
        self._apply_adapter_training_modes()

        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        total = 0
        
        training_config = self.config.get('training', {})
        replay_config = self.config.get('replay_buffer', {})
        use_gradient_accumulation = training_config.get('use_gradient_accumulation', False)
        accumulation_steps = max(1, int(training_config.get('accumulation_steps', 1))) if use_gradient_accumulation else 1
        progress_update_interval = max(1, int(training_config.get('progress_update_interval', 10)))
        max_nonfinite_batches = max(
            1,
            int(training_config.get('max_nonfinite_batches_per_epoch', 5))
        )
        use_cuda = (self.device == 'cuda')
        replay_enabled = bool(replay_config.get('enabled', True))
        replay_batch_size = max(1, int(replay_config.get('replay_batch_size', 32)))
        replay_interval_batches = max(1, int(replay_config.get('replay_interval_batches', 1)))
        use_distillation_loss = bool(replay_config.get('use_distillation_loss', True))
        replay_buffer_len = len(self.replay_buffer) if replay_enabled else 0
        has_replay_data = replay_enabled and replay_buffer_len > 0
        criterion = self.criterion
        has_class_balanced_loss = criterion is not None
        distill_temp = max(float(self.replay_buffer.distillation_temp), 1e-6)
        train_loader_len = len(train_loader)
        train_forward = self.train_model
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        manage_local_pbar = (task_progress_bar is None)
        if manage_local_pbar:
            pbar = tqdm(
                enumerate(train_loader),
                total=train_loader_len,
                desc=f"Epoch {epoch + 1}",
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'
            )
            batch_iterator = pbar
        else:
            pbar = task_progress_bar
            batch_iterator = enumerate(train_loader)

        num_output_classes = int(
            getattr(
                self.model.fc,
                'out_features',
                self.config.get('benchmark', {}).get('total_classes', 7)
            )
        )
        if has_class_balanced_loss and hasattr(criterion, 'weights'):
            weight_count = int(criterion.weights.numel())
            if weight_count != num_output_classes:
                raise RuntimeError(
                    f"Class weight mismatch at task {self._task_label(task_id)}: criterion has {weight_count} weights "
                    f"but model outputs {num_output_classes} classes."
                )

        nonfinite_batches = 0
        has_pending_grad = False
        
        self.optimizer.zero_grad(set_to_none=True)
        
        collected_count = 0
        num_classes = int(self.config['benchmark'].get('total_classes', 7))
        benchmark_name = str(self.config.get('benchmark', {}).get('name', '')).strip().lower()
        buffer_max_size = int(getattr(self.replay_buffer, 'max_size', replay_config.get('max_size', 1000)))
        auto_collection_multiplier = float(replay_config.get('collection_multiplier', 1.5))
        auto_collection_limit = max(
            buffer_max_size,
            int(round(buffer_max_size * max(1.0, auto_collection_multiplier)))
        )
        explicit_samples_per_class = replay_config.get('samples_per_class', None)
        if explicit_samples_per_class is not None:
            samples_per_class_limit = max(1, int(explicit_samples_per_class))
            collection_limit = samples_per_class_limit * num_classes
        elif benchmark_name == 'bdd-c':
            collection_limit = auto_collection_limit
            samples_per_class_limit = max(1, (collection_limit + max(1, num_classes) - 1) // max(1, num_classes))
        else:
            samples_per_class_limit = 200
            collection_limit = samples_per_class_limit * num_classes
        collection_enabled = collection_dict is not None and collection_limit > 0
        class_capture_remaining = torch.full(
            (num_classes,),
            samples_per_class_limit,
            device=self.device,
            dtype=torch.long
        )

        for batch_idx, (images, labels) in batch_iterator:
            try:
                if use_cuda:
                    images = images.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )
                else:
                    images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                
                labels = labels.to(self.device, non_blocking=True)
                
                capture_indices = None
                if collection_enabled and collected_count < collection_limit:
                    remaining_slots = int(collection_limit - collected_count)
                    valid_mask = (labels >= 0) & (labels < num_classes)
                    if bool(valid_mask.any().item()) and remaining_slots > 0:
                        valid_positions = torch.nonzero(valid_mask, as_tuple=True)[0]
                        valid_labels = labels.index_select(0, valid_positions)
                        selected_chunks = []

                        for class_tensor in valid_labels.unique(sorted=False):
                            class_id = int(class_tensor.item())
                            remaining_for_class = int(class_capture_remaining[class_id].item())
                            if remaining_for_class <= 0:
                                continue

                            class_positions = valid_positions[valid_labels == class_tensor]
                            if class_positions.numel() == 0:
                                continue

                            take_count = min(
                                int(class_positions.numel()),
                                remaining_for_class,
                                remaining_slots
                            )
                            if take_count <= 0:
                                continue

                            selected_chunks.append(class_positions[:take_count])
                            remaining_slots -= take_count
                            if remaining_slots <= 0:
                                break

                        if selected_chunks:
                            capture_indices = torch.cat(selected_chunks, dim=0)

                replay_batch = None
                r_imgs = None
                r_lbls = None
                r_out = None
                use_replay_this_batch = (
                    has_replay_data
                    and ((batch_idx + 1) % replay_interval_batches == 0)
                )

                if use_replay_this_batch:
                    replay_batch = self.replay_buffer.get_batch(
                        batch_size=min(replay_batch_size, replay_buffer_len)
                    )

                    if replay_batch is not None:
                        if use_cuda:
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

                if r_imgs is not None:
                    combined_images = torch.cat((images, r_imgs), dim=0)
                    current_batch_size = int(images.size(0))
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        combined_outputs = train_forward(combined_images)
                    outputs = combined_outputs[:current_batch_size]
                    r_out = combined_outputs[current_batch_size:]
                else:
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        outputs = train_forward(images)
                
                if capture_indices is not None and capture_indices.numel() > 0:
                    selected_images = images.index_select(0, capture_indices).detach().cpu()
                    selected_labels = labels.index_select(0, capture_indices)
                    selected_logits = outputs.index_select(0, capture_indices).detach().cpu()
                    selected_labels_cpu = selected_labels.detach().cpu()

                    collection_dict['images'].append(selected_images)
                    collection_dict['labels'].append(selected_labels_cpu)
                    collection_dict['logits'].append(selected_logits)

                    if num_classes > 0:
                        captured_counts = torch.bincount(selected_labels, minlength=num_classes)
                        class_capture_remaining.sub_(captured_counts[:num_classes])
                        class_capture_remaining.clamp_(min=0)

                    collected_count += int(selected_labels.size(0))

                outputs_float = outputs.float()

                if has_class_balanced_loss:
                    loss = criterion(outputs_float, labels)
                else:
                    loss = F.cross_entropy(outputs_float, labels)
            
                if replay_batch is not None and r_imgs is not None and r_lbls is not None:
                    if r_out is None:
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            r_out = train_forward(r_imgs)

                    loss += 0.5 * F.cross_entropy(r_out.float(), r_lbls)

                    if use_distillation_loss and replay_batch['soft_targets'] is not None:
                        soft_targets = replay_batch['soft_targets'].to(self.device, non_blocking=True).float()

                        # Re-use replay logits from the fused/current replay forward when dimensions match.
                        if r_out.size(1) == soft_targets.size(1):
                            r_out_dist = r_out
                        else:
                            with torch.amp.autocast('cuda', enabled=self.use_amp):
                                r_out_dist = train_forward(r_imgs)

                        log_probs = F.log_softmax(r_out_dist.float() / distill_temp, dim=1)
                        loss += 0.3 * (distill_temp ** 2) * F.kl_div(log_probs, soft_targets, reduction='batchmean')

                if not torch.isfinite(loss).all():
                    nonfinite_batches += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    has_pending_grad = False
                    print(
                        f"\n[WARNING] Non-finite loss at task {self._task_label(task_id)}, epoch {epoch + 1}, "
                        f"batch {batch_idx}; skipped ({nonfinite_batches}/{max_nonfinite_batches})"
                    )
                    if nonfinite_batches >= max_nonfinite_batches:
                        raise RuntimeError(
                            f"Too many non-finite losses in task {self._task_label(task_id)}, epoch {epoch + 1}. "
                            "Stopping to prevent model corruption."
                        )
                    continue

                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                has_pending_grad = True
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if has_pending_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
            
            if torch.is_tensor(loss):
                total_loss += loss.detach().float() * accumulation_steps
            _, predicted = outputs_float.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().detach()
            
            if total > 0 and (
                (batch_idx + 1) % progress_update_interval == 0
                or (batch_idx + 1) == train_loader_len
            ):
                if manage_local_pbar:
                    pbar.set_postfix({
                        'loss': f'{float(total_loss.item()) / (batch_idx + 1):.4f}',
                        'acc': f'{100.0 * float(correct.item()) / total:.2f}%'
                    })
        
        if has_pending_grad:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        if manage_local_pbar:
            pbar.close()

        epoch_avg_loss = float(total_loss.item() / max(1, train_loader_len))
        epoch_acc = float(100.0 * correct.item() / max(1, total))
        return epoch_avg_loss, epoch_acc

    def _update_replay_buffer_from_memory(self, task_id: int, collected_data: Dict):
        """Optimized update using pre-computed logits."""
        if not collected_data['images']:
            print("No samples collected for replay buffer!")
            return

        print(f"\nUpdating replay buffer for task {self._task_label(task_id)} (from memory)")
        
        all_images = torch.cat(collected_data['images'], dim=0)
        all_labels = torch.cat(collected_data['labels'], dim=0)
        soft_targets = None
        use_soft_targets = bool(self.config.get('replay_buffer', {}).get('use_soft_targets', True))

        if use_soft_targets and 'logits' in collected_data and len(collected_data['logits']) > 0:
            all_logits = torch.cat(collected_data['logits'], dim=0)
            finite_logits_mask = torch.isfinite(all_logits).all(dim=1)
            finite_count = int(finite_logits_mask.sum().item())
            if finite_count <= 0:
                print(
                    f"[WARNING] Replay update skipped for task {self._task_label(task_id)}: "
                    "all cached logits are non-finite."
                )
                return
            if finite_count < all_logits.size(0):
                dropped = int(all_logits.size(0) - finite_count)
                print(
                    f"[WARNING] Dropping {dropped} non-finite replay candidates "
                    f"for task {self._task_label(task_id)}."
                )
                all_images = all_images[finite_logits_mask]
                all_labels = all_labels[finite_logits_mask]
                all_logits = all_logits[finite_logits_mask]
            soft_targets = F.softmax(all_logits.float() / self.replay_buffer.distillation_temp, dim=1)
        elif use_soft_targets:
            print("Computing logits (fallback)")
            module_training_modes = {module: module.training for module in self.model.modules()}
            self.model.eval()
            soft_targets_list = []
            batch_size = 128
            
            try:
                with torch.inference_mode():
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
            finally:
                for module, was_training in module_training_modes.items():
                    module.train(was_training)
            soft_targets = torch.cat(soft_targets_list, dim=0)

            finite_soft_mask = torch.isfinite(soft_targets).all(dim=1)
            finite_count = int(finite_soft_mask.sum().item())
            if finite_count <= 0:
                print(
                    f"[WARNING] Replay update skipped for task {self._task_label(task_id)}: "
                    "all computed soft targets are non-finite."
                )
                return
            if finite_count < soft_targets.size(0):
                dropped = int(soft_targets.size(0) - finite_count)
                print(
                    f"[WARNING] Dropping {dropped} non-finite fallback replay candidates "
                    f"for task {self._task_label(task_id)}."
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
            print(f"Replay buffer empty after task {self._task_label(task_id)}; skipping auto-distillation")
            return
        if self._model_has_nonfinite_parameters():
            print(
                f"[WARNING] Skipping auto-distillation after task {self._task_label(task_id)}: "
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
            print(f"Auto-distilled replay buffer at end of task {self._task_label(task_id)}")
        except Exception as e:
            print(f"[WARNING] Auto replay-buffer distillation failed after task {self._task_label(task_id)}: {e}")

    def _post_reallocation_recovery(self, task_id: int, train_loader: DataLoader, reallocation_applied: bool):
        """Run a short recovery fine-tuning phase after reallocation to stabilize accuracy."""
        if not reallocation_applied:
            return

        recovery_lr = max(self.learning_rate * self.reallocation_recovery_lr_scale, 1e-8)
        recovery_tmax = max(1, self.reallocation_recovery_epochs)

        self._refresh_training_state_after_reallocation(
            task_id=task_id,
            learning_rate=recovery_lr,
            scheduler_tmax=recovery_tmax
        )

        if not self.reallocation_recovery_enabled:
            return
        if self.reallocation_recovery_epochs <= 0:
            return

        print(f"\n--- Post-Reallocation Recovery (Task {self._task_label(task_id)}) ---")
        print(
            f"Recovery epochs={self.reallocation_recovery_epochs}, "
            f"learning_rate={recovery_lr:.6g}"
        )

        for recovery_epoch in range(self.reallocation_recovery_epochs):
            display_epoch = self.epochs_per_task + recovery_epoch
            self._train_epoch(
                task_id=task_id,
                train_loader=train_loader,
                epoch=display_epoch,
                collection_dict=None
            )

        print("Post-reallocation recovery complete")

    def _collect_reallocatable_adapter_candidates(self) -> List[Dict]:
        """Collect removable adapter candidates with optional AMCA-aware Taylor scoring."""
        candidates = []
        use_taylor_scoring = (self.reallocation_importance == 'taylor')

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
                    if PROTECT_CURRENT_TASK_ADAPTERS and is_new_adapter:
                        continue
                    if self.reallocation_scope == 'older_adapters_only' and is_new_adapter:
                        continue

                    adapter_params = int(sum(p.numel() for p in adapter.parameters()))
                    if adapter_params <= 0:
                        continue

                    # Lower score means lower importance and thus higher reallocation priority.
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

    def _collect_reallocatable_dense_parameters(self) -> List[tuple]:
        """
        Collect Conv/Linear weights across the whole model for magnitude reallocation.

        Current-task adapters are excluded so freshly introduced task adapters
        remain untouched at this task boundary.
        """
        protected_module_ids = set()
        if PROTECT_CURRENT_TASK_ADAPTERS:
            for _, _, adapter in self._iter_adapters():
                if id(adapter) not in self.current_task_new_adapter_ids:
                    continue
                for m in adapter.modules():
                    protected_module_ids.add(id(m))

        modules_to_reallocate = []
        classifier_module = getattr(self.model, 'fc', None) if PROTECT_CLASSIFIER else None
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

            if PROTECT_CLASSIFIER and module_name == 'fc':
                continue

            if PROTECT_DOWNSAMPLE and 'downsample' in module_name:
                continue

            modules_to_reallocate.append((module, 'weight'))

        return modules_to_reallocate

    def _get_effective_entries_for_parameter(self, module: nn.Module, param_name: str) -> int:
        """Count active entries for one potentially reallocated parameter tensor."""
        if hasattr(module, param_name):
            tensor = getattr(module, param_name)
            if isinstance(tensor, nn.Parameter):
                return int(tensor.numel())
            if torch.is_tensor(tensor):
                return int(tensor.numel())

        return 0

    def _count_reallocatable_dense_effective_params(self, modules_to_reallocate: List[tuple]) -> int:
        """Count active entries across all dense parameters eligible for reallocation."""
        return int(
            sum(
                self._get_effective_entries_for_parameter(module, param_name)
                for module, param_name in modules_to_reallocate
            )
        )

    def _auto_structural_reallocation_whole_model(
        self,
        task_id: int,
        train_loader: Optional[DataLoader] = None
    ) -> bool:
        """Apply physical channel/filter reallocation for whole-model mode."""
        if tp is None:
            print(
                "[WARNING] Structural reallocation requested but the dependency is not available. "
                "Install the structural dependency to enable whole-model reallocation."
            )
            return False

        modules_to_reallocate = self._collect_reallocatable_dense_parameters()
        if not modules_to_reallocate:
            print(
                f"No eligible whole-model parameters found at task {self._task_label(task_id)} "
                "(current-task adapters are protected)."
            )
            return False

        reallocatable_effective = self._count_reallocatable_dense_effective_params(modules_to_reallocate)
        if reallocatable_effective <= 0:
            print(f"No active dense parameters available for reallocation at task {self._task_label(task_id)}")
            return False

        reallocation_amount = self._resolve_reallocation_amount(
            reallocatable_effective_params=reallocatable_effective,
            task_id=task_id
        )
        if reallocation_amount <= 0.0:
            return False

        example_inputs = self._get_reallocation_example_input(train_loader=train_loader)
        if example_inputs is None:
            print(f"[WARNING] No valid example input available for structural reallocation at task {self._task_label(task_id)}")
            return False

        ignored_layers = self._collect_structural_reallocation_ignored_layers()
        before_effective = self._get_effective_model_params()
        before_total = int(self.model.get_num_parameters()) if hasattr(self.model, 'get_num_parameters') else int(sum(p.numel() for p in self.model.parameters()))
        pre_reallocation_snapshot = None
        try:
            pre_reallocation_snapshot = copy.deepcopy(self.model).cpu()
        except Exception as e:
            print(f"[WARNING] Could not snapshot model before structural reallocation at task {self._task_label(task_id)}: {e}")

        try:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(example_inputs)

            importance, importance_name = self._resolve_structural_importance(
                train_loader=train_loader
            )
            reallocator = tp.pruner.MagnitudePruner(
                self.model,
                example_inputs=example_inputs,
                importance=importance,
                pruning_ratio=float(reallocation_amount),
                ignored_layers=ignored_layers,
            )

            try:
                reallocator.step()
            except Exception as reallocation_error:
                if importance_name == 'taylor':
                    print(
                        f"[WARNING] Taylor structural reallocation failed at task {self._task_label(task_id)}: {reallocation_error}. "
                        "Retrying once with magnitude importance."
                    )

                    if pre_reallocation_snapshot is None:
                        raise

                    restored = copy.deepcopy(pre_reallocation_snapshot).to(self.device)
                    if self.device == 'cuda':
                        restored = restored.to(memory_format=torch.channels_last)
                    self.model = restored
                    self.train_model = self.model

                    self.model.eval()
                    with torch.no_grad():
                        _ = self.model(example_inputs)

                    fallback_importance = tp.importance.MagnitudeImportance(p=1)
                    reallocator = tp.pruner.MagnitudePruner(
                        self.model,
                        example_inputs=example_inputs,
                        importance=fallback_importance,
                        pruning_ratio=float(reallocation_amount),
                        ignored_layers=ignored_layers,
                    )
                    reallocator.step()
                    importance_name = 'magnitude_fallback'
                else:
                    raise

            if self.device == 'cuda':
                torch.cuda.empty_cache()

            after_effective = self._get_effective_model_params()
            after_total = int(self.model.get_num_parameters()) if hasattr(self.model, 'get_num_parameters') else int(sum(p.numel() for p in self.model.parameters()))
            capacity_before = int(before_total)
            capacity_after = int(after_total)
            reallocated_now = max(0, capacity_before - capacity_after)

            self.reallocation_history.append({
                'task_id': int(task_id),
                'scope': self.reallocation_scope,
                'method': 'structural',
                'amount': float(reallocation_amount),
                'importance': importance_name,
                'capacity_params_before': int(capacity_before),
                'capacity_params_after': int(capacity_after),
                'effective_params_before': int(before_effective),
                'effective_params_after': int(after_effective),
                'reallocated_params': int(reallocated_now),
                'total_params_before': int(before_total),
                'total_params_after': int(after_total),
                'ignored_layers': int(len(ignored_layers)),
                'example_input_shape': list(example_inputs.shape),
                'taylor_replay_batches': int(self.last_taylor_gradient_stats.get('replay_batches', 0)),
                'taylor_train_batches': int(self.last_taylor_gradient_stats.get('train_batches', 0))
            })

            self.model.zero_grad(set_to_none=True)

            print(
                f"Auto-structural-reallocated whole model at end of task {self._task_label(task_id)}: "
                f"amount={reallocation_amount:.4f}, ignored_layers={len(ignored_layers)}, "
                f"importance={importance_name}, "
                f"capacity params {capacity_before:,} -> {capacity_after:,} "
                f"(freed {reallocated_now:,})"
            )
            return True
        except Exception as e:
            if pre_reallocation_snapshot is not None:
                try:
                    restored = pre_reallocation_snapshot.to(self.device)
                    if self.device == 'cuda':
                        restored = restored.to(memory_format=torch.channels_last)
                    self.model = restored
                    self.train_model = self.model
                    print(
                        f"[WARNING] Restored pre-reallocation snapshot after structural reallocation failure "
                        f"at task {self._task_label(task_id)}."
                    )
                except Exception as restore_error:
                    print(
                        f"[WARNING] Could not restore pre-reallocation snapshot at task {self._task_label(task_id)}: "
                        f"{restore_error}"
                    )
            print(f"[WARNING] Structural whole-model reallocation failed after task {self._task_label(task_id)}: {e}")
            return False

    @staticmethod
    def _combine_hybrid_reallocation_entries(task_id: int, entries: List[Dict]) -> Dict:
        """Collapse the two internal hybrid reallocation steps into one task-level record."""
        if not entries:
            return {
                'task_id': int(task_id),
                'scope': 'hybrid',
                'method': 'hybrid',
                'amount': 0.0,
                'reallocated_params': 0
            }

        first = entries[0]
        last = entries[-1]
        whole_model_entries = [entry for entry in entries if entry.get('scope') == 'whole_model']
        adapter_entries = [
            entry for entry in entries
            if entry.get('scope') in {'older_adapters_only', 'all_adapters'}
        ]
        importances = sorted({
            str(entry.get('importance'))
            for entry in entries
            if entry.get('importance') not in {None, 'n/a'}
        })

        combined = {
            'task_id': int(task_id),
            'scope': 'hybrid',
            'method': 'hybrid',
            'amount': float(last.get('amount', 0.0)),
            'importance': '+'.join(importances) if importances else 'n/a',
            'capacity_params_before': int(first.get('capacity_params_before', first.get('effective_params_before', 0))),
            'capacity_params_after': int(last.get('capacity_params_after', last.get('effective_params_after', 0))),
            'effective_params_before': int(first.get('effective_params_before', first.get('capacity_params_before', 0))),
            'effective_params_after': int(last.get('effective_params_after', last.get('capacity_params_after', 0))),
            'reallocated_params': int(sum(int(entry.get('reallocated_params', 0)) for entry in entries)),
            'components': entries,
            'component_count': int(len(entries)),
            'whole_model_reallocated_params': int(sum(int(entry.get('reallocated_params', 0)) for entry in whole_model_entries)),
            'adapter_reallocated_params': int(sum(int(entry.get('reallocated_params', 0)) for entry in adapter_entries)),
            'removed_adapters': int(sum(int(entry.get('removed_adapters', 0)) for entry in adapter_entries)),
            'whole_model_amount': float(whole_model_entries[-1].get('amount', 0.0)) if whole_model_entries else 0.0,
            'adapter_amount': float(adapter_entries[-1].get('amount', 0.0)) if adapter_entries else 0.0,
            'adapter_scope': str(adapter_entries[-1].get('scope', 'n/a')) if adapter_entries else 'n/a',
        }

        for key in (
            'taylor_replay_batches',
            'taylor_train_batches',
        ):
            values = [entry.get(key) for entry in entries if key in entry]
            if values:
                combined[key] = values[-1]

        return combined

    def _auto_parameter_reallocation_end_of_task(
        self,
        task_id: int,
        train_loader: Optional[DataLoader] = None
    ) -> bool:
        """Apply parameter reallocation after replay-buffer distillation at each task boundary."""
        if not self.reallocation_enabled:
            return False

        if task_id < self.reallocation_warmup_tasks:
            print(
                f"Skipping reallocation at task {self._task_label(task_id)}: warmup active "
                f"(warmup_tasks={self.reallocation_warmup_tasks})"
            )
            return False

        if self.reallocation_scope == 'hybrid':
            print(f"\n--- Executing HYBRID reallocation at task {self._task_label(task_id)} ---")

            original_scope = self.reallocation_scope
            original_amount = self.reallocation_amount
            history_start = len(self.reallocation_history)

            whole_model_success = False
            adapter_success = False
            try:
                self.reallocation_scope = 'whole_model'
                self.reallocation_amount = self.hybrid_whole_model_amount
                whole_model_success = self._auto_structural_reallocation_whole_model(
                    task_id=task_id,
                    train_loader=train_loader
                )

                expected_num_classes = int(self.config.get('benchmark', {}).get('total_classes', 7))
                self._ensure_classifier_head(expected_num_classes)

                self.reallocation_scope = self.hybrid_adapter_scope
                self.reallocation_amount = self.hybrid_adapter_amount
                adapter_success = self._auto_parameter_reallocation_end_of_task(
                    task_id=task_id,
                    train_loader=train_loader
                )
            finally:
                self.reallocation_scope = original_scope
                self.reallocation_amount = original_amount

            new_entries = self.reallocation_history[history_start:]
            if new_entries:
                combined_entry = self._combine_hybrid_reallocation_entries(task_id, new_entries)
                del self.reallocation_history[history_start:]
                self.reallocation_history.append(combined_entry)

            return bool(whole_model_success or adapter_success)

        if self.reallocation_scope == 'whole_model':
            return self._auto_structural_reallocation_whole_model(
                task_id=task_id,
                train_loader=train_loader
            )

        if self.reallocation_scope not in {'older_adapters_only', 'all_adapters'}:
            print(
                f"[WARNING] Unsupported reallocation scope '{self.reallocation_scope}'. "
                "Falling back to 'whole_model'."
            )
            self.reallocation_scope = 'whole_model'
            return self._auto_parameter_reallocation_end_of_task(task_id, train_loader=train_loader)

        adapter_grad_state = []
        try:
            if self.reallocation_importance == 'taylor' and self.reallocation_scope in {'older_adapters_only', 'all_adapters'}:
                print("Populating Replay gradients to evaluate Adapter AMCA utility...")

                # Adapter reallocation happens after training where historical adapters are
                # often frozen. Temporarily enable grads so Taylor can score them.
                for _, _, adapter in self._iter_adapters():
                    is_new_adapter = id(adapter) in self.current_task_new_adapter_ids
                    if is_new_adapter and (
                        PROTECT_CURRENT_TASK_ADAPTERS
                        or self.reallocation_scope == 'older_adapters_only'
                    ):
                        continue
                    for p in adapter.parameters():
                        adapter_grad_state.append((p, bool(p.requires_grad)))
                        p.requires_grad = True

                taylor_ready = self._prepare_taylor_gradients_for_reallocation(train_loader=train_loader)
                if not taylor_ready:
                    print(
                        "[WARNING] Could not prepare Taylor gradients for adapter reallocation; "
                        "falling back to magnitude scoring."
                    )

            candidates = self._collect_reallocatable_adapter_candidates()
            if not candidates:
                print(
                    f"No removable adapters found for scope '{self.reallocation_scope}' at task {self._task_label(task_id)} "
                    "(current-task adapters are protected)."
                )
                return False

            before_effective = self._get_effective_model_params()
            total_candidate_params = int(sum(c['adapter_params'] for c in candidates))
            if total_candidate_params <= 0:
                print(f"No removable adapter parameters available at task {self._task_label(task_id)}")
                return False

            reallocation_amount = self._resolve_reallocation_amount(
                reallocatable_effective_params=total_candidate_params,
                task_id=task_id
            )
            if reallocation_amount <= 0.0:
                return False

            target_reallocation_params = int(total_candidate_params * reallocation_amount)
            if reallocation_amount > 0.0:
                target_reallocation_params = max(1, target_reallocation_params)

            candidates_sorted = sorted(candidates, key=lambda c: c['score'])
            selected = []
            removed_param_budget = 0
            for candidate in candidates_sorted:
                if removed_param_budget >= target_reallocation_params:
                    break
                selected.append(candidate)
                removed_param_budget += int(candidate['adapter_params'])

            if not selected:
                print(f"No adapters selected for reallocation at task {self._task_label(task_id)}")
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
                capacity_before = int(before_effective)
                capacity_after = int(after_effective)
                reallocated_now = max(0, capacity_before - capacity_after)

                # Keep tracking set aligned after structural deletions.
                alive_adapter_ids = {id(adapter) for _, _, adapter in self._iter_adapters()}
                self.current_task_new_adapter_ids &= alive_adapter_ids

                self.reallocation_history.append({
                    'task_id': int(task_id),
                    'scope': self.reallocation_scope,
                    'method': 'adapter_delete',
                    'amount': float(reallocation_amount),
                    'importance': ('taylor' if self.reallocation_importance == 'taylor' else 'magnitude'),
                    'capacity_params_before': int(capacity_before),
                    'capacity_params_after': int(capacity_after),
                    'effective_params_before': int(before_effective),
                    'effective_params_after': int(after_effective),
                    'reallocated_params': int(reallocated_now),
                    'removed_params': int(removed_params),
                    'removed_adapters': int(removed_adapters),
                    'candidate_params': int(total_candidate_params),
                    'target_reallocation_params': int(target_reallocation_params)
                })

                print(
                    f"Auto-reallocated at end of task {self._task_label(task_id)}: "
                    f"scope={self.reallocation_scope}, amount={reallocation_amount:.4f}, "
                    f"capacity params {capacity_before:,} -> {capacity_after:,} "
                    f"(freed {reallocated_now:,}, removed_adapters={removed_adapters}, "
                    f"removed_params={removed_params:,})"
                )
                return True
            except Exception as e:
                print(f"[WARNING] Auto parameter reallocation failed after task {self._task_label(task_id)}: {e}")
                return False
        finally:
            for param, original_requires_grad in adapter_grad_state:
                param.requires_grad = original_requires_grad
            self.model.zero_grad(set_to_none=True)

    @staticmethod
    def _resolve_metric_groups(metric_groups_cfg: Dict) -> Dict[str, List[str]]:
        """Normalize benchmark metric-group config to {group_name: [display values]}."""
        resolved = {}
        for group_name, group_cfg in (metric_groups_cfg or {}).items():
            names = []
            if isinstance(group_cfg, dict):
                names = group_cfg.get('names', []) or []
            elif isinstance(group_cfg, list):
                names = group_cfg
            if names:
                resolved[str(group_name)] = [str(name) for name in names]
        if not resolved:
            resolved['location'] = ['Citystreet', 'Countryroad', 'Highway']
        return resolved

    def _resolve_group_value_for_dataset_index(self, dataset, idx: int, group_name: str):
        """Resolve a configured group value from wrapped/concat/subset datasets."""
        if hasattr(dataset, 'get_group_value'):
            return dataset.get_group_value(idx, group_name)

        if group_name == 'location' and hasattr(dataset, 'get_location'):
            return dataset.get_location(idx)

        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            parent_idx = dataset.indices[idx]
            return self._resolve_group_value_for_dataset_index(dataset.dataset, parent_idx, group_name)

        if hasattr(dataset, 'datasets'):
            sample_idx = idx
            for d in dataset.datasets:
                if sample_idx < len(d):
                    return self._resolve_group_value_for_dataset_index(d, sample_idx, group_name)
                sample_idx -= len(d)

        return None

    def _build_group_id_tensor(
        self,
        combined_test_dataset,
        group_name: str,
        group_names: List[str]
    ) -> torch.Tensor:
        """Create a dense group-id tensor aligned with dataset order."""
        group_to_id = {name: idx for idx, name in enumerate(group_names)}
        group_ids = torch.full((len(combined_test_dataset),), -1, dtype=torch.long)
        for idx in range(len(combined_test_dataset)):
            group_value = self._resolve_group_value_for_dataset_index(
                combined_test_dataset,
                idx,
                group_name
            )
            group_ids[idx] = int(group_to_id.get(group_value, -1))
        return group_ids

    def _evaluate_and_update_metrics(
        self,
        task_id: int,
        test_loader: DataLoader,
        group_id_tensors: Optional[Dict[str, torch.Tensor]] = None,
        run_amca: bool = True
    ):
        if test_loader is None:
            return None

        overall_accuracy = 0.0
        per_class_acc = {}

        try:
            if group_id_tensors:
                eval_results = evaluate_model_with_groups(
                    self.model,
                    test_loader,
                    task_id=task_id,
                    device=self.device,
                    group_id_tensors=group_id_tensors,
                    group_names=self.metric_groups
                )
                overall_accuracy = float(eval_results.get('accuracy', 0.0))
                per_class_acc = eval_results.get('per_class_acc', {}) or {}
                self.metrics.update_accuracy(task_id, overall_accuracy, per_class_acc)

                grouped_acc = eval_results.get('group_accuracy', {}) or {}
                grouped_per_class = eval_results.get('group_per_class_acc', {}) or {}
                for group_name, group_values in grouped_acc.items():
                    per_class_by_value = grouped_per_class.get(group_name, {}) or {}
                    for group_value, value_acc in group_values.items():
                        self.metrics.update_group_accuracy(
                            group_name=group_name,
                            group_value=group_value,
                            task_id=task_id,
                            accuracy=float(value_acc),
                            class_accuracies=per_class_by_value.get(group_value, {})
                        )
            else:
                overall_accuracy, per_class_acc = evaluate_model(
                    self.model,
                    test_loader,
                    task_id=task_id,
                    device=self.device,
                    return_per_class=True
                )
                self.metrics.update_accuracy(task_id, overall_accuracy, per_class_acc)
            
            if self.use_amca and self.amca_tester:
                if run_amca:
                    try:
                        # Reuse per-class results from standard evaluation to avoid
                        # a redundant second full forward pass over the test loader.
                        if per_class_acc:
                            amca_results = self.amca_tester.record_from_per_class_dict(
                                per_class_acc,
                                overall_accuracy=overall_accuracy
                            )
                        else:
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
            return None
        
        model_stats = compute_model_size(self.model)
        buffer_memory = self.replay_buffer.get_memory_size()
        self.metrics.update_memory(model_stats['total_mb'] + buffer_memory)

        self.metrics.update_num_parameters(self._get_effective_model_params())
        return {
            'overall_accuracy': float(overall_accuracy),
            'per_class_acc': per_class_acc
        }

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

            adapt_downsample_blocks = bool(getattr(block, 'adapt_downsample_blocks', True))
            if not adapt_downsample_blocks and (in_channels != out_channels or stride != 1):
                continue

            hidden = max(1, in_channels // 16)
            delta += (in_channels * hidden) + (hidden * out_channels)

        return int(delta)

    def save_checkpoint(self, path: str, task_id: int):
        checkpoint_dir = os.path.dirname(path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        tmp_path = f"{path}.tmp"
        torch.save({
            'task_id': task_id,
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'config': self.config,
            'task_expansions': self.task_expansions,
            'expansion_history': self.expansion_history,
            'reallocation_history': self.reallocation_history
        }, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint['metrics']
        self.task_expansions = checkpoint.get('task_expansions', {})
        self.expansion_history = checkpoint.get('expansion_history', [])
        self.reallocation_history = checkpoint.get('reallocation_history', [])


def create_trainer(config: Dict, model, replay_buffer, device='cuda', use_amca=False):
    return ContinualLearner(
        model=model,
        replay_buffer=replay_buffer,
        config=config,
        device=device,
        use_amca=use_amca
    )
