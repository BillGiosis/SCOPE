"""
Metrics and evaluation utilities for continual learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import time


class AMCATester:
    """
    AMCA (Average Mean Class Accuracy) Tester for CLAD-C benchmark.
    """
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Total number of classes in the dataset
        """
        self.num_classes = num_classes
        self.class_accuracies_over_time = []  # List of per-class accuracies at each eval
        self.mean_class_accuracies = []  # MCA at each evaluation point
        
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: str = 'cuda',
        show_progress: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate model and compute per-class accuracies using vectorized operations.
        """
        model.eval()
        use_cuda = str(device).startswith('cuda')
        
        max_cls = max(self.num_classes, self.num_classes + 1)
        class_correct_tensor = torch.zeros(max_cls, device=device, dtype=torch.long)
        class_total_tensor = torch.zeros(max_cls, device=device, dtype=torch.long)

        with torch.inference_mode():
            iterator = dataloader
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(dataloader, desc="AMCA Evaluation", leave=False)

            for images, labels in iterator:
                if use_cuda:
                    images = images.to(
                        device,
                        dtype=torch.float32,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )
                else:
                    images = images.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Standard forward pass
                outputs = model(images)

                _, predicted = outputs.max(1)

                # Vectorized per-class counting
                # bincount requires 1D non-negative
                class_total_tensor += torch.bincount(labels, minlength=max_cls)[:max_cls]

                correct_mask = (predicted == labels)
                class_correct_tensor += torch.bincount(labels[correct_mask], minlength=max_cls)[:max_cls]
        
        # Move final stats to CPU once at the end
        class_correct = class_correct_tensor.cpu().numpy()
        class_total = class_total_tensor.cpu().numpy()
        
        # Compute per-class accuracies
        class_accuracies = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if c < len(class_total) and class_total[c] > 0:
                class_accuracies[c] = class_correct[c] / class_total[c]
            else:
                class_accuracies[c] = 0.0
        
        # Store results
        self.class_accuracies_over_time.append(class_accuracies.copy())
        
        # Compute Mean Class Accuracy (MCA) for this evaluation
        # Only consider classes that have samples
        valid_classes_mask = class_total[:self.num_classes] > 0
        if valid_classes_mask.sum() > 0:
            mca = class_accuracies[valid_classes_mask].mean()
        else:
            mca = 0.0
        self.mean_class_accuracies.append(mca)
        
        overall_acc = class_correct.sum() / class_total.sum() if class_total.sum() > 0 else 0.0
        
        return {
            'class_accuracies': class_accuracies,
            'mean_class_accuracy': mca,
            'overall_accuracy': overall_acc
        }
    
    def compute_amca(self) -> float:
        if len(self.mean_class_accuracies) == 0:
            return 0.0
        return float(np.mean(self.mean_class_accuracies))

    def record_from_per_class_dict(
        self,
        per_class_acc: Dict[int, float],
        overall_accuracy: Optional[float] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Record AMCA statistics from an existing per-class accuracy dictionary."""
        class_accuracies = np.zeros(self.num_classes, dtype=np.float64)
        valid_mask = np.zeros(self.num_classes, dtype=bool)

        for class_id, acc in per_class_acc.items():
            if 0 <= int(class_id) < self.num_classes:
                class_accuracies[int(class_id)] = float(acc)
                valid_mask[int(class_id)] = True

        self.class_accuracies_over_time.append(class_accuracies.copy())

        if valid_mask.any():
            mca = float(class_accuracies[valid_mask].mean())
        else:
            mca = 0.0
        self.mean_class_accuracies.append(mca)

        if overall_accuracy is None:
            overall_accuracy = 0.0

        return {
            'class_accuracies': class_accuracies,
            'mean_class_accuracy': float(mca),
            'overall_accuracy': float(overall_accuracy)
        }
    
    def get_summary(self) -> Dict:
        return {
            'amca': self.compute_amca(),
            'num_evaluations': len(self.mean_class_accuracies),
            'mean_class_accuracies': self.mean_class_accuracies,
            'final_mca': self.mean_class_accuracies[-1] if self.mean_class_accuracies else 0.0
        }
    
    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "="*60)
        print("AMCA (Average Mean Class Accuracy) Summary")
        print("="*60)
        print(f"AMCA Score:           {summary['amca']:.4f}")
        print(f"Final MCA:            {summary['final_mca']:.4f}")
        print(f"Number of Evals:      {summary['num_evaluations']}")
        print("\nMCA at each evaluation:")
        for i, mca in enumerate(summary['mean_class_accuracies']):
            print(f"  Eval {i}: {mca:.4f}")
        print("="*60 + "\n")
    
    def reset(self):
        self.class_accuracies_over_time = []
        self.mean_class_accuracies = []


class MetricsTracker:
    """
    Track various metrics during continual learning.
    """
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        
        # Accuracy tracking: task_id -> list of accuracies over time
        self.task_accuracies = defaultdict(list)
        
        # Per-class accuracy tracking: task_id -> {class_id: accuracy}
        self.task_class_accuracies = {}  # Latest per-class accuracy for each task

        # Per-location tracking:
        # task_id -> {location: overall_accuracy}
        self.task_group_accuracies = defaultdict(lambda: defaultdict(dict))
        # task_id -> {location: {class_id: accuracy}}
        self.task_group_class_accuracies = defaultdict(lambda: defaultdict(dict))
        # location -> [mca over evaluation points]
        self.group_mca_history = defaultdict(lambda: defaultdict(list))
        # location -> [overall accuracy over evaluation points]
        self.group_overall_history = defaultdict(lambda: defaultdict(list))

        # Backward-compatible location aliases for CLAD-C code paths.
        self.task_location_accuracies = self.task_group_accuracies['location']
        self.task_location_class_accuracies = self.task_group_class_accuracies['location']
        self.location_mca_history = self.group_mca_history['location']
        self.location_overall_history = self.group_overall_history['location']
        
        # Best accuracy achieved for each task
        self.best_task_accuracies = {}
        
        # Memory and compute tracking
        self.memory_usage = []  # In MB
        self.compute_costs = []  # In seconds
        self.num_parameters = []
        
        # Task training times
        self.task_times = {}
        
    def update_accuracy(self, task_id: int, accuracy: float, class_accuracies: Optional[Dict[int, float]] = None):
        """Record accuracy for a specific task."""
        self.task_accuracies[task_id].append(accuracy)
        
        # Update per-class accuracies if provided
        if class_accuracies is not None:
            self.task_class_accuracies[task_id] = class_accuracies
        
        # Update best accuracy
        if task_id not in self.best_task_accuracies:
            self.best_task_accuracies[task_id] = accuracy
        else:
            self.best_task_accuracies[task_id] = max(
                self.best_task_accuracies[task_id], accuracy
            )

    def update_group_accuracy(
        self,
        group_name: str,
        group_value: str,
        task_id: int,
        accuracy: float,
        class_accuracies: Optional[Dict[int, float]] = None
    ):
        """Record grouped overall and per-class accuracy."""
        self.task_group_accuracies[group_name][task_id][group_value] = accuracy

        if class_accuracies is None:
            class_accuracies = {}
        self.task_group_class_accuracies[group_name][task_id][group_value] = class_accuracies

        if class_accuracies:
            mca = float(np.mean(list(class_accuracies.values())))
        else:
            mca = 0.0
        self.group_mca_history[group_name][group_value].append(mca)
        self.group_overall_history[group_name][group_value].append(float(accuracy))

    def update_location_accuracy(
        self,
        task_id: int,
        location: str,
        accuracy: float,
        class_accuracies: Optional[Dict[int, float]] = None
    ):
        """Record location-wise overall and per-class accuracy."""
        self.update_group_accuracy(
            group_name='location',
            group_value=location,
            task_id=task_id,
            accuracy=accuracy,
            class_accuracies=class_accuracies
        )
    
    def update_memory(self, memory_mb: float):
        """Record memory usage."""
        self.memory_usage.append(memory_mb)
    
    def update_compute_cost(self, cost_seconds: float):
        """Record compute cost."""
        self.compute_costs.append(cost_seconds)
    
    def update_num_parameters(self, num_params: int):
        """Record number of model parameters."""
        self.num_parameters.append(num_params)
    
    def record_task_time(self, task_id: int, time_seconds: float):
        """Record time taken to train a task."""
        self.task_times[task_id] = time_seconds
    
    def get_average_accuracy(self, up_to_task: Optional[int] = None) -> float:
        """Get average accuracy across all tasks."""
        if up_to_task is None:
            up_to_task = self.num_tasks - 1
        
        accuracies = []
        for task_id in range(up_to_task + 1):
            if task_id in self.task_accuracies and len(self.task_accuracies[task_id]) > 0:
                # Use most recent accuracy
                accuracies.append(self.task_accuracies[task_id][-1])
        
        return float(np.mean(accuracies)) if accuracies else 0.0
    
    def get_forgetting(self, current_task: int) -> float:
        """Calculate average forgetting."""
        forgetting_scores = []
        
        for task_id in range(current_task):
            if task_id in self.best_task_accuracies and \
               task_id in self.task_accuracies and \
               len(self.task_accuracies[task_id]) > 0:
                
                best_acc = self.best_task_accuracies[task_id]
                current_acc = self.task_accuracies[task_id][-1]
                forgetting = max(0, best_acc - current_acc)
                forgetting_scores.append(forgetting)
        
        return float(np.mean(forgetting_scores)) if forgetting_scores else 0.0
    
    def get_current_memory(self) -> float:
        return self.memory_usage[-1] if self.memory_usage else 0.0
    
    def get_total_compute_cost(self) -> float:
        if self.compute_costs:
            return float(sum(self.compute_costs))
        if self.task_times:
            return float(sum(self.task_times.values()))
        return 0.0
    
    def get_summary(self, current_task: int) -> Dict:
        summary = {
            'average_accuracy': self.get_average_accuracy(current_task),
            'forgetting': self.get_forgetting(current_task),
            'memory_mb': self.get_current_memory(),
            'total_compute_seconds': self.get_total_compute_cost(),
            'num_parameters': self.num_parameters[-1] if self.num_parameters else 0,
            'task_accuracies': {k: v[-1] for k, v in self.task_accuracies.items() 
                              if len(v) > 0}
        }
        return summary
    
    def print_summary(self, current_task: int):
        summary = self.get_summary(current_task)
        
        print("\n" + "="*60)
        print(f"Metrics Summary (After Task {current_task})")
        print("="*60)
        print(f"Average Accuracy:     {summary['average_accuracy']:.4f}")
        print(f"Forgetting:           {summary['forgetting']:.4f}")
        print(f"Memory Usage:         {summary['memory_mb']:.2f} MB")
        print(f"Total Compute Time:   {summary['total_compute_seconds']:.2f} s")
        print(f"Parameters:           {summary['num_parameters']:,}")
        print("\nPer-Task Accuracies:")
        for task_id, acc in sorted(summary['task_accuracies'].items()):
            print(f"  Task {task_id}: {acc:.4f}")
        print("="*60 + "\n")


def evaluate_model(
    model: nn.Module,
    dataloader,
    task_id: int,
    device: str = 'cuda',
    return_per_class: bool = False,
    show_progress: bool = False
) -> Union[float, Tuple[float, Dict[int, float]]]:
    """
    Evaluate model on a specific task with vectorized GPU operations.
    """
    model.eval()
    use_cuda = str(device).startswith('cuda')
    
    max_cls = max(20, int(getattr(getattr(model, 'fc', None), 'out_features', 20)))
    total_class_counts = torch.zeros(max_cls, device=device, dtype=torch.long)
    correct_class_counts = torch.zeros(max_cls, device=device, dtype=torch.long)
    
    total_samples = 0
    total_correct = 0
    progress_update_interval = 10
    total_batches = len(dataloader)
    
    # Track prediction distribution for debugging
    with torch.inference_mode():
        iterator = dataloader
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc=f"Task {task_id}", leave=False)

        for batch_idx, (images, labels) in enumerate(iterator):
            if use_cuda:
                images = images.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            _, predicted = outputs.max(1)

            # Global stats
            batch_correct = predicted.eq(labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)

            if return_per_class:
                # Vectorized class counting
                total_class_counts += torch.bincount(labels, minlength=max_cls)[:max_cls]

                mask = (predicted == labels)
                correct_class_counts += torch.bincount(labels[mask], minlength=max_cls)[:max_cls]

            # Update progress bar
            if show_progress and total_samples > 0 and (
                (batch_idx + 1) % progress_update_interval == 0
                or (batch_idx + 1) == total_batches
            ):
                iterator.set_postfix({'acc': f'{100.0 * total_correct / total_samples:.2f}%'})
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    if return_per_class:
        # Move final stats to CPU
        t_counts = total_class_counts.cpu().numpy()
        c_counts = correct_class_counts.cpu().numpy()
        
        per_class_acc = {}
        for c in range(max_cls):
            total = t_counts[c]
            if total > 0:
                per_class_acc[c] = c_counts[c] / total
        
        # Clean up empty classes from dict if desired, or keep only relevant ones
        # CLAD-C has classes 1-6 (idx 1-6), 0 is unused.
        return accuracy, per_class_acc
    
    return accuracy


def evaluate_model_with_groups(
    model: nn.Module,
    dataloader,
    task_id: int,
    device: str = 'cuda',
    group_id_tensors: Optional[Dict[str, torch.Tensor]] = None,
    group_names: Optional[Dict[str, List[str]]] = None,
    show_progress: bool = False
) -> Dict:
    """Evaluate once and return overall/per-class plus per-group metrics.

    The optional group-id tensors must align with dataset order in
    ``dataloader`` (``shuffle=False``).
    """
    model.eval()
    use_cuda = str(device).startswith('cuda')

    max_cls = max(20, int(getattr(getattr(model, 'fc', None), 'out_features', 20)))
    total_class_counts = torch.zeros(max_cls, device=device, dtype=torch.long)
    correct_class_counts = torch.zeros(max_cls, device=device, dtype=torch.long)

    if group_id_tensors is None:
        group_id_tensors = {}
    if group_names is None:
        group_names = {}

    active_groups = {}
    for group_name, names in group_names.items():
        ids = group_id_tensors.get(group_name)
        if ids is None or not torch.is_tensor(ids) or ids.numel() == 0 or not names:
            continue
        active_groups[group_name] = {
            'names': list(names),
            'ids_cpu': ids.detach().to(dtype=torch.long, device='cpu'),
            'total_samples': torch.zeros(len(names), device=device, dtype=torch.long),
            'total_correct': torch.zeros(len(names), device=device, dtype=torch.long),
            'total_class_counts': torch.zeros(len(names), max_cls, device=device, dtype=torch.long),
            'correct_class_counts': torch.zeros(len(names), max_cls, device=device, dtype=torch.long),
        }

    total_samples = 0
    total_correct = 0
    sample_offset = 0
    progress_update_interval = 10
    total_batches = len(dataloader)

    with torch.inference_mode():
        iterator = dataloader
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc=f"Task {task_id}", leave=False)

        for batch_idx, (images, labels) in enumerate(iterator):
            if use_cuda:
                images = images.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = outputs.max(1)

            batch_correct = predicted.eq(labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)

            total_class_counts += torch.bincount(labels, minlength=max_cls)[:max_cls]
            correct_mask = (predicted == labels)
            correct_class_counts += torch.bincount(labels[correct_mask], minlength=max_cls)[:max_cls]

            if active_groups:
                batch_size = int(labels.size(0))
                next_offset = sample_offset + batch_size

                for group_stats in active_groups.values():
                    ids_cpu = group_stats['ids_cpu']
                    if next_offset > int(ids_cpu.numel()):
                        continue

                    batch_group_ids = ids_cpu[sample_offset:next_offset].to(
                        device,
                        non_blocking=True
                    )

                    for value_id in range(len(group_stats['names'])):
                        value_mask = (batch_group_ids == value_id)
                        if not bool(value_mask.any().item()):
                            continue

                        value_labels = labels[value_mask]
                        value_preds = predicted[value_mask]
                        group_stats['total_samples'][value_id] += int(value_labels.size(0))
                        group_stats['total_correct'][value_id] += int(value_preds.eq(value_labels).sum().item())

                        group_stats['total_class_counts'][value_id] += torch.bincount(
                            value_labels,
                            minlength=max_cls
                        )[:max_cls]

                        value_correct_mask = value_preds.eq(value_labels)
                        if bool(value_correct_mask.any().item()):
                            group_stats['correct_class_counts'][value_id] += torch.bincount(
                                value_labels[value_correct_mask],
                                minlength=max_cls
                            )[:max_cls]

                sample_offset = next_offset

            if show_progress and total_samples > 0 and (
                (batch_idx + 1) % progress_update_interval == 0
                or (batch_idx + 1) == total_batches
            ):
                iterator.set_postfix({'acc': f'{100.0 * total_correct / total_samples:.2f}%'})

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    t_counts = total_class_counts.cpu().numpy()
    c_counts = correct_class_counts.cpu().numpy()

    per_class_acc = {}
    for c in range(max_cls):
        total = t_counts[c]
        if total > 0:
            per_class_acc[c] = c_counts[c] / total

    group_accuracy = {}
    group_per_class_acc = {}
    for group_name, group_stats in active_groups.items():
        group_accuracy[group_name] = {}
        group_per_class_acc[group_name] = {}
        total_class_np = group_stats['total_class_counts'].cpu().numpy()
        correct_class_np = group_stats['correct_class_counts'].cpu().numpy()

        for value_id, value_name in enumerate(group_stats['names']):
            group_total = int(group_stats['total_samples'][value_id].item())
            if group_total <= 0:
                continue

            group_correct = int(group_stats['total_correct'][value_id].item())
            group_accuracy[group_name][value_name] = float(group_correct / group_total)

            value_per_class = {}
            for c in range(max_cls):
                cls_total = int(total_class_np[value_id][c])
                if cls_total > 0:
                    value_per_class[c] = float(correct_class_np[value_id][c] / cls_total)
            group_per_class_acc[group_name][value_name] = value_per_class

    return {
        'accuracy': float(accuracy),
        'per_class_acc': per_class_acc,
        'group_accuracy': group_accuracy,
        'group_per_class_acc': group_per_class_acc
    }


def evaluate_model_with_locations(
    model: nn.Module,
    dataloader,
    task_id: int,
    device: str = 'cuda',
    location_ids: Optional[torch.Tensor] = None,
    location_names: Optional[List[str]] = None,
    show_progress: bool = False
) -> Dict:
    """Backward-compatible location-only wrapper."""
    if location_names is None:
        location_names = ['Citystreet', 'Countryroad', 'Highway']
    results = evaluate_model_with_groups(
        model=model,
        dataloader=dataloader,
        task_id=task_id,
        device=device,
        group_id_tensors={'location': location_ids} if location_ids is not None else {},
        group_names={'location': location_names},
        show_progress=show_progress
    )
    return {
        'accuracy': results['accuracy'],
        'per_class_acc': results['per_class_acc'],
        'location_accuracy': results.get('group_accuracy', {}).get('location', {}),
        'location_per_class_acc': results.get('group_per_class_acc', {}).get('location', {})
    }


def compute_model_size(model: nn.Module) -> Dict[str, float]:
    """Compute model size statistics."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'total_mb': total_size / (1024 ** 2),
        'params_mb': param_size / (1024 ** 2),
        'buffers_mb': buffer_size / (1024 ** 2),
        'num_params': sum(p.numel() for p in model.parameters()),
        'num_trainable_params': sum(p.numel() for p in model.parameters() 
                                   if p.requires_grad)
    }


def compute_inference_efficiency(
    model: nn.Module,
    device: str = 'cuda',
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    warmup_runs: int = 5,
    measure_runs: int = 20
) -> Dict[str, Union[str, Tuple[int, int, int, int], int, float]]:
    """
    Compute approximate MACs/FLOPs and measured inference latency.

    FLOPs are approximated from Conv2d/Linear hooks as 2 * MACs.
    """
    warmup_runs = max(0, int(warmup_runs))
    measure_runs = max(1, int(measure_runs))

    if len(input_shape) != 4:
        input_shape = (1, 3, 224, 224)

    requested_cuda = str(device).startswith('cuda')
    run_device = torch.device('cuda' if (requested_cuda and torch.cuda.is_available()) else 'cpu')

    model_was_training = model.training
    model.eval()

    # Keep model on its current device unless it differs from requested run device.
    try:
        current_device = next(model.parameters()).device
    except StopIteration:
        current_device = run_device

    if current_device != run_device:
        model.to(run_device)
    else:
        run_device = current_device

    sample = torch.randn(*input_shape, device=run_device)
    if run_device.type == 'cuda':
        sample = sample.contiguous(memory_format=torch.channels_last)

    macs = 0

    def _as_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            return output[0]
        return None

    def conv_hook(module, _inputs, output):
        nonlocal macs
        out = _as_tensor(output)
        if out is None or out.ndim < 4:
            return

        batch = int(out.shape[0])
        out_h = int(out.shape[-2])
        out_w = int(out.shape[-1])
        if isinstance(module.kernel_size, tuple):
            k_h, k_w = module.kernel_size
        else:
            k_h = module.kernel_size
            k_w = module.kernel_size

        in_per_group = int(module.in_channels // module.groups)
        macs += int(batch * module.out_channels * out_h * out_w * in_per_group * k_h * k_w)

    def linear_hook(module, _inputs, output):
        nonlocal macs
        out = _as_tensor(output)
        if out is None:
            return

        batch = int(out.shape[0]) if out.ndim > 1 else 1
        macs += int(batch * module.in_features * module.out_features)

    hook_handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_handles.append(m.register_forward_hook(linear_hook))

    with torch.inference_mode():
        _ = model(sample)

    for h in hook_handles:
        h.remove()

    if run_device.type == 'cuda':
        torch.cuda.synchronize(run_device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(sample)

    if run_device.type == 'cuda':
        torch.cuda.synchronize(run_device)

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(measure_runs):
            _ = model(sample)
    if run_device.type == 'cuda':
        torch.cuda.synchronize(run_device)
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed * 1000.0) / float(measure_runs)
    throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    input_shape_4 = (
        int(input_shape[0]),
        int(input_shape[1]),
        int(input_shape[2]),
        int(input_shape[3])
    )

    if model_was_training:
        model.train()

    return {
        'device': str(run_device),
        'input_shape': input_shape_4,
        'macs': int(macs),
        'flops': int(macs * 2),
        'latency_ms': float(latency_ms),
        'throughput_fps': float(throughput_fps),
    }
