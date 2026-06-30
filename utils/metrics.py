"""
Metrics and evaluation utilities for continual learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict


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
        self.class_accuracies_over_time = []
        self.mean_class_accuracies = []
        
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

                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(images)

                _, predicted = outputs.max(1)

                class_total_tensor += torch.bincount(labels, minlength=max_cls)[:max_cls]

                correct_mask = (predicted == labels)
                class_correct_tensor += torch.bincount(labels[correct_mask], minlength=max_cls)[:max_cls]
        
        class_correct = class_correct_tensor.cpu().numpy()
        class_total = class_total_tensor.cpu().numpy()

        class_accuracies = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if c < len(class_total) and class_total[c] > 0:
                class_accuracies[c] = class_correct[c] / class_total[c]
            else:
                class_accuracies[c] = 0.0
        
        self.class_accuracies_over_time.append(class_accuracies.copy())

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
        
        self.task_accuracies = defaultdict(list)

        self.task_class_accuracies = {}

        # task_id -> {group_value: overall_accuracy}
        self.task_group_accuracies = defaultdict(lambda: defaultdict(dict))
        # task_id -> {group_value: {class_id: accuracy}}
        self.task_group_class_accuracies = defaultdict(lambda: defaultdict(dict))
        self.group_mca_history = defaultdict(lambda: defaultdict(list))
        self.group_overall_history = defaultdict(lambda: defaultdict(list))
        
        self.best_task_accuracies = {}

        self.memory_usage = []
        self.compute_costs = []
        self.num_parameters = []

        self.task_times = {}

    @staticmethod
    def _nested_defaultdict_dict():
        return defaultdict(dict)

    @staticmethod
    def _nested_defaultdict_list():
        return defaultdict(list)

    @staticmethod
    def _to_plain_dict(value):
        """Recursively convert defaultdicts to pickle-safe plain dictionaries."""
        if isinstance(value, defaultdict):
            value = dict(value)
        if isinstance(value, dict):
            return {key: MetricsTracker._to_plain_dict(child) for key, child in value.items()}
        if isinstance(value, list):
            return [MetricsTracker._to_plain_dict(child) for child in value]
        return value

    @staticmethod
    def _to_group_task_defaultdict(value):
        result = defaultdict(MetricsTracker._nested_defaultdict_dict)
        for group_name, task_values in (value or {}).items():
            nested = MetricsTracker._nested_defaultdict_dict()
            for task_id, group_values in (task_values or {}).items():
                nested[task_id] = dict(group_values or {})
            result[group_name] = nested
        return result

    @staticmethod
    def _to_group_history_defaultdict(value):
        result = defaultdict(MetricsTracker._nested_defaultdict_list)
        for group_name, group_values in (value or {}).items():
            nested = MetricsTracker._nested_defaultdict_list()
            for group_value, history in (group_values or {}).items():
                nested[group_value] = list(history or [])
            result[group_name] = nested
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state['task_accuracies'] = self._to_plain_dict(self.task_accuracies)
        state['task_group_accuracies'] = self._to_plain_dict(self.task_group_accuracies)
        state['task_group_class_accuracies'] = self._to_plain_dict(self.task_group_class_accuracies)
        state['group_mca_history'] = self._to_plain_dict(self.group_mca_history)
        state['group_overall_history'] = self._to_plain_dict(self.group_overall_history)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.task_accuracies = defaultdict(list, {
            task_id: list(values)
            for task_id, values in self.task_accuracies.items()
        })
        self.task_group_accuracies = self._to_group_task_defaultdict(self.task_group_accuracies)
        self.task_group_class_accuracies = self._to_group_task_defaultdict(self.task_group_class_accuracies)
        self.group_mca_history = self._to_group_history_defaultdict(self.group_mca_history)
        self.group_overall_history = self._to_group_history_defaultdict(self.group_overall_history)
        
    def update_accuracy(self, task_id: int, accuracy: float, class_accuracies: Optional[Dict[int, float]] = None):
        """Record accuracy for a specific task."""
        self.task_accuracies[task_id].append(accuracy)
        
        if class_accuracies is not None:
            self.task_class_accuracies[task_id] = class_accuracies

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
        print(f"Metrics Summary (After Task {int(current_task) + 1})")
        print("="*60)
        print(f"Average Accuracy:     {summary['average_accuracy']:.4f}")
        print(f"Forgetting:           {summary['forgetting']:.4f}")
        print(f"Memory Usage:         {summary['memory_mb']:.2f} MB")
        print(f"Total Compute Time:   {summary['total_compute_seconds']:.2f} s")
        print(f"Parameters:           {summary['num_parameters']:,}")
        print("\nPer-Task Accuracies:")
        for task_id, acc in sorted(summary['task_accuracies'].items()):
            print(f"  Task {int(task_id) + 1}: {acc:.4f}")
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
    
    with torch.inference_mode():
        iterator = dataloader
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc=f"Task {int(task_id) + 1}", leave=False)

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

            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(images)

            _, predicted = outputs.max(1)

            batch_correct = predicted.eq(labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)

            if return_per_class:
                total_class_counts += torch.bincount(labels, minlength=max_cls)[:max_cls]

                mask = (predicted == labels)
                correct_class_counts += torch.bincount(labels[mask], minlength=max_cls)[:max_cls]

            if show_progress and total_samples > 0 and (
                (batch_idx + 1) % progress_update_interval == 0
                or (batch_idx + 1) == total_batches
            ):
                iterator.set_postfix({'acc': f'{100.0 * total_correct / total_samples:.2f}%'})
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    if return_per_class:
        t_counts = total_class_counts.cpu().numpy()
        c_counts = correct_class_counts.cpu().numpy()
        
        per_class_acc = {}
        for c in range(max_cls):
            total = t_counts[c]
            if total > 0:
                per_class_acc[c] = c_counts[c] / total
        
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
        group_ids = ids.detach().to(device=device, dtype=torch.long, non_blocking=True)
        active_groups[group_name] = {
            'names': list(names),
            'ids': group_ids,
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
            iterator = tqdm(dataloader, desc=f"Task {int(task_id) + 1}", leave=False)

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

            with torch.amp.autocast('cuda', enabled=use_cuda):
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
                    ids = group_stats['ids']
                    if next_offset > int(ids.numel()):
                        continue

                    num_group_values = len(group_stats['names'])
                    batch_group_ids = ids[sample_offset:next_offset]
                    valid_group_mask = (batch_group_ids >= 0) & (batch_group_ids < num_group_values)
                    if not bool(valid_group_mask.any().item()):
                        continue

                    valid_group_ids = batch_group_ids[valid_group_mask]
                    valid_labels = labels[valid_group_mask]
                    valid_correct_mask = correct_mask[valid_group_mask]

                    group_stats['total_samples'] += torch.bincount(
                        valid_group_ids,
                        minlength=num_group_values
                    )[:num_group_values]

                    if bool(valid_correct_mask.any().item()):
                        group_stats['total_correct'] += torch.bincount(
                            valid_group_ids[valid_correct_mask],
                            minlength=num_group_values
                        )[:num_group_values]

                    valid_label_mask = (valid_labels >= 0) & (valid_labels < max_cls)
                    if bool(valid_label_mask.any().item()):
                        class_group_ids = valid_group_ids[valid_label_mask]
                        class_labels = valid_labels[valid_label_mask]
                        combined_indices = class_group_ids * max_cls + class_labels
                        class_counts = torch.bincount(
                            combined_indices,
                            minlength=num_group_values * max_cls
                        ).view(num_group_values, max_cls)
                        group_stats['total_class_counts'] += class_counts

                        correct_label_mask = valid_label_mask & valid_correct_mask
                        if bool(correct_label_mask.any().item()):
                            correct_group_ids = valid_group_ids[correct_label_mask]
                            correct_labels = valid_labels[correct_label_mask]
                            correct_indices = correct_group_ids * max_cls + correct_labels
                            correct_counts = torch.bincount(
                                correct_indices,
                                minlength=num_group_values * max_cls
                            ).view(num_group_values, max_cls)
                            group_stats['correct_class_counts'] += correct_counts

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
