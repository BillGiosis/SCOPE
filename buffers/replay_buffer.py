"""
Adaptive Replay Buffer with Knowledge Distillation.
Optimized: Vectorized Sampling, Cached Weights, Zero-Copy Indexing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class ReplayBuffer:
    """
    Replay Buffer class storing images and soft targets for rehearsal

    """
    
    def __init__(
        self,
        max_size: int = 1000,
        distillation_temp: float = 1.5,
        storage_device: str = 'cpu', 
        use_fp16: bool = True
    ):
        self.max_size = max(1, int(max_size))
        self.distillation_temp = distillation_temp
        self.storage_device = storage_device
        self.use_fp16 = use_fp16
        
        self.num_samples = 0
        self.initialized = False
        self.sampling_weights = None

        self.images: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.task_ids: Optional[torch.Tensor] = None
        self.soft_targets: Optional[torch.Tensor] = None

    def _init_storage(self, img_shape: Tuple[int, ...], num_classes_soft: int = 0):
        img_dtype = torch.float16 if self.use_fp16 else torch.float32
        soft_dtype = torch.float16 if self.use_fp16 else torch.float32
        do_pin = (self.storage_device == 'cpu' and torch.cuda.is_available())
        
        self.images = torch.empty((self.max_size, *img_shape), dtype=img_dtype, device=self.storage_device, pin_memory=do_pin)
        self.labels = torch.empty((self.max_size,), dtype=torch.long, device=self.storage_device, pin_memory=do_pin)
        self.task_ids = torch.empty((self.max_size,), dtype=torch.long, device=self.storage_device, pin_memory=do_pin)
        
        if num_classes_soft > 0:
            self.soft_targets = torch.empty((self.max_size, num_classes_soft), dtype=soft_dtype, device=self.storage_device, pin_memory=do_pin)
        
        self.initialized = True
        print(f"Buffer Storage Initialized: {self.max_size} slots")

    def _select_class_aware_overflow_indices(self, overflow_count: int) -> torch.Tensor:
        """Select replacement indices by evicting overrepresented classes first."""
        if overflow_count <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.storage_device)

        if self.num_samples <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.storage_device)

        labels_cpu = self.labels[:self.num_samples].detach().cpu()
        unique_labels = torch.unique(labels_cpu)

        class_to_indices = {}
        class_counts = {}
        for class_tensor in unique_labels:
            class_id = int(class_tensor.item())
            class_indices = torch.nonzero(labels_cpu == class_id, as_tuple=True)[0]
            if class_indices.numel() == 0:
                continue
            shuffled = class_indices[torch.randperm(class_indices.numel())].tolist()
            class_to_indices[class_id] = shuffled
            class_counts[class_id] = len(shuffled)

        selected = []
        min_per_class = 1

        for _ in range(overflow_count):
            eligible_classes = [
                class_id
                for class_id, count in class_counts.items()
                if count > min_per_class and class_to_indices.get(class_id)
            ]

            if eligible_classes:
                evict_class = max(eligible_classes, key=lambda class_id: class_counts[class_id])
                evict_index = int(class_to_indices[evict_class].pop())
                class_counts[evict_class] -= 1
            else:
                evict_index = int(torch.randint(0, self.num_samples, (1,)).item())

            selected.append(evict_index)

        return torch.tensor(selected, dtype=torch.long, device=self.storage_device)

    def add_samples(self, images: torch.Tensor, labels: torch.Tensor, task_id: int, soft_targets: Optional[torch.Tensor] = None):
        """Vectorized Add without Dictionary Overhead."""
        batch_size = images.size(0)
        if batch_size == 0: return

        if not self.initialized:
            soft_dim = soft_targets.size(1) if soft_targets is not None else 0
            self._init_storage(images.shape[1:], soft_dim)

        self.sampling_weights = None

        target_img_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        cpu_images = images.detach().to(device=self.storage_device, dtype=target_img_dtype, non_blocking=True)
        cpu_labels = labels.detach().to(device=self.storage_device, non_blocking=True)
        
        cpu_soft = None
        if self.soft_targets is not None and soft_targets is not None:
            cpu_soft = soft_targets.detach().to(device=self.storage_device, dtype=target_img_dtype, non_blocking=True)

        free_slots = max(0, self.max_size - self.num_samples)
        fill_count = min(batch_size, free_slots)

        if fill_count > 0:
            fill_start = self.num_samples
            fill_end = self.num_samples + fill_count
            fill_indices = torch.arange(fill_start, fill_end, device=self.storage_device)

            self.images[fill_indices] = cpu_images[:fill_count]
            self.labels[fill_indices] = cpu_labels[:fill_count]
            self.task_ids[fill_indices] = task_id
            if cpu_soft is not None:
                self.soft_targets[fill_indices] = cpu_soft[:fill_count]

            self.num_samples = fill_end

        overflow_count = batch_size - fill_count
        if overflow_count > 0:
            overflow_indices = self._select_class_aware_overflow_indices(overflow_count)
            overflow_images = cpu_images[fill_count:fill_count + overflow_count]
            overflow_labels = cpu_labels[fill_count:fill_count + overflow_count]

            self.images[overflow_indices] = overflow_images
            self.labels[overflow_indices] = overflow_labels
            self.task_ids[overflow_indices] = task_id
            if cpu_soft is not None:
                self.soft_targets[overflow_indices] = cpu_soft[fill_count:fill_count + overflow_count]

        self.num_samples = min(self.num_samples, self.max_size)

    def get_batch(self, batch_size: int, task_id: Optional[int] = None, device: Optional[str] = None) -> Dict:
        """Fully vectorized retrieval with cached weights.

        Returns tensors on the buffer storage device by default. If ``device`` is
        provided, tensors are moved to that device before returning.
        """
        if self.num_samples == 0: return None
        
        if task_id is not None:
            mask = (self.task_ids[:self.num_samples] == task_id)
            valid_indices = torch.nonzero(mask, as_tuple=True)[0]
            if len(valid_indices) == 0: return None
            
            sample_idxs = torch.randint(0, len(valid_indices), (batch_size,), device=self.storage_device)
            batch_indices = valid_indices[sample_idxs]
            
        else:
            if self.sampling_weights is None or self.sampling_weights.size(0) != self.num_samples:
                current_labels = self.labels[:self.num_samples]
                counts = torch.bincount(current_labels)
                class_counts = counts[current_labels].float().clamp_min(1.0)
                weights = class_counts.rsqrt()
                weights = weights / weights.mean().clamp_min(1e-6)
                weights = torch.clamp(weights, min=0.5, max=2.0)
                self.sampling_weights = weights
            
            batch_indices = torch.multinomial(self.sampling_weights, batch_size, replacement=True)

        b_images = self.images[batch_indices]
        b_labels = self.labels[batch_indices]
        b_tasks = self.task_ids[batch_indices]
        b_soft = self.soft_targets[batch_indices] if self.soft_targets is not None else None

        b_images = b_images.contiguous()
        b_labels = b_labels.contiguous()
        b_tasks = b_tasks.contiguous()
        if b_soft is not None:
            b_soft = b_soft.contiguous()
        
        if device is None:
            return {
                'images': b_images,
                'labels': b_labels,
                'task_ids': b_tasks,
                'soft_targets': b_soft
            }

        return {
            'images': b_images.to(device, non_blocking=True),
            'labels': b_labels.to(device, non_blocking=True),
            'task_ids': b_tasks.to(device, non_blocking=True),
            'soft_targets': b_soft.to(device, non_blocking=True) if b_soft is not None else None
        }

    def distill(self, model: nn.Module, compression_ratio: float = None, target_size: int = None, device='cuda'):
        """Optimized Distillation."""
        if self.num_samples == 0: return
        if target_size is None:
            target_size = int(self.max_size * 0.9) if compression_ratio is None else int(self.num_samples * compression_ratio)
        target_size = max(1, min(target_size, self.num_samples))
        if self.num_samples <= target_size: return

        self.sampling_weights = None

        module_training_modes = {module: module.training for module in model.modules()}
        model.eval()
        importance_scores = torch.zeros(self.num_samples, device=self.storage_device)
        eval_batch_size = 256

        try:
            with torch.inference_mode():
                for i in range(0, self.num_samples, eval_batch_size):
                    end = min(i + eval_batch_size, self.num_samples)
                    batch_imgs = self.images[i:end].to(device, non_blocking=True)

                    use_cuda_amp = str(device).startswith('cuda')
                    with torch.amp.autocast(device_type='cuda', enabled=use_cuda_amp):
                        outputs = model(batch_imgs)
                        probs = F.softmax(outputs, dim=1)
                        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

                    importance_scores[i:end] = entropy.to(self.storage_device)
        finally:
            for module, was_training in module_training_modes.items():
                module.train(was_training)

        sorted_indices = torch.argsort(importance_scores, descending=True)
        unique_labels = torch.unique(self.labels[:self.num_samples])
        max_per_class = max(1, target_size // len(unique_labels))
        sorted_labels = self.labels[sorted_indices]
        
        keep_mask = torch.zeros(self.num_samples, dtype=torch.bool, device=self.storage_device)
        
        count = 0
        for cls in unique_labels:
            cls_mask = (sorted_labels == cls)
            cls_indices_in_sorted = torch.nonzero(cls_mask, as_tuple=True)[0]
            valid_count = min(len(cls_indices_in_sorted), max_per_class)
            
            if valid_count > 0:
                original_indices_to_keep = sorted_indices[cls_indices_in_sorted[:valid_count]]
                keep_mask[original_indices_to_keep] = True
                count += valid_count
        
        if count < target_size:
            remaining = target_size - count
            not_selected = sorted_indices[~keep_mask[sorted_indices]]
            keep_mask[not_selected[:remaining]] = True
            
        indices_to_keep = torch.nonzero(keep_mask, as_tuple=True)[0]
        self.num_samples = len(indices_to_keep)
        
        self.images[:self.num_samples] = self.images[indices_to_keep]
        self.labels[:self.num_samples] = self.labels[indices_to_keep]
        self.task_ids[:self.num_samples] = self.task_ids[indices_to_keep]
        if self.soft_targets is not None:
            self.soft_targets[:self.num_samples] = self.soft_targets[indices_to_keep]
        
        print(f"Buffer distilled to {self.num_samples} samples")

    def get_memory_size(self) -> float:
        if not self.initialized:
            return 0.0

        # Report logical memory budget based on fixed configured replay capacity.
        slots = self.max_size
        element_size = self.images.element_size()

        image_elements_per_sample = self.images[0].numel()
        size_bytes = slots * image_elements_per_sample * element_size

        size_bytes += slots * 8
        size_bytes += slots * 8

        if self.soft_targets is not None:
            soft_elements_per_sample = self.soft_targets[0].numel()
            size_bytes += slots * soft_elements_per_sample * element_size
        return size_bytes / (1024 ** 2)
    
    def __len__(self):
        return self.num_samples
