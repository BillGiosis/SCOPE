"""Sampling helpers shared by benchmark loaders."""

import torch
from torch.utils.data import WeightedRandomSampler


def make_inverse_frequency_sampler(labels, min_classes: int = 0):
    """Create a class-balanced replacement sampler from integer labels."""
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    if label_tensor.numel() == 0:
        return None, torch.zeros(max(0, int(min_classes)), dtype=torch.long)

    inferred_classes = int(label_tensor.max().item()) + 1
    class_counts = torch.bincount(label_tensor, minlength=max(int(min_classes), inferred_classes))
    class_weights = torch.zeros_like(class_counts, dtype=torch.float32)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = 1.0 / class_counts[nonzero_mask].float()

    sample_weights = class_weights[label_tensor]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, class_counts
