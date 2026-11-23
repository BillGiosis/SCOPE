"""Class-Balanced Loss module for imbalanced classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    """
    
    def __init__(self, class_counts, beta=0.9999, reduction='mean'):
        """
        Args:
            class_counts: List of sample counts per class
            beta: Hyperparameter for effective number calculation
            reduction: 'mean', 'sum', or 'none'
        """
        super(ClassBalancedLoss, self).__init__()

        beta = float(beta)
        counts = torch.tensor(class_counts, dtype=torch.float)
        counts = torch.clamp(counts, min=0.0)

        # Compute effective numbers only for classes seen in current task.
        # Unseen classes get zero weight instead of inf/nan.
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = torch.zeros_like(effective_num)
        valid_mask = counts > 0
        if valid_mask.any():
            weights[valid_mask] = (1.0 - beta) / torch.clamp(effective_num[valid_mask], min=1e-12)

            valid_sum = torch.clamp(weights[valid_mask].sum(), min=1e-12)
            weights[valid_mask] = weights[valid_mask] / valid_sum * float(valid_mask.sum().item())
        else:
            # Degenerate fallback; keep finite uniform weights.
            weights = torch.ones_like(weights)

        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.register_buffer('weights', weights)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (batch_size, num_classes) - logits
            targets: Ground truth labels (batch_size,) - class indices
            
        Returns:
            Class-balanced cross entropy loss
        """
        return F.cross_entropy(
            inputs, 
            targets, 
            weight=self.weights,
            reduction=self.reduction
        )
