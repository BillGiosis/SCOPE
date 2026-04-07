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
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(beta, torch.tensor(class_counts, dtype=torch.float))
        weights = (1.0 - beta) / effective_num
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
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
