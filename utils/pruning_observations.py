"""Record dependency-group pruning and handle nonfinite importance scores."""

import copy

import torch


class SafeStructuralImportance:
    """Preserve finite scores and record zero-score or magnitude fallbacks."""

    def __init__(self, criterion, magnitude, module_names):
        self.criterion = criterion
        self.magnitude = magnitude
        self.module_names = module_names
        self.events = []

    def __getattr__(self, name):
        return getattr(self.criterion, name)

    def __call__(self, group):
        scores = self.criterion(group)
        if scores is None or bool(torch.isfinite(scores).all()):
            return scores
        root = self.module_names.get(group[0].dep.target.module, '<unnamed>')
        # Torch-Pruning's default mean normalization divides an all-zero group
        # by zero. Preserve the zero-importance tie rather than ranking NaNs.
        raw_criterion = copy.copy(self.criterion)
        raw_criterion.normalizer = None
        raw = raw_criterion(group)
        if raw is not None and bool(torch.isfinite(raw).all()) and not bool(raw.abs().any()):
            self.events.append({'root': root, 'reason': 'zero_group_normalization', 'action': 'zero_scores'})
            return torch.zeros_like(scores)
        fallback = self.magnitude(group)
        if fallback is None or not bool(torch.isfinite(fallback).all()):
            raise RuntimeError(f'Non-finite structural importance including magnitude fallback: {root}')
        self.events.append({'root': root, 'reason': 'nonfinite_importance', 'action': 'magnitude_fallback'})
        return fallback


def prune_with_records(pruner, model):
    """Execute pruning groups in their original order and record shape changes."""
    names = {module: name for name, module in model.named_modules()}
    records = []
    for group in pruner.step(interactive=True):
        dependencies = []
        for dependency, indices in group:
            module = dependency.target.module
            weight = getattr(module, 'weight', None)
            dependencies.append({
                'module': names.get(module, str(type(module).__name__)),
                'handler': getattr(dependency.handler, '__name__', str(dependency.handler)),
                'removed_indices_count': len(indices),
                'weight_shape_before': list(weight.shape) if weight is not None else None,
            })
        record = {'root': dependencies[0]['module'], 'root_indices': [int(i) for i in group[0].idxs],
                  'dependencies': dependencies}
        group.prune()
        for entry, (dependency, _) in zip(dependencies, group):
            weight = getattr(dependency.target.module, 'weight', None)
            entry['weight_shape_after'] = list(weight.shape) if weight is not None else None
        records.append(record)
    return records
