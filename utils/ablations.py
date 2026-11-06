"""Named SCOPE ablations applied to the benchmark configurations."""

import copy
from pathlib import Path


ABLATION_NAMES = (
    'reference',
    'no_recovery',
    'no_compaction',
    'expansion_only',
    'joint_adapters',
    'no_kd',
    'fixed_architecture',
    'no_recovery_no_compaction',
    'tight_param_cap',
    'small_replay',
)


def apply_ablation(config, name=None, *, seed=42):
    """Apply one ablation to a copy of the benchmark configuration.

    Output directories include the condition and seed. All unspecified training,
    evaluation and replay settings are inherited from the supplied YAML.
    """
    if name is None:
        return config
    if name not in ABLATION_NAMES:
        raise ValueError(f'Unknown SCOPE ablation: {name}')
    dataset = str(config.get('benchmark', {}).get('name', '')).upper()
    if dataset not in {'CLAD-C', 'BDD-C'}:
        raise ValueError(f'Ablation presets require CLAD-C or BDD-C, got {dataset!r}')

    result = copy.deepcopy(config)
    if name == 'reference':
        pass  # Isolate output by seed without changing the reference recipe.
    elif name == 'no_recovery':
        result.setdefault('parameter_reallocation', {}).setdefault('recovery', {})['enabled'] = False
    elif name == 'no_compaction':
        result.setdefault('replay_buffer', {})['auto_distill_end_of_task'] = False
    elif name == 'expansion_only':
        # Recovery already returns immediately when no reallocation was applied.
        # Preserve replay and compaction settings in the expansion-only control.
        result.setdefault('parameter_reallocation', {})['enabled'] = False
    elif name == 'joint_adapters':
        result.setdefault('training', {})['historical_adapter_policy'] = 'joint'
    elif name == 'no_kd':
        # Keep rehearsal CE, soft-target storage and entropy compaction intact.
        result.setdefault('replay_buffer', {})['use_distillation_loss'] = False
    elif name == 'fixed_architecture':
        result.setdefault('network_expansion', {})['enabled'] = False
        result.setdefault('parameter_reallocation', {})['enabled'] = False
        result['parameter_reallocation'].setdefault('recovery', {})['enabled'] = False
    elif name == 'no_recovery_no_compaction':
        result.setdefault('parameter_reallocation', {}).setdefault('recovery', {})['enabled'] = False
        result.setdefault('replay_buffer', {})['auto_distill_end_of_task'] = False
    elif name == 'tight_param_cap':
        result['benchmark']['max_params_ratio'] = 1.025 if dataset == 'CLAD-C' else 1.05
    elif name == 'small_replay':
        result.setdefault('replay_buffer', {})['max_size'] = 500 if dataset == 'CLAD-C' else 2500

    logging = result.setdefault('logging', {})
    logging['save_dir'] = str(Path(logging.get('save_dir', './checkpoints')) /
                              'ablations' / name / f'seed_{int(seed)}')
    result.setdefault('run', {})['ablation'] = name
    return result
