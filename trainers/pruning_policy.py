"""Permanent pruning policy constants."""


REBUILD_OPTIMIZER_AFTER_PRUNE = True
RECOMPILE_AFTER_PRUNE = True
PROTECT_CURRENT_TASK_ADAPTERS = True
PROTECT_CLASSIFIER = True
PROTECT_DOWNSAMPLE = True


PRUNING_SAFEGUARDS = {
    'rebuild_optimizer_after_prune': REBUILD_OPTIMIZER_AFTER_PRUNE,
    'recompile_after_prune': RECOMPILE_AFTER_PRUNE,
    'protect_current_task_adapters': PROTECT_CURRENT_TASK_ADAPTERS,
    'protect_classifier': PROTECT_CLASSIFIER,
    'protect_downsample': PROTECT_DOWNSAMPLE,
}

