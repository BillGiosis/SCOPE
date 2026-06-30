"""Permanent reallocation policy constants."""


REBUILD_OPTIMIZER_AFTER_REALLOCATION = True
RECOMPILE_AFTER_REALLOCATION = True
PROTECT_CURRENT_TASK_ADAPTERS = True
PROTECT_CLASSIFIER = True
PROTECT_DOWNSAMPLE = True


REALLOCATION_SAFEGUARDS = {
    'rebuild_optimizer_after_reallocation': REBUILD_OPTIMIZER_AFTER_REALLOCATION,
    'recompile_after_reallocation': RECOMPILE_AFTER_REALLOCATION,
    'protect_current_task_adapters': PROTECT_CURRENT_TASK_ADAPTERS,
    'protect_classifier': PROTECT_CLASSIFIER,
    'protect_downsample': PROTECT_DOWNSAMPLE,
}

