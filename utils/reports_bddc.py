"""BDD-C result-file writer."""

import os
from datetime import datetime

from utils.constants import BDDC_INPUT_SIZE
from utils.report_common import (
    write_amca_summary,
    write_common_replay_data_reallocation_sections,
    write_expansion_history,
    write_final_performance,
    write_group_amca_summary,
    write_group_sections,
    write_model_stats,
    write_per_class_history,
    write_per_task_overall_history,
    write_reallocation_history,
)


def write_bddc_results(config: dict, trainer, timestamp: str) -> str:
    """Write a BDD-C result report and return the report path."""
    num_tasks = config['benchmark']['num_tasks']
    class_names = list(config['benchmark']['class_names'])
    metric_groups = dict(config['benchmark'].get('metric_groups', {}))
    reallocation_cfg = config.get('parameter_reallocation', {})
    expansion_cfg = config.get('network_expansion', {})
    model_cfg = config.get('model', {})
    expansion_strategy = str(expansion_cfg.get('strategy', 'all_layers')).strip().lower()
    total_time = sum(trainer.metrics.task_times.values())
    amca_score = 0.0
    if trainer.use_amca and trainer.amca_tester is not None:
        amca_score = trainer.amca_tester.get_summary()['amca']

    results_file = os.path.join(config['logging']['save_dir'], f'results_{timestamp}.txt')
    os.makedirs(config['logging']['save_dir'], exist_ok=True)

    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BDD-C TRAINING RESULTS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AMCA Score: {amca_score:.4f}\n")
        f.write(f"Training Time: {total_time / 3600:.2f} hours ({total_time:.0f} seconds)\n")
        f.write("=" * 80 + "\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Benchmark: {config['benchmark']['name']}\n")
        f.write(f"Data Root: {config['benchmark']['data_root']}\n")
        f.write(f"Evaluation Split: {config['benchmark'].get('eval_split', 'val')}\n")
        f.write(f"Number of Tasks: {num_tasks}\n")
        f.write(f"Max Params Ratio: {config['benchmark'].get('max_params_ratio', 'N/A')}\n")
        f.write(f"Total Classes: {config['benchmark']['total_classes']}\n")
        f.write(f"Class Names: {', '.join(class_names)}\n")
        f.write(f"Metric Groups: {', '.join(metric_groups.keys())}\n\n")

        f.write("Model:\n")
        f.write(f"  Backbone: {model_cfg.get('backbone', 'resnet50d')}\n")
        f.write(f"  Adapt Downsample Blocks: {model_cfg.get('adapt_downsample_blocks', True)}\n")

        f.write("Training:\n")
        f.write(f"  Batch Size: {config['training']['batch_size']}\n")
        f.write(f"  Epochs per Task: {config['training']['epochs_per_task']}\n")
        f.write(f"  Learning Rate: {config['training']['learning_rate']}\n")
        f.write(f"  Optimizer: {config['training']['optimizer']}\n")
        f.write(f"  Weight Decay: {config['training']['weight_decay']}\n")
        f.write(f"  AMP: {config['training'].get('use_amp', False)}\n")
        f.write(f"  Class Balanced Loss: {config['training'].get('use_class_balanced_loss', True)}\n")
        f.write(f"  Class Balanced Beta: {config['training'].get('cb_beta', 'N/A')}\n\n")

        write_common_replay_data_reallocation_sections(f, config, reallocation_cfg, expansion_cfg, expansion_strategy)

        if trainer.use_amca and trainer.amca_tester is not None:
            write_amca_summary(f, trainer, num_tasks)

        write_per_task_overall_history(f, trainer, num_tasks)
        write_per_class_history(f, trainer, num_tasks, class_names)
        write_group_sections(f, trainer, num_tasks, class_names, metric_groups)
        write_model_stats(f, trainer, total_time)

        write_expansion_history(f, trainer, num_tasks)
        write_reallocation_history(f, trainer, num_tasks)
        write_final_performance(f, trainer, num_tasks, class_names, metric_groups)
        write_group_amca_summary(f, trainer, metric_groups)
        f.write("\n" + "=" * 80 + "\n")

    return results_file
