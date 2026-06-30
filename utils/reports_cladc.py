"""CLAD-C result-file writer."""

import os
from datetime import datetime

from utils.constants import CLADC_CLASS_NAMES, CLADC_INPUT_SIZE, CLADC_LOCATION_NAMES
from utils.report_common import (
    avg,
    task_label,
    write_amca_summary,
    write_common_replay_data_reallocation_sections,
    write_expansion_history,
    write_final_performance,
    write_model_stats,
    write_per_class_history,
    write_per_task_overall_history,
    write_reallocation_history,
)


def write_cladc_results(config: dict, trainer, timestamp: str) -> str:
    """Write a CLAD-C result report and return the report path."""
    num_tasks = config['benchmark']['num_tasks']
    reallocation_cfg = config.get('parameter_reallocation', {})
    expansion_cfg = config.get('network_expansion', {})
    model_cfg = config.get('model', {})
    expansion_strategy = str(expansion_cfg.get('strategy', 'all_layers')).strip().lower()
    total_time = sum(trainer.metrics.task_times.values())
    total_hours = total_time / 3600
    amca_score = 0.0
    if trainer.use_amca and trainer.amca_tester is not None:
        amca_score = trainer.amca_tester.get_summary()['amca']

    results_file = os.path.join(config['logging']['save_dir'], f'results_{timestamp}.txt')
    os.makedirs(config['logging']['save_dir'], exist_ok=True)

    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLAD-C TRAINING RESULTS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AMCA Score: {amca_score:.4f}\n")
        f.write(f"Training Time: {total_hours:.2f} hours ({total_time:.0f} seconds)\n")
        f.write("=" * 80 + "\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Benchmark: {config['benchmark']['name']}\n")
        f.write(f"Data Root: {config['benchmark'].get('data_root', 'N/A')}\n")
        f.write(f"CLAD Repo Root: {config['benchmark'].get('clad_repo_root', 'N/A')}\n")
        f.write(f"Number of Tasks: {config['benchmark']['num_tasks']}\n")
        f.write(f"Total Classes: {config['benchmark']['total_classes']}\n")
        f.write(f"Classes per Task: {config['benchmark'].get('num_classes_per_task', 'N/A')}\n")
        f.write(f"Max Params Ratio: {config['benchmark'].get('max_params_ratio', 'N/A')}\n\n")

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
        f.write(f"  Gradient Accumulation: {config['training'].get('use_gradient_accumulation', False)}\n")
        f.write(f"  Accumulation Steps: {config['training'].get('accumulation_steps', 1)}\n")
        f.write(f"  Class Balanced Loss: {config['training'].get('use_class_balanced_loss', True)}\n")
        f.write(f"  Class Balanced Beta: {config['training'].get('cb_beta', 'N/A')}\n\n")

        write_common_replay_data_reallocation_sections(f, config, reallocation_cfg, expansion_cfg, expansion_strategy)

        if trainer.use_amca and trainer.amca_tester is not None:
            write_amca_summary(f, trainer, num_tasks)

        write_per_task_overall_history(f, trainer, num_tasks)
        write_per_class_history(f, trainer, num_tasks, CLADC_CLASS_NAMES)
        write_cladc_location_history(f, trainer, num_tasks)
        write_model_stats(f, trainer, total_time)

        write_expansion_history(f, trainer, num_tasks)
        write_reallocation_history(f, trainer, num_tasks)
        write_final_performance(f, trainer, num_tasks, CLADC_CLASS_NAMES, {'location': CLADC_LOCATION_NAMES})
        write_cladc_location_summary(f, trainer)
        write_cladc_combo_metrics(f, trainer, num_tasks)
        f.write("=" * 80 + "\n")

    return results_file


def write_cladc_location_history(f, trainer, num_tasks: int):
    location_class_stats = trainer.metrics.task_group_class_accuracies.get('location', {})
    if not location_class_stats:
        return

    f.write("Per-Location Per-Class Accuracies at End of Each Task:\n")
    for task_id in range(num_tasks):
        task_loc_stats = location_class_stats.get(task_id, {})
        if not task_loc_stats:
            continue
        f.write(f"\n  Task {task_label(task_id)}:\n")
        for location in CLADC_LOCATION_NAMES:
            if location not in task_loc_stats:
                continue
            loc_class_stats = task_loc_stats[location]
            f.write(f"    {location}:\n")
            for class_id in range(1, len(CLADC_CLASS_NAMES)):
                class_name = CLADC_CLASS_NAMES[class_id]
                if class_id in loc_class_stats:
                    f.write(f"      {class_name}: {loc_class_stats[class_id]:.4f}\n")
                else:
                    f.write(f"      {class_name}: N/A\n")
            f.write(f"      MCA: {avg(list(loc_class_stats.values())):.4f}\n")
    f.write("\n")

    f.write("Per-Location AMCA (across all evaluation points):\n")
    location_history = trainer.metrics.group_mca_history.get('location', {})
    for location in CLADC_LOCATION_NAMES:
        hist = location_history.get(location, [])
        if hist:
            f.write(f"  {location}: {avg(hist):.4f}\n")
        else:
            f.write(f"  {location}: N/A\n")
    f.write("\n")


def write_cladc_location_summary(f, trainer):
    location_class_stats = trainer.metrics.task_group_class_accuracies.get('location', {})
    if location_class_stats:
        f.write(
            "Final Per-Location Per-Class Accuracies: "
            "see 'Per-Location Per-Class Accuracies at End of Each Task' above\n"
        )

    f.write("\nPer-Location AMCA (across all evaluations):\n")
    location_history = trainer.metrics.group_mca_history.get('location', {})
    for location in CLADC_LOCATION_NAMES:
        hist = location_history.get(location, [])
        if hist:
            f.write(f"  {location}: {avg(hist):.4f}\n")
        else:
            f.write(f"  {location}: N/A\n")


def write_cladc_combo_metrics(f, trainer, num_tasks: int):
    combo_groups = {
        'Day Combinations (Tasks 1,3,5)': [0, 2, 4],
        'Night Combinations (Tasks 2,4,6)': [1, 3, 5],
    }
    f.write("\nTask-Combination Per-Class + AMCA:\n")
    for combo_name, combo_task_ids in combo_groups.items():
        valid_task_ids = [tid for tid in combo_task_ids if tid < num_tasks and tid in trainer.metrics.task_class_accuracies]
        f.write(f"  {combo_name}:\n")
        if not valid_task_ids:
            f.write("    N/A\n")
            continue

        combo_class_values = {cid: [] for cid in range(1, len(CLADC_CLASS_NAMES))}
        for tid in valid_task_ids:
            class_stats = trainer.metrics.task_class_accuracies[tid]
            for cid in range(1, len(CLADC_CLASS_NAMES)):
                if cid in class_stats:
                    combo_class_values[cid].append(class_stats[cid])

        combo_means = {}
        for cid in range(1, len(CLADC_CLASS_NAMES)):
            values = combo_class_values[cid]
            if values:
                combo_means[cid] = avg(values)

        if combo_means:
            combo_amca = avg(list(combo_means.values()))
            for cid in range(1, len(CLADC_CLASS_NAMES)):
                class_name = CLADC_CLASS_NAMES[cid]
                if cid in combo_means:
                    f.write(f"    {class_name}: {combo_means[cid]:.4f}\n")
                else:
                    f.write(f"    {class_name}: N/A\n")
            f.write(f"    AMCA: {combo_amca:.4f}\n")
        else:
            f.write("    N/A\n")
