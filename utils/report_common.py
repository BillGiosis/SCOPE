"""Shared result-report helpers."""

from trainers.reallocation_policy import REALLOCATION_SAFEGUARDS


def avg(values):
    return float(sum(values) / len(values)) if values else 0.0


def task_label(task_id: int) -> int:
    return int(task_id) + 1


def write_reallocation_safeguards(f):
    f.write(f"  Rebuild Optimizer After Reallocation: {REALLOCATION_SAFEGUARDS['rebuild_optimizer_after_reallocation']}\n")
    f.write(f"  Recompile After Reallocation: {REALLOCATION_SAFEGUARDS['recompile_after_reallocation']}\n")
    f.write(f"  Protect Current-Task Adapters: {REALLOCATION_SAFEGUARDS['protect_current_task_adapters']}\n")
    f.write(f"  Protect Classifier: {REALLOCATION_SAFEGUARDS['protect_classifier']}\n")
    f.write(f"  Protect Downsample: {REALLOCATION_SAFEGUARDS['protect_downsample']}\n")


def write_reallocation_history(f, trainer, num_tasks: int):
    f.write("\nParameter Reallocation Per Task:\n")
    reallocation_by_task = {entry['task_id']: entry for entry in trainer.reallocation_history}
    for task_id in range(num_tasks):
        entry = reallocation_by_task.get(task_id)
        if not entry:
            f.write(f"  Task {task_label(task_id)}: none\n")
            continue
        capacity_before = entry.get('capacity_params_before', entry.get('effective_params_before', 0))
        capacity_after = entry.get('capacity_params_after', entry.get('effective_params_after', 0))
        if entry.get('method') == 'hybrid':
            amount_text = (
                f"whole_amount={entry.get('whole_model_amount', 0.0):.4f}, "
                f"adapter_amount={entry.get('adapter_amount', 0.0):.4f}"
            )
        else:
            amount_text = f"amount={entry['amount']:.4f}"
        f.write(
            f"  Task {task_label(task_id)}: reallocated={entry['reallocated_params']:,}, "
            f"capacity={capacity_before:,}->{capacity_after:,}, "
            f"{amount_text}, scope={entry['scope']}, "
            f"method={entry.get('method', 'structural')}, "
            f"importance={entry.get('importance', 'n/a')}\n"
        )
        for component in entry.get('components', [entry]):
            if component.get('method') == 'structural':
                f.write(f"    Channel ratio: {component['amount']:.4f}; actual scalar parameter reduction: "
                        f"{component.get('scalar_parameter_removal_fraction', 0):.4%}; "
                        f"executed groups: {len(component.get('actual_pruned_groups', []))}\n")
                f.write(f"    Importance diagnostics: {component.get('importance_events', [])}\n")
            elif component.get('method') == 'adapter_delete':
                f.write(f"    Whole-adapter deletion; scalar target={component.get('target_reallocation_params', 0):,}; "
                        f"removed={component.get('removed_params', 0):,}; "
                        f"fallback={component.get('fallback_reason')}\n")


def write_expansion_history(f, trainer, num_tasks: int):
    f.write("Network Expansions Per Task:\n")
    for task_id in range(num_tasks):
        expansions = trainer.task_expansions.get(task_id, [])
        if expansions:
            f.write(f"  Task {task_label(task_id)}: {', '.join(expansions)}\n")
        else:
            f.write(f"  Task {task_label(task_id)}: none\n")


def write_group_sections(f, trainer, num_tasks: int, class_names, metric_groups):
    group_class_stats = trainer.metrics.task_group_class_accuracies
    if not group_class_stats:
        return

    for group_name, group_values in metric_groups.items():
        if not group_values:
            continue

        title = group_name.replace('_', ' ').title()
        f.write(f"Per-{title} Per-Class Accuracies at End of Each Task:\n")
        group_task_stats = group_class_stats.get(group_name, {})
        for task_id in range(num_tasks):
            task_stats = group_task_stats.get(task_id, {})
            if not task_stats:
                continue
            f.write(f"\n  Task {task_label(task_id)}:\n")
            for group_value in group_values:
                if group_value not in task_stats:
                    continue
                f.write(f"    {group_value}:\n")
                per_class = task_stats[group_value]
                for class_id, class_name in enumerate(class_names):
                    if class_id in per_class:
                        f.write(f"      {class_name}: {per_class[class_id]:.4f}\n")
                    else:
                        f.write(f"      {class_name}: N/A\n")
                f.write(f"      MCA: {avg(list(per_class.values())):.4f}\n")
        f.write("\n")

        f.write(f"Per-{title} AMCA (across all evaluation points):\n")
        group_history = trainer.metrics.group_mca_history.get(group_name, {})
        for group_value in group_values:
            hist = group_history.get(group_value, [])
            if hist:
                f.write(f"  {group_value}: {avg(hist):.4f}\n")
            else:
                f.write(f"  {group_value}: N/A\n")
        f.write("\n")


def write_group_amca_summary(f, trainer, metric_groups):
    f.write("\nGrouped AMCA (across all evaluations):\n")
    for group_name, group_values in metric_groups.items():
        title = group_name.replace('_', ' ').title()
        f.write(f"  {title}:\n")
        group_history = trainer.metrics.group_mca_history.get(group_name, {})
        for group_value in group_values:
            hist = group_history.get(group_value, [])
            if hist:
                f.write(f"    {group_value}: {avg(hist):.4f}\n")
            else:
                f.write(f"    {group_value}: N/A\n")


def write_common_replay_data_reallocation_sections(
    f,
    config: dict,
    reallocation_cfg: dict,
    expansion_cfg: dict,
    expansion_strategy: str
):
    f.write("Replay Buffer:\n")
    f.write(f"  Replay Buffer Enabled: {config['replay_buffer'].get('enabled', True)}\n")
    f.write(f"  Max Size: {config['replay_buffer']['max_size']}\n")
    if 'samples_per_class' in config['replay_buffer']:
        f.write(f"  Samples Per Class: {config['replay_buffer'].get('samples_per_class', 'N/A')}\n")
    f.write(f"  Distillation Temp: {config['replay_buffer']['distillation_temp']}\n")
    f.write(f"  Soft Targets: {config['replay_buffer'].get('use_soft_targets', True)}\n")
    f.write(f"  Distillation Loss: {config['replay_buffer'].get('use_distillation_loss', True)}\n")
    f.write(f"  Replay Batch Size: {config['replay_buffer']['replay_batch_size']}\n")
    f.write(f"  Replay Interval Batches: {config['replay_buffer'].get('replay_interval_batches', 1)}\n")
    f.write(f"  Auto Distill End Of Task: {config['replay_buffer'].get('auto_distill_end_of_task', True)}\n")
    f.write(
        "  End Of Task Distill Compression Ratio: "
        f"{config['replay_buffer'].get('end_of_task_distill_compression_ratio', 'N/A')}\n\n"
    )

    f.write("Data Loading:\n")
    f.write(f"  Num Workers: {config['data_loading']['num_workers']}\n")
    f.write(f"  Pin Memory: {config['data_loading']['pin_memory']}\n")
    f.write(f"  Prefetch Factor: {config['data_loading'].get('prefetch_factor', 'N/A')}\n\n")

    f.write("Network Expansion:\n")
    f.write(f"  Enabled: {expansion_cfg.get('enabled', True)}\n")
    f.write(f"  Strategy: {expansion_strategy}\n")
    if expansion_strategy == 'layer2_layer3_split':
        f.write("  Layer Allocation: request layer2 then layer3, each subject to the cap; parameter costs differ\n")
    else:
        f.write("  Layer Allocation: budget-aware even distribution across all 4 layers\n")
    f.write("\n")

    f.write("Parameter Reallocation:\n")
    f.write("  Structural amount is a local channel ratio, not a scalar parameter removal fraction.\n")
    f.write("  whole_model uses dependency-graph roots; historical adapters can also be modified.\n")
    f.write("  Ignored roots may still have dependent input dimensions updated.\n")
    f.write(f"  Historical adapter training policy: {config.get('training', {}).get('historical_adapter_policy', 'frozen')}; "
            "the shared backbone remains trainable.\n")
    f.write("  Replay targets are cached during the final main epoch, before pruning/recovery.\n")
    f.write(f"  Enabled: {reallocation_cfg.get('enabled', False)}\n")
    f.write(f"  Non-Hybrid Amount: {reallocation_cfg.get('amount', 0.0)}\n")
    f.write(f"  Warmup Tasks: {reallocation_cfg.get('warmup_tasks', 2)}\n")
    f.write(f"  Scope: {reallocation_cfg.get('scope', 'whole_model')}\n")
    hybrid_cfg = reallocation_cfg.get('hybrid', {})
    f.write(f"  Hybrid Whole Model Amount: {hybrid_cfg.get('whole_model_amount', 'N/A')}\n")
    f.write(f"  Hybrid Adapter Amount: {hybrid_cfg.get('adapter_amount', 'N/A')}\n")
    f.write(f"  Hybrid Adapter Scope: {hybrid_cfg.get('adapter_scope', 'N/A')}\n")
    f.write(f"  Importance: {reallocation_cfg.get('importance', 'magnitude')}\n")
    f.write(f"  Use Replay For Taylor: {reallocation_cfg.get('use_replay_for_taylor', True)}\n")
    f.write(f"  Taylor Replay Batches: {reallocation_cfg.get('taylor_replay_batches', 2)}\n")
    f.write(f"  Taylor Train Batches: {reallocation_cfg.get('taylor_train_batches', 0)}\n")
    f.write(f"  Taylor Replay Batch Size: {reallocation_cfg.get('taylor_replay_batch_size', config['replay_buffer'].get('replay_batch_size', 32))}\n")
    write_reallocation_safeguards(f)
    recovery_cfg = reallocation_cfg.get('recovery', {})
    f.write(f"  Recovery Enabled: {recovery_cfg.get('enabled', True)}\n")
    f.write(f"  Recovery Epochs: {recovery_cfg.get('epochs', 2)}\n")
    f.write(f"  Recovery LR Scale: {recovery_cfg.get('learning_rate_scale', 0.25)}\n\n")

    f.write("Logging:\n")
    f.write(f"  Save Dir: {config['logging']['save_dir']}\n\n")


def write_amca_summary(f, trainer, num_tasks: int):
    amca_summary = trainer.amca_tester.get_summary()
    f.write(f"AMCA Score: {amca_summary['amca']:.4f}\n")
    f.write(f"Final MCA (Mean Class Acc): {amca_summary['final_mca']:.4f}\n")
    f.write(f"Number of Evaluations: {amca_summary['num_evaluations']}\n\n")
    f.write("MCA at each evaluation (grouped by training task):\n")
    mca_list = amca_summary['mean_class_accuracies']
    evals_per_task = len(mca_list) // num_tasks if num_tasks > 0 else len(mca_list)
    for task_id in range(num_tasks):
        start_idx = task_id * evals_per_task
        end_idx = min((task_id + 1) * evals_per_task, len(mca_list))
        task_mcas = mca_list[start_idx:end_idx]
        if task_mcas:
            f.write(
                f"  During Task {task_label(task_id)}: "
                f"Final MCA={task_mcas[-1]:.4f}, Best MCA={max(task_mcas):.4f}\n"
            )
    f.write("\n")


def write_per_task_overall_history(f, trainer, num_tasks: int):
    f.write("Evaluation overall accuracy at each training checkpoint (History):\n")
    for task_id in range(num_tasks):
        accs = trainer.metrics.task_accuracies.get(task_id, [])
        if accs:
            f.write(f"  End of Task {task_label(task_id)}: {accs[-1]:.4f}\n")
    f.write("\n")


def write_per_class_history(f, trainer, num_tasks: int, class_names):
    f.write("Per-Class Accuracies at End of Each Task:\n")
    for task_id in range(num_tasks):
        class_stats = trainer.metrics.task_class_accuracies.get(task_id)
        if not class_stats:
            continue
        f.write(f"\n  Task {task_label(task_id)}:\n")
        for class_id, class_name in enumerate(class_names):
            if class_id in class_stats:
                f.write(f"    {class_name}: {class_stats[class_id]:.4f}\n")
            else:
                f.write(f"    {class_name}: N/A\n")
    f.write("\n")


def write_model_stats(f, trainer, total_time: float):
    if trainer.metrics.memory_usage:
        f.write(f"Final Model + Replay Tensor Storage: {trainer.metrics.memory_usage[-1]:.2f} MB (excludes optimizer/activations/process RAM)\n")
    if trainer.metrics.num_parameters:
        f.write(f"Final Parameters: {trainer.metrics.num_parameters[-1]:,}\n")
    f.write(f"Total Training Time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)\n\n")
    write_observation_summary(f, trainer)


def write_observation_summary(f, trainer):
    f.write(f"Run ID: {trainer.config.get('run', {}).get('id', 'unrecorded')}\n")
    f.write(f"Torch seed: {trainer.config.get('run', {}).get('seed', 'unrecorded')}\n")
    f.write(f"Observation artifacts: {trainer.observations.directory}\n")
    replay = trainer.replay_buffer.get_storage_stats()
    f.write(f"Replay occupancy/capacity: {replay['samples']}/{replay['capacity_samples']} samples\n")
    f.write(f"Replay occupied payload: {replay['occupied_payload_bytes'] / 1024**2:.2f} MB; "
            f"allocated backing tensors: {replay['allocated_backing_bytes'] / 1024**2:.2f} MB\n")
    f.write("Compaction lowers occupancy; maximum-capacity backing tensors remain allocated.\n")
    f.write("Phase wall times (nested scoring is included in reallocation_total):\n")
    phases = {}
    for entry in trainer.observations.phases:
        phases[entry['phase']] = phases.get(entry['phase'], 0.0) + entry['wall_seconds']
    for name, seconds in phases.items():
        f.write(f"  {name}: {seconds:.3f} s\n")
    f.write("CUDA synchronization/peak measurement: " + str(trainer.observations.profile_cuda) + "\n")
    memory = next((entry['parent_process_memory'] for entry in reversed(trainer.observations.phases)
                   if entry.get('parent_process_memory')), None)
    if memory:
        f.write(f"Parent-process lifetime peak RSS: {memory['lifetime_peak_rss_bytes'] / 1024**2:.2f} MB (excludes live workers)\n")
    final_task = max(trainer.metrics.task_times, default=0)
    for name, value in [('Seen-task average accuracy', trainer.metrics.get_average_accuracy(final_task)),
                        ('Forgetting', trainer.metrics.get_forgetting(final_task)),
                        ('Backward transfer', trainer.metrics.get_backward_transfer(final_task))]:
        f.write(f"{name}: {'N/A (no task evaluation matrix)' if value is None else f'{value:.4f}'}\n")
    f.write("\n")


def write_final_performance(f, trainer, num_tasks: int, class_names, metric_groups):
    f.write("\n" + "=" * 80 + "\n")
    f.write("FINAL MODEL PERFORMANCE (End of Training)\n")
    f.write("=" * 80 + "\n")

    final_task_id = num_tasks - 1
    final_overall_acc = trainer.metrics.task_accuracies.get(final_task_id, [])
    f.write(
        f"Final Overall Model Accuracy: "
        f"{final_overall_acc[-1]:.4f}\n" if final_overall_acc else "Final Overall Model Accuracy: N/A\n"
    )
    f.write(
        f"Final Per-Class Accuracies: see 'Per-Class Accuracies at End of Each Task' -> "
        f"Task {task_label(final_task_id)}\n"
    )
    if metric_groups:
        f.write(
            "Final Grouped Per-Class Accuracies: "
            f"see grouped history sections above -> Task {task_label(final_task_id)}\n"
        )

    f.write("\nAverage Per-Class Accuracies (across all evaluations):\n")
    if trainer.metrics.task_class_accuracies:
        class_accuracies_over_time = {class_id: [] for class_id in range(len(class_names))}
        for task_id in range(num_tasks):
            class_stats = trainer.metrics.task_class_accuracies.get(task_id, {})
            for class_id in range(len(class_names)):
                if class_id in class_stats:
                    class_accuracies_over_time[class_id].append(class_stats[class_id])
        for class_id, class_name in enumerate(class_names):
            values = class_accuracies_over_time[class_id]
            if values:
                f.write(f"  {class_name}: {avg(values):.4f}\n")
            else:
                f.write(f"  {class_name}: N/A\n")
    else:
        f.write("  N/A\n")
