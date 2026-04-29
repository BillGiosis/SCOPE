"""
Main training script for BDD-C benchmark.
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from benchmarks.bddc import create_bddc_benchmark
from buffers.replay_buffer import ReplayBuffer
from models.expandable_resnet import create_expandable_resnet
from trainers.continual_trainer import create_trainer
from utils.metrics import compute_inference_efficiency


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def infer_input_shape_from_loaders(loaders, fallback=(1, 3, 224, 224)):
    if not loaders:
        return fallback

    for loader in loaders:
        if loader is None:
            continue
        try:
            batch = next(iter(loader))
        except Exception:
            continue

        images = None
        if isinstance(batch, (tuple, list)) and len(batch) > 0 and torch.is_tensor(batch[0]):
            images = batch[0]
        elif isinstance(batch, dict):
            candidate = batch.get('images', None)
            if torch.is_tensor(candidate):
                images = candidate

        if images is not None and images.ndim >= 4:
            return (1, int(images.shape[1]), int(images.shape[2]), int(images.shape[3]))

    return fallback


def _avg(values):
    return float(sum(values) / len(values)) if values else 0.0


def _write_group_sections(f, trainer, num_tasks: int, class_names, metric_groups):
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
            f.write(f"\n  Task {task_id}:\n")
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
                f.write(f"      MCA: {_avg(list(per_class.values())):.4f}\n")
        f.write("\n")

        f.write(f"Per-{title} AMCA (across all evaluation points):\n")
        group_history = trainer.metrics.group_mca_history.get(group_name, {})
        for group_value in group_values:
            hist = group_history.get(group_value, [])
            if hist:
                f.write(f"  {group_value}: {_avg(hist):.4f}\n")
            else:
                f.write(f"  {group_value}: N/A\n")
        f.write("\n")


def _write_final_group_sections(f, trainer, final_task_id: int, class_names, metric_groups):
    f.write("\nFinal Grouped Per-Class Accuracies:\n")

    any_output = False
    for group_name, group_values in metric_groups.items():
        if not group_values:
            continue

        title = group_name.replace('_', ' ').title()
        f.write(f"  {title}:\n")
        task_group_stats = trainer.metrics.task_group_class_accuracies.get(group_name, {})
        final_group_stats = task_group_stats.get(final_task_id, {})

        if not final_group_stats:
            f.write("    N/A\n")
            continue

        any_output = True
        for group_value in group_values:
            if group_value not in final_group_stats:
                f.write(f"    {group_value}: N/A\n")
                continue
            f.write(f"    {group_value}:\n")
            per_class = final_group_stats[group_value]
            for class_id, class_name in enumerate(class_names):
                if class_id in per_class:
                    f.write(f"      {class_name}: {per_class[class_id]:.4f}\n")
                else:
                    f.write(f"      {class_name}: N/A\n")
            f.write(f"      MCA: {_avg(list(per_class.values())):.4f}\n")

    if not any_output:
        f.write("  N/A\n")

    f.write("\nGrouped AMCA (across all evaluations):\n")
    for group_name, group_values in metric_groups.items():
        title = group_name.replace('_', ' ').title()
        f.write(f"  {title}:\n")
        group_history = trainer.metrics.group_mca_history.get(group_name, {})
        for group_value in group_values:
            hist = group_history.get(group_value, [])
            if hist:
                f.write(f"    {group_value}: {_avg(hist):.4f}\n")
            else:
                f.write(f"    {group_value}: N/A\n")


def main(args):
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    config = load_config(args.config)
    config.setdefault('training', {})
    config['training']['use_gradient_accumulation'] = False
    config['training']['accumulation_steps'] = 1
    expansion_cfg = config.get('network_expansion', {})
    expansion_strategy = str(expansion_cfg.get('strategy', 'all_layers')).strip().lower()
    pruning_cfg = config.get('network_pruning', {})

    print(f"Configuration loaded from {args.config}")
    print(f"Network Expansion Strategy: {expansion_strategy}")
    print(f"Network Pruning Enabled: {pruning_cfg.get('enabled', False)}")
    print(f"Device: {args.device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("\nLoading BDD-C benchmark")
    benchmark = create_bddc_benchmark(config)
    config['benchmark']['total_classes'] = benchmark.num_classes
    config['benchmark']['class_names'] = benchmark.class_names
    config['benchmark']['num_tasks'] = len(benchmark.task_names)
    config['benchmark']['metric_groups'] = benchmark.metric_groups

    print("\nCreating expandable model")
    model = create_expandable_resnet(
        num_initial_classes=config['benchmark']['total_classes']
    )
    effective_params = model.get_effective_num_parameters() if hasattr(model, 'get_effective_num_parameters') else model.get_num_parameters()
    print(
        f"Model created with {model.get_num_parameters():,} total parameters "
        f"({effective_params:,} effective for budget)"
    )

    print("\nCreating replay buffer")
    buffer_config = config['replay_buffer']
    replay_buffer = ReplayBuffer(
        max_size=buffer_config['max_size'],
        distillation_temp=buffer_config['distillation_temp']
    )

    print("\nCreating trainer")
    trainer = create_trainer(
        config=config,
        model=model,
        replay_buffer=replay_buffer,
        device=args.device,
        use_amca=True
    )

    print(f"\n{'=' * 60}")
    print("Starting continual learning training")
    print(f"{'=' * 60}\n")

    num_tasks = config['benchmark']['num_tasks']
    test_loaders = benchmark.get_all_test_dataloaders()
    last_test_loaders = test_loaders

    for task_id in range(num_tasks):
        train_loader = benchmark.get_task_dataloader(task_id, train=True)
        trainer.train_task(task_id, train_loader, test_loaders)

    validation_cfg = pruning_cfg.get('validation', {})
    efficiency_metrics = None
    if validation_cfg.get('report_efficiency_metrics', True):
        input_shape = infer_input_shape_from_loaders(last_test_loaders, fallback=(1, 3, 224, 224))
        warmup_runs = int(validation_cfg.get('latency_warmup_runs', 5))
        latency_runs = int(validation_cfg.get('latency_runs', 20))
        try:
            efficiency_metrics = compute_inference_efficiency(
                trainer.model,
                device=args.device,
                input_shape=input_shape,
                warmup_runs=warmup_runs,
                measure_runs=latency_runs
            )
            print(
                "Efficiency metrics: "
                f"MACs={efficiency_metrics['macs']:,}, "
                f"FLOPs={efficiency_metrics['flops']:,}, "
                f"latency={efficiency_metrics['latency_ms']:.3f} ms, "
                f"throughput={efficiency_metrics['throughput_fps']:.2f} FPS"
            )
        except Exception as e:
            print(f"[WARNING] Could not compute efficiency metrics: {e}")

    print(f"\n{'=' * 60}")
    print("Training complete. Final summary:")
    print(f"{'=' * 60}\n")
    trainer.metrics.print_summary(num_tasks - 1)

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    results_file = os.path.join(config['logging']['save_dir'], f'results_{timestamp}.txt')
    os.makedirs(config['logging']['save_dir'], exist_ok=True)

    amca_score = 0.0
    if trainer.use_amca and trainer.amca_tester is not None:
        amca_score = trainer.amca_tester.get_summary()['amca']

    total_time = sum(trainer.metrics.task_times.values())
    class_names = list(config['benchmark']['class_names'])
    metric_groups = dict(config['benchmark'].get('metric_groups', {}))

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

        f.write("Training:\n")
        f.write(f"  Batch Size: {config['training']['batch_size']}\n")
        f.write(f"  Epochs per Task: {config['training']['epochs_per_task']}\n")
        f.write(f"  Learning Rate: {config['training']['learning_rate']}\n")
        f.write(f"  Optimizer: {config['training']['optimizer']}\n")
        f.write(f"  Weight Decay: {config['training']['weight_decay']}\n")
        f.write(f"  AMP: {config['training'].get('use_amp', False)}\n")
        f.write("  Gradient Accumulation: False\n")
        f.write(f"  Class Balanced Beta: {config['training'].get('cb_beta', 'N/A')}\n\n")

        f.write("Replay Buffer:\n")
        f.write(f"  Replay Buffer Enabled: {config['replay_buffer'].get('enabled', True)}\n")
        f.write(f"  Max Size: {config['replay_buffer']['max_size']}\n")
        f.write(f"  Distillation Temp: {config['replay_buffer']['distillation_temp']}\n")
        f.write(f"  Replay Batch Size: {config['replay_buffer']['replay_batch_size']}\n")
        f.write(f"  Replay Interval Batches: {config['replay_buffer'].get('replay_interval_batches', 1)}\n")
        f.write(f"  Fuse Replay Forward: {config['replay_buffer'].get('fuse_replay_forward', False)}\n")
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
            f.write("  Layer Allocation: layer2 and layer3 both expanded each task (50%/50%)\n")
        else:
            f.write("  Layer Allocation: budget-aware even distribution across all 4 layers\n")
        f.write("\n")

        f.write("Network Pruning:\n")
        f.write(f"  Enabled: {pruning_cfg.get('enabled', False)}\n")
        f.write(f"  Amount (max per task): {pruning_cfg.get('amount', 0.0)}\n")
        f.write(f"  Warmup Tasks: {pruning_cfg.get('warmup_tasks', 2)}\n")
        f.write(f"  Pruning Interval Tasks: {pruning_cfg.get('interval_tasks', 1)}\n")
        f.write(f"  Min Amount: {pruning_cfg.get('min_amount', 0.0)}\n")
        f.write(f"  Max Amount: {pruning_cfg.get('max_amount', pruning_cfg.get('amount', 0.0))}\n")
        f.write(f"  Scope: {pruning_cfg.get('scope', 'whole_model')}\n")
        f.write(f"  Method: {pruning_cfg.get('method', 'structural')}\n")
        hybrid_cfg = pruning_cfg.get('hybrid', {})
        f.write(f"  Hybrid Whole Model Amount: {hybrid_cfg.get('whole_model_amount', 'N/A')}\n")
        f.write(f"  Hybrid Adapter Amount: {hybrid_cfg.get('adapter_amount', 'N/A')}\n")
        f.write(f"  Hybrid Adapter Scope: {hybrid_cfg.get('adapter_scope', 'N/A')}\n")
        f.write(f"  Importance: {pruning_cfg.get('importance', 'magnitude')}\n")
        f.write(f"  Use Replay For Taylor: {pruning_cfg.get('use_replay_for_taylor', True)}\n")
        f.write(f"  Taylor Replay Batches: {pruning_cfg.get('taylor_replay_batches', 2)}\n")
        f.write(f"  Taylor Train Batches: {pruning_cfg.get('taylor_train_batches', 0)}\n")
        f.write(f"  Taylor Replay Batch Size: {pruning_cfg.get('taylor_replay_batch_size', config['replay_buffer'].get('replay_batch_size', 32))}\n")
        f.write(f"  Rebuild Optimizer After Prune: {pruning_cfg.get('rebuild_optimizer_after_prune', True)}\n")
        f.write(f"  Recompile After Prune: {pruning_cfg.get('recompile_after_prune', True)}\n")
        f.write(f"  Protect Current-Task Adapters: {pruning_cfg.get('protect_current_task_adapters', True)}\n")
        f.write(f"  Protect Classifier: {pruning_cfg.get('protect_classifier', True)}\n")
        f.write(f"  Protect Downsample: {pruning_cfg.get('protect_downsample', True)}\n")
        rollback_cfg = pruning_cfg.get('rollback', {})
        recovery_cfg = pruning_cfg.get('recovery', {})
        f.write(f"  Rollback Enabled: {rollback_cfg.get('enabled', False)}\n")
        f.write(f"  Rollback Max MCA Drop: {rollback_cfg.get('max_mca_drop', 0.01)}\n")
        f.write(f"  Rollback Eval Max Batches: {rollback_cfg.get('eval_max_batches', 20)}\n")
        f.write(f"  Recovery Enabled: {recovery_cfg.get('enabled', True)}\n")
        f.write(f"  Recovery Epochs: {recovery_cfg.get('epochs', 2)}\n")
        f.write(f"  Recovery LR Scale: {recovery_cfg.get('learning_rate_scale', 0.25)}\n")
        f.write(f"  Report Efficiency Metrics: {validation_cfg.get('report_efficiency_metrics', True)}\n")
        f.write(f"  Latency Warmup Runs: {validation_cfg.get('latency_warmup_runs', 5)}\n")
        f.write(f"  Latency Runs: {validation_cfg.get('latency_runs', 20)}\n\n")
        f.write("Logging:\n")
        f.write(f"  Save Dir: {config['logging']['save_dir']}\n\n")

        f.write("FULL RESOLVED CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(yaml.safe_dump(config, sort_keys=True))
        f.write("\n")
        f.write("=" * 80 + "\n\n")

        if trainer.use_amca and trainer.amca_tester is not None:
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
                        f"  During Task {task_id}: "
                        f"Final MCA={task_mcas[-1]:.4f}, Best MCA={max(task_mcas):.4f}\n"
                    )
            f.write("\n")

        f.write("Per-task overall accuracies (History):\n")
        for task_id in range(num_tasks):
            accs = trainer.metrics.task_accuracies.get(task_id, [])
            if accs:
                f.write(f"  End of Task {task_id}: {accs[-1]:.4f}\n")
        f.write("\n")

        f.write("Per-Class Accuracies at End of Each Task:\n")
        for task_id in range(num_tasks):
            class_stats = trainer.metrics.task_class_accuracies.get(task_id)
            if not class_stats:
                continue
            f.write(f"\n  Task {task_id}:\n")
            for class_id, class_name in enumerate(class_names):
                if class_id in class_stats:
                    f.write(f"    {class_name}: {class_stats[class_id]:.4f}\n")
                else:
                    f.write(f"    {class_name}: N/A\n")
        f.write("\n")

        _write_group_sections(f, trainer, num_tasks, class_names, metric_groups)

        if trainer.metrics.memory_usage:
            f.write(f"Final Memory Usage: {trainer.metrics.memory_usage[-1]:.2f} MB\n")
        if trainer.metrics.num_parameters:
            f.write(f"Final Parameters: {trainer.metrics.num_parameters[-1]:,}\n")
        if efficiency_metrics is not None:
            f.write(f"Approx MACs: {int(efficiency_metrics['macs']):,}\n")
            f.write(f"Approx FLOPs: {int(efficiency_metrics['flops']):,}\n")
            f.write(f"Inference Device: {efficiency_metrics['device']}\n")
            f.write(f"Inference Input Shape: {efficiency_metrics['input_shape']}\n")
            f.write(f"Inference Latency: {efficiency_metrics['latency_ms']:.3f} ms\n")
            f.write(f"Inference Throughput: {efficiency_metrics['throughput_fps']:.2f} FPS\n")
        f.write(f"Total Training Time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)\n\n")

        f.write("Network Expansions Per Task:\n")
        for task_id in range(num_tasks):
            expansions = trainer.task_expansions.get(task_id, [])
            if expansions:
                f.write(f"  Task {task_id}: {', '.join(expansions)}\n")
            else:
                f.write(f"  Task {task_id}: none\n")

        f.write("\nNetwork Pruning Per Task:\n")
        pruning_by_task = {entry['task_id']: entry for entry in trainer.pruning_history}
        for task_id in range(num_tasks):
            entry = pruning_by_task.get(task_id)
            if not entry:
                f.write(f"  Task {task_id}: none\n")
                continue
            rollback_suffix = ', rollback=applied' if entry.get('rolled_back', False) else ''
            f.write(
                f"  Task {task_id}: pruned={entry['pruned_params']:,}, "
                f"effective={entry['effective_params_before']:,}->{entry['effective_params_after']:,}, "
                f"amount={entry['amount']:.4f}, scope={entry['scope']}, "
                f"method={entry.get('method', 'unstructured')}, "
                f"importance={entry.get('importance', 'n/a')}{rollback_suffix}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL MODEL PERFORMANCE (End of Training)\n")
        f.write("=" * 80 + "\n")

        final_task_id = num_tasks - 1
        final_overall_acc = trainer.metrics.task_accuracies.get(final_task_id, [])
        f.write(
            f"Final Overall Model Accuracy: "
            f"{final_overall_acc[-1]:.4f}\n" if final_overall_acc else "Final Overall Model Accuracy: N/A\n"
        )

        f.write("\nFinal Per-Class Accuracies:\n")
        final_class_stats = trainer.metrics.task_class_accuracies.get(final_task_id, {})
        if final_class_stats:
            for class_id, class_name in enumerate(class_names):
                if class_id in final_class_stats:
                    f.write(f"  {class_name}: {final_class_stats[class_id]:.4f}\n")
                else:
                    f.write(f"  {class_name}: N/A\n")
        else:
            f.write("  N/A\n")

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
                    f.write(f"  {class_name}: {_avg(values):.4f}\n")
                else:
                    f.write(f"  {class_name}: N/A\n")
        else:
            f.write("  N/A\n")

        _write_final_group_sections(f, trainer, final_task_id, class_names, metric_groups)
        f.write("\n" + "=" * 80 + "\n")

    print(f"\nFinal metrics saved to: {results_file}")

    final_path = os.path.join(config['logging']['save_dir'], f'final_model_{timestamp}.pt')
    trainer.save_checkpoint(final_path, num_tasks - 1)
    latest_path = os.path.join(config['logging']['save_dir'], 'final_model.pt')
    trainer.save_checkpoint(latest_path, num_tasks - 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on BDD-C benchmark')
    parser.add_argument('--config', type=str, default='config/bddc_config.yaml', help='Path to configuration file')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    main(parser.parse_args())
