"""
Main training script for CLAD-C benchmark.
"""

import os
import sys

# Suppress CLAD's "No Detectron installation found" message in workers
os.environ['CLAD_SUPPRESS_DETECTRON_MSG'] = '1'

import torch
import torch.multiprocessing as mp
import yaml
import argparse

# Set multiprocessing start method to avoid fork warnings
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.expandable_resnet import create_expandable_resnet
from buffers.replay_buffer import ReplayBuffer
from benchmarks.cladc import create_cladc_benchmark
from trainers.continual_trainer import create_trainer
from utils.metrics import compute_inference_efficiency


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def infer_input_shape_from_loaders(loaders, fallback=(1, 3, 224, 224)):
    """Infer a representative NCHW input shape from available dataloaders."""
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

        if images is None or images.ndim < 4:
            continue

        return (1, int(images.shape[1]), int(images.shape[2]), int(images.shape[3]))

    return fallback


def main(args):

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # Load configuration
    config = load_config(args.config)
    expansion_cfg = config.get('network_expansion', {})
    expansion_strategy = str(expansion_cfg.get('strategy', 'all_layers')).strip().lower()
    pruning_cfg = config.get('network_pruning', {})
    
    print(f"Configuration loaded from {args.config}")
    print(f"Backbone: resnet50d")
    print(f"Network Expansion Strategy: {expansion_strategy}")
    print(f"Network Pruning Enabled: {pruning_cfg.get('enabled', False)}")
    print(f"Device: {args.device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create benchmark
    print("\nLoading CLAD-C benchmark")
    benchmark = create_cladc_benchmark(config)
    
    # Create model
    print("\nCreating expandable ResNet model")
    # Model needs 7 outputs (indices 0-6) since labels are category_id 1-6
    num_classes = config['benchmark']['total_classes']
    model = create_expandable_resnet(
        num_initial_classes=num_classes
    )
    effective_params = model.get_effective_num_parameters() if hasattr(model, 'get_effective_num_parameters') else model.get_num_parameters()
    print(
        f"Model created with {model.get_num_parameters():,} total parameters "
        f"({effective_params:,} effective for budget)"
    )
    
    # Create replay buffer
    print("\nCreating replay buffer")
    buffer_config = config['replay_buffer']
    replay_buffer = ReplayBuffer(
        initial_size=buffer_config['initial_size'],
        max_size=buffer_config['max_size'],
        distillation_temp=buffer_config['distillation_temp']
    )
    
    # Create trainer
    print("\nCreating trainer")
    trainer = create_trainer(
        config=config,
        model=model,
        replay_buffer=replay_buffer,
        device=args.device,
        use_amca=True  # Enable AMCA metric for CLAD-C
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting continual learning training")
    print(f"{'='*60}\n")
    
    num_tasks = config['benchmark']['num_tasks']
    last_test_loaders = None
    
    for task_id in range(num_tasks):
        # Get data loaders
        train_loader = benchmark.get_task_dataloader(task_id, train=True)
        test_loaders = benchmark.get_all_test_dataloaders()
        last_test_loaders = test_loaders
        
        # Train on current task
        trainer.train_task(task_id, train_loader, test_loaders)

    # Efficiency metrics (FLOPs/MACs + latency/throughput)
    validation_cfg = pruning_cfg.get('validation', {})
    efficiency_metrics = None
    if validation_cfg.get('report_efficiency_metrics', True):
        fallback_shape = tuple(pruning_cfg.get('example_input_shape', [1, 3, 224, 224]))
        input_shape = infer_input_shape_from_loaders(last_test_loaders, fallback=fallback_shape)
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
    
    # Final evaluation and summary
    print(f"\n{'='*60}")
    print("Training complete. Final summary:")
    print(f"{'='*60}\n")
    
    trainer.metrics.print_summary(num_tasks - 1)
    
    # Save final metrics to file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    
    # Get AMCA score
    amca_score = 0.0
    if trainer.use_amca and trainer.amca_tester is not None:
        amca_summary = trainer.amca_tester.get_summary()
        amca_score = amca_summary['amca']
    
    # Calculate total training time
    total_time = sum(trainer.metrics.task_times.values())
    total_hours = total_time / 3600
    
    # Create filename with key metrics
    results_filename = f'results_{timestamp}.txt'
    results_file = os.path.join(config['logging']['save_dir'], results_filename)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLAD-C TRAINING RESULTS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AMCA Score: {amca_score:.4f}\n")
        f.write(f"Training Time: {total_hours:.2f} hours ({total_time:.0f} seconds)\n")
        f.write("="*80 + "\n\n")
        
        # Hyperparameters section
        f.write("HYPERPARAMETERS\n")
        f.write("="*80 + "\n")
        f.write(f"Benchmark: {config['benchmark']['name']}\n")
        f.write(f"Number of Tasks: {config['benchmark']['num_tasks']}\n")
        f.write(f"Total Classes: {config['benchmark']['total_classes']}\n")
        f.write(f"Classes per Task: {config['benchmark'].get('num_classes_per_task', 'N/A')}\n\n")

        f.write(f"Training:\n")
        f.write(f"  Batch Size: {config['training']['batch_size']}\n")
        f.write(f"  Epochs per Task: {config['training']['epochs_per_task']}\n")
        f.write(f"  Learning Rate: {config['training']['learning_rate']}\n")
        f.write(f"  Optimizer: {config['training']['optimizer']}\n")
        f.write(f"  Weight Decay: {config['training']['weight_decay']}\n")
        f.write(f"  Eval Frequency: {config['training']['eval_frequency']}\n")
        f.write(f"  AMP: {config['training'].get('use_amp', False)}\n")
        f.write(f"  Gradient Accumulation: {config['training'].get('use_gradient_accumulation', False)}\n")
        f.write(f"  Accumulation Steps: {config['training'].get('accumulation_steps', 1)}\n")
        f.write(f"  Class Balanced Beta: {config['training'].get('cb_beta', 'N/A')}\n")
        f.write("\n")
        
        f.write(f"Replay Buffer:\n")
        f.write(f"  Replay Buffer Enabled: {config['replay_buffer'].get('enabled', True)}\n")
        f.write(f"  Max Size: {config['replay_buffer']['max_size']}\n")
        f.write(f"  Initial Size: {config['replay_buffer']['initial_size']}\n")
        f.write(f"  Distillation Temp: {config['replay_buffer']['distillation_temp']}\n")
        f.write(f"  Replay Batch Size: {config['replay_buffer']['replay_batch_size']}\n\n")
        
        f.write(f"Data Loading:\n")
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
        f.write(f"  Budget Driven: {pruning_cfg.get('budget_driven', True)}\n")
        f.write(f"  Min Amount: {pruning_cfg.get('min_amount', 0.0)}\n")
        f.write(f"  Max Amount: {pruning_cfg.get('max_amount', pruning_cfg.get('amount', 0.0))}\n")
        f.write(f"  Scope: {pruning_cfg.get('scope', 'whole_model')}\n")
        f.write(f"  Method: {pruning_cfg.get('method', 'structural')}\n")
        f.write(f"  Rebuild Optimizer After Prune: {pruning_cfg.get('rebuild_optimizer_after_prune', True)}\n")
        f.write(f"  Recompile After Prune: {pruning_cfg.get('recompile_after_prune', True)}\n")
        f.write(f"  Example Input Shape: {pruning_cfg.get('example_input_shape', [1, 3, 224, 224])}\n")
        f.write(f"  Protect Current-Task Adapters: {pruning_cfg.get('protect_current_task_adapters', True)}\n")
        f.write(f"  Protect Classifier: {pruning_cfg.get('protect_classifier', True)}\n")
        f.write(f"  Protect Downsample: {pruning_cfg.get('protect_downsample', True)}\n")
        recovery_cfg = pruning_cfg.get('recovery', {})
        f.write(f"  Recovery Enabled: {recovery_cfg.get('enabled', True)}\n")
        f.write(f"  Recovery Epochs: {recovery_cfg.get('epochs', 2)}\n")
        f.write(f"  Recovery LR Scale: {recovery_cfg.get('learning_rate_scale', 0.25)}\n")
        f.write(f"  Report Efficiency Metrics: {validation_cfg.get('report_efficiency_metrics', True)}\n")
        f.write(f"  Latency Warmup Runs: {validation_cfg.get('latency_warmup_runs', 5)}\n")
        f.write(f"  Latency Runs: {validation_cfg.get('latency_runs', 20)}\n")
        f.write("\n")
        
        f.write("="*80 + "\n\n")
        
        # AMCA metrics
        if trainer.use_amca and trainer.amca_tester is not None:
            amca_summary = trainer.amca_tester.get_summary()
            f.write(f"AMCA Score: {amca_summary['amca']:.4f}\n")
            f.write(f"Final MCA (Mean Class Acc): {amca_summary['final_mca']:.4f}\n")
            f.write(f"Number of Evaluations: {amca_summary['num_evaluations']}\n\n")
            
            # Show MCA evolution during training
            f.write("MCA at each evaluation (grouped by training task):\n")
            mca_list = amca_summary['mean_class_accuracies']
            evals_per_task = len(mca_list) // num_tasks if num_tasks > 0 else len(mca_list)
            for task in range(num_tasks):
                start_idx = task * evals_per_task
                end_idx = min((task + 1) * evals_per_task, len(mca_list))
                task_mcas = mca_list[start_idx:end_idx]
                if task_mcas:
                    final_mca_task = task_mcas[-1]
                    best_mca_task = max(task_mcas)
                    f.write(f"  During Task {task}: Final MCA={final_mca_task:.4f}, Best MCA={best_mca_task:.4f}\n")
            f.write("\n")
        
        # Per-task accuracies (history)
        f.write("Per-task overall accuracies (History):\n")
        for task_id in range(num_tasks):
            accs = trainer.metrics.task_accuracies.get(task_id, [])
            if accs:
                f.write(f"  End of Task {task_id}: {accs[-1]:.4f}\n")
        f.write("\n")
        
        # Per-task per-class accuracies
        class_names = ['unused', 'pedestrian', 'cyclist', 'car', 'truck', 'tram', 'tricycle']
        f.write("Per-Class Accuracies at End of Each Task:\n")
        for task_id in range(num_tasks):
            if task_id in trainer.metrics.task_class_accuracies:
                f.write(f"\n  Task {task_id}:\n")
                class_stats = trainer.metrics.task_class_accuracies[task_id]
                for class_id in sorted(class_stats.keys()):
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    acc = class_stats[class_id]
                    f.write(f"    {class_name}: {acc:.4f}\n")
        f.write("\n")

        # Per-task per-location per-class accuracies
        if trainer.metrics.task_location_class_accuracies:
            f.write("Per-Location Per-Class Accuracies at End of Each Task:\n")
            for task_id in range(num_tasks):
                task_loc_stats = trainer.metrics.task_location_class_accuracies.get(task_id, {})
                if not task_loc_stats:
                    continue
                f.write(f"\n  Task {task_id}:\n")
                for location in ['Citystreet', 'Countryroad', 'Highway']:
                    if location not in task_loc_stats:
                        continue
                    loc_class_stats = task_loc_stats[location]
                    f.write(f"    {location}:\n")
                    for class_id in range(1, len(class_names)):
                        class_name = class_names[class_id]
                        if class_id in loc_class_stats:
                            f.write(f"      {class_name}: {loc_class_stats[class_id]:.4f}\n")
                        else:
                            f.write(f"      {class_name}: N/A\n")
                    loc_mca = float(sum(loc_class_stats.values()) / len(loc_class_stats)) if loc_class_stats else 0.0
                    f.write(f"      MCA: {loc_mca:.4f}\n")
            f.write("\n")

            f.write("Per-Location AMCA (across all evaluation points):\n")
            for location in ['Citystreet', 'Countryroad', 'Highway']:
                hist = trainer.metrics.location_mca_history.get(location, [])
                if hist:
                    f.write(f"  {location}: {float(sum(hist) / len(hist)):.4f}\n")
                else:
                    f.write(f"  {location}: N/A\n")
            f.write("\n")
        
        # Model statistics
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
        total_time = sum(trainer.metrics.task_times.values())
        f.write(f"Total Training Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n\n")
        
        # Per-task expansion breakdown
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
            method = entry.get('method', 'unstructured')
            f.write(
                f"  Task {task_id}: pruned={entry['pruned_params']:,}, "
                f"effective={entry['effective_params_before']:,}->{entry['effective_params_after']:,}, "
                f"amount={entry['amount']:.4f}, scope={entry['scope']}, method={method}\n"
            )
        
        f.write("\n" + "="*80 + "\n")
        f.write("FINAL MODEL PERFORMANCE (End of Training)\n")
        f.write("="*80 + "\n")
        
        class_names = ['unused', 'pedestrian', 'cyclist', 'car', 'truck', 'tram', 'tricycle']
        num_classes = len(class_names)
        final_task_id = num_tasks - 1
        
        # Final Overall Accuracy (Standard Accuracy, NOT MCA)
        if final_task_id in trainer.metrics.task_accuracies:
            final_overall_acc = trainer.metrics.task_accuracies[final_task_id][-1]
            f.write(f"Final Overall Model Accuracy: {final_overall_acc:.4f}\n")
        else:
            f.write("Final Overall Model Accuracy: N/A\n")
            
        # Final Per-Class Accuracy
        f.write("\nFinal Per-Class Accuracies:\n")
        if final_task_id in trainer.metrics.task_class_accuracies:
            final_class_stats = trainer.metrics.task_class_accuracies[final_task_id]
            for class_id in range(num_classes):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                if class_id in final_class_stats:
                    acc = final_class_stats[class_id]
                    f.write(f"  {class_name}: {acc:.4f}\n")
                else:
                    f.write(f"  {class_name}: N/A\n")
        else:
            f.write("  N/A\n")
        
        # Average Per-Class Accuracies (across all tasks)
        f.write("\nAverage Per-Class Accuracies (across all evaluations):\n")
        if trainer.metrics.task_class_accuracies:
            # Collect accuracies for each class across all tasks
            class_accuracies_over_time = {class_id: [] for class_id in range(num_classes)}
            
            for task_id in range(num_tasks):
                if task_id in trainer.metrics.task_class_accuracies:
                    class_stats = trainer.metrics.task_class_accuracies[task_id]
                    for class_id in range(num_classes):
                        if class_id in class_stats:
                            class_accuracies_over_time[class_id].append(class_stats[class_id])
            
            # Compute and write averages
            for class_id in range(num_classes):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                if class_accuracies_over_time[class_id]:
                    avg_acc = sum(class_accuracies_over_time[class_id]) / len(class_accuracies_over_time[class_id])
                    f.write(f"  {class_name}: {avg_acc:.4f}\n")
                else:
                    f.write(f"  {class_name}: N/A\n")
        else:
            f.write("  N/A\n")

        # Final per-location per-class metrics and location AMCA
        f.write("\nFinal Per-Location Per-Class Accuracies:\n")
        if final_task_id in trainer.metrics.task_location_class_accuracies:
            final_loc_stats = trainer.metrics.task_location_class_accuracies[final_task_id]
            for location in ['Citystreet', 'Countryroad', 'Highway']:
                if location not in final_loc_stats:
                    f.write(f"  {location}: N/A\n")
                    continue
                f.write(f"  {location}:\n")
                loc_class_stats = final_loc_stats[location]
                for class_id in range(1, num_classes):
                    class_name = class_names[class_id]
                    if class_id in loc_class_stats:
                        f.write(f"    {class_name}: {loc_class_stats[class_id]:.4f}\n")
                    else:
                        f.write(f"    {class_name}: N/A\n")
                loc_mca = float(sum(loc_class_stats.values()) / len(loc_class_stats)) if loc_class_stats else 0.0
                f.write(f"    MCA: {loc_mca:.4f}\n")
        else:
            f.write("  N/A\n")

        f.write("\nPer-Location AMCA (across all evaluations):\n")
        for location in ['Citystreet', 'Countryroad', 'Highway']:
            hist = trainer.metrics.location_mca_history.get(location, [])
            if hist:
                f.write(f"  {location}: {float(sum(hist) / len(hist)):.4f}\n")
            else:
                f.write(f"  {location}: N/A\n")

        # Custom task-combination metrics
        combo_groups = {
            'Day Combinations (Tasks 1,2,4)': [0, 2, 4],
            'Night Combinations (Tasks 2,3,6)': [1, 3, 5],
        }
        f.write("\nTask-Combination Per-Class + AMCA:\n")
        for combo_name, combo_task_ids in combo_groups.items():
            valid_task_ids = [tid for tid in combo_task_ids if tid in trainer.metrics.task_class_accuracies]
            f.write(f"  {combo_name}:\n")
            if not valid_task_ids:
                f.write("    N/A\n")
                continue

            combo_class_values = {cid: [] for cid in range(1, num_classes)}
            for tid in valid_task_ids:
                class_stats = trainer.metrics.task_class_accuracies[tid]
                for cid in range(1, num_classes):
                    if cid in class_stats:
                        combo_class_values[cid].append(class_stats[cid])

            combo_means = {}
            for cid in range(1, num_classes):
                vals = combo_class_values[cid]
                if vals:
                    combo_means[cid] = float(sum(vals) / len(vals))

            if combo_means:
                combo_amca = float(sum(combo_means.values()) / len(combo_means))
                for cid in range(1, num_classes):
                    class_name = class_names[cid]
                    if cid in combo_means:
                        f.write(f"    {class_name}: {combo_means[cid]:.4f}\n")
                    else:
                        f.write(f"    {class_name}: N/A\n")
                f.write(f"    AMCA: {combo_amca:.4f}\n")
            else:
                f.write("    N/A\n")
            
        f.write("="*80 + "\n")
    
    print(f"\nFinal metrics saved to: {results_file}")
    
    # Save final model
    final_path = os.path.join(config['logging']['save_dir'], 'final_model.pt')
    trainer.save_checkpoint(final_path, num_tasks - 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on CLAD-C benchmark')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/cladc_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    main(args)
