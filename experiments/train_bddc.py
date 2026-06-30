"""Main training script for BDD-C benchmark."""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from benchmarks.bddc import create_bddc_benchmark
from utils.experiment import (
    configure_torch_runtime,
    create_model_from_config,
    create_replay_buffer_from_config,
    create_trainer_from_config,
    load_config,
    make_timestamp,
    save_final_checkpoints,
)
from utils.reports_bddc import write_bddc_results


def main(args):
    configure_torch_runtime(args.seed)

    config = load_config(args.config)
    config.setdefault('training', {})
    config['training']['use_gradient_accumulation'] = False
    config['training']['accumulation_steps'] = 1

    expansion_cfg = config.get('network_expansion', {})
    reallocation_cfg = config.get('parameter_reallocation', {})
    model_cfg = config.get('model', {})
    expansion_strategy = str(expansion_cfg.get('strategy', 'all_layers')).strip().lower()

    print(f"Configuration loaded from {args.config}")
    print(f"Adapt Downsample Blocks: {model_cfg.get('adapt_downsample_blocks', True)}")
    print(f"Network Expansion Strategy: {expansion_strategy}")
    print(f"Parameter Reallocation Enabled: {reallocation_cfg.get('enabled', False)}")
    print(f"Device: {args.device}")

    print("\nLoading BDD-C benchmark")
    benchmark = create_bddc_benchmark(config)
    config['benchmark']['total_classes'] = benchmark.num_classes
    config['benchmark']['class_names'] = benchmark.class_names
    config['benchmark']['num_tasks'] = len(benchmark.task_names)
    config['benchmark']['metric_groups'] = benchmark.metric_groups

    print("\nCreating expandable model")
    model = create_model_from_config(config)
    effective_params = (
        model.get_effective_num_parameters()
        if hasattr(model, 'get_effective_num_parameters')
        else model.get_num_parameters()
    )
    print(
        f"Model created with {model.get_num_parameters():,} total parameters "
        f"({effective_params:,} effective for budget)"
    )

    print("\nCreating replay buffer")
    replay_buffer = create_replay_buffer_from_config(config)

    print("\nCreating trainer")
    trainer = create_trainer_from_config(
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
    for task_id in range(num_tasks):
        train_loader = benchmark.get_task_dataloader(task_id, train=True)
        trainer.train_task(task_id, train_loader, test_loaders)

    print(f"\n{'=' * 60}")
    print("Training complete. Final summary:")
    print(f"{'=' * 60}\n")
    trainer.metrics.print_summary(num_tasks - 1)

    timestamp = make_timestamp()
    results_file = write_bddc_results(config, trainer, timestamp)
    print(f"\nFinal metrics saved to: {results_file}")

    save_final_checkpoints(
        trainer,
        save_dir=config['logging']['save_dir'],
        timestamp=timestamp,
        task_id=num_tasks - 1
    )


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
