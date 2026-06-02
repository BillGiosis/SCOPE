"""Shared experiment setup helpers."""

import os
from datetime import datetime
from typing import Iterable, Tuple

import torch
import yaml

from buffers.replay_buffer import ReplayBuffer
from models.expandable_resnet import create_expandable_resnet
from trainers.continual_trainer import create_trainer
from utils.metrics import compute_inference_efficiency


def load_config(config_path: str) -> dict:
    """Load a YAML experiment config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def configure_torch_runtime(seed: int):
    """Apply common torch runtime settings used by both experiments."""
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def infer_input_shape_from_loaders(
    loaders: Iterable,
    fallback: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> Tuple[int, int, int, int]:
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

        if images is not None and images.ndim >= 4:
            return (1, int(images.shape[1]), int(images.shape[2]), int(images.shape[3]))

    return fallback


def create_model_from_config(config: dict):
    """Create the expandable model from benchmark/model config."""
    model_cfg = config.get('model', {})
    model = create_expandable_resnet(
        backbone=model_cfg.get('backbone', 'resnet50d'),
        num_initial_classes=config['benchmark']['total_classes'],
        adapt_downsample_blocks=model_cfg.get('adapt_downsample_blocks', True)
    )
    return model


def create_replay_buffer_from_config(config: dict) -> ReplayBuffer:
    """Create replay buffer from config."""
    buffer_config = config['replay_buffer']
    return ReplayBuffer(
        max_size=buffer_config['max_size'],
        distillation_temp=buffer_config['distillation_temp']
    )


def create_trainer_from_config(config: dict, model, replay_buffer, device: str, use_amca: bool = True):
    """Create the shared continual trainer."""
    return create_trainer(
        config=config,
        model=model,
        replay_buffer=replay_buffer,
        device=device,
        use_amca=use_amca
    )


def run_efficiency_metrics(config: dict, model, device: str, loaders, fallback_shape=(1, 3, 224, 224)):
    """Optionally compute inference efficiency metrics from config."""
    pruning_cfg = config.get('network_pruning', {})
    validation_cfg = pruning_cfg.get('validation', {})
    if not validation_cfg.get('report_efficiency_metrics', True):
        return None

    input_shape = infer_input_shape_from_loaders(loaders, fallback=fallback_shape)
    warmup_runs = int(validation_cfg.get('latency_warmup_runs', 5))
    latency_runs = int(validation_cfg.get('latency_runs', 20))
    try:
        metrics = compute_inference_efficiency(
            model,
            device=device,
            input_shape=input_shape,
            warmup_runs=warmup_runs,
            measure_runs=latency_runs
        )
        print(
            "Efficiency metrics: "
            f"MACs={metrics['macs']:,}, "
            f"FLOPs={metrics['flops']:,}, "
            f"latency={metrics['latency_ms']:.3f} ms, "
            f"throughput={metrics['throughput_fps']:.2f} FPS"
        )
        return metrics
    except Exception as e:
        print(f"[WARNING] Could not compute efficiency metrics: {e}")
        return None


def make_timestamp() -> str:
    return datetime.now().strftime("%d%m%Y_%H%M")


def save_final_checkpoints(trainer, save_dir: str, timestamp: str, task_id: int):
    """Save timestamped and latest final checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, f'final_model_{timestamp}.pt')
    trainer.save_checkpoint(final_path, task_id)
    latest_path = os.path.join(save_dir, 'final_model.pt')
    trainer.save_checkpoint(latest_path, task_id)
