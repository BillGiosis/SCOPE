"""Shared experiment setup helpers."""

import os
from datetime import datetime

import torch
import yaml

from buffers.replay_buffer import ReplayBuffer
from models.expandable_resnet import create_expandable_resnet
from trainers.continual_trainer import create_trainer


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


def make_timestamp() -> str:
    return datetime.now().strftime("%d%m%Y_%H%M")


def save_final_checkpoints(trainer, save_dir: str, timestamp: str, task_id: int):
    """Save timestamped and latest final checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, f'final_model_{timestamp}.pt')
    trainer.save_checkpoint(final_path, task_id)
    latest_path = os.path.join(save_dir, 'final_model.pt')
    trainer.save_checkpoint(latest_path, task_id)
