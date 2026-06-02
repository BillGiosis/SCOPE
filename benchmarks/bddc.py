"""
BDD-C benchmark dataset loader.

Consumes pre-cropped task folders plus precomputed per-task manifests from:
    <data_root>/manifests/<split>/<task_name>.json
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.constants import BDDC_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from utils.sampling import make_inverse_frequency_sampler


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def _load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


class BDDCDataset(Dataset):
    """Dataset for one BDD-C task split backed by a precomputed manifest."""

    _GROUP_INDEX = {
        'weather': 2,
        'timeofday': 3,
    }

    def __init__(
        self,
        samples: Sequence[Tuple[str, int, str, str]],
        class_names: Sequence[str],
        transform=None,
        task_name: Optional[str] = None
    ):
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform
        self.task_name = task_name or ''
        self._targets = [sample[1] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, _, _ = self.samples[idx]
        image = _load_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def targets(self):
        return self._targets

    def get_group_value(self, idx: int, group_name: str):
        field_idx = self._GROUP_INDEX.get(group_name)
        if field_idx is None:
            return None
        return self.samples[idx][field_idx]


class BDDCBenchmark:
    """
    Benchmark manager for BDD-C using pre-cropped object images and manifests.

    Expected structure:
      - <root>/train/task_*
      - <root>/val/task_*
      - <root>/test/task_*
      - <root>/manifests/<split>/<task_name>.json
      - <root>/manifests/metadata.json
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        eval_split: str = 'val'
    ):
        self.root_dir = Path(root_dir)
        self.manifest_root = self.root_dir / 'manifests'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.eval_split = str(eval_split).strip().lower()

        self.train_transform = transforms.Compose([
            transforms.Resize((BDDC_INPUT_SIZE, BDDC_INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((BDDC_INPUT_SIZE, BDDC_INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        train_task_dirs = self._get_task_dirs('train')
        eval_task_dirs = self._get_task_dirs(self.eval_split)
        if not train_task_dirs:
            raise ValueError(f"No BDD-C training task folders found under {self.root_dir / 'train'}")
        if not eval_task_dirs:
            raise ValueError(f"No BDD-C evaluation task folders found under {self.root_dir / self.eval_split}")

        metadata_path = self.manifest_root / 'metadata.json'
        if not metadata_path.is_file():
            raise ValueError(
                f"Missing BDD-C metadata manifest: {metadata_path}. "
                "Run scripts/bddc_split.py first."
            )

        metadata = _load_json(metadata_path)
        self.class_names = list(metadata.get('class_names', []))
        if not self.class_names:
            raise ValueError(f"No class_names found in {metadata_path}")
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.metric_groups = {
            str(group_name): [str(name) for name in names]
            for group_name, names in (metadata.get('metric_groups', {}) or {}).items()
        }

        print("Loading BDD-C datasets")
        self.task_names = [task_name for task_name, _ in train_task_dirs]
        self.eval_task_names = [task_name for task_name, _ in eval_task_dirs]
        self._train_sets = [
            self._build_dataset(split='train', task_name=task_name, transform=self.train_transform)
            for task_name, _ in train_task_dirs
        ]
        self._eval_sets = [
            self._build_dataset(split=self.eval_split, task_name=task_name, transform=self.test_transform)
            for task_name, _ in eval_task_dirs
        ]

        print(
            f"Loaded {len(self._train_sets)} training tasks, "
            f"{len(self._eval_sets)} evaluation tasks, "
            f"{self.num_classes} classes"
        )

    @staticmethod
    def _task_sort_key(task_name: str):
        match = re.search(r'(\d+)', task_name)
        task_num = int(match.group(1)) if match else sys.maxsize
        return (task_num, task_name)

    def _get_task_dirs(self, split: str) -> List[Tuple[str, str]]:
        split_dir = self.root_dir / split
        if not split_dir.is_dir():
            return []

        task_dirs = []
        for entry in os.scandir(split_dir):
            if entry.is_dir():
                task_dirs.append((entry.name, entry.path))
        return sorted(task_dirs, key=lambda item: self._task_sort_key(item[0]))

    def _manifest_path(self, split: str, task_name: str) -> Path:
        return self.manifest_root / split / f'{task_name}.json'

    def _load_manifest_samples(self, split: str, task_name: str) -> List[Tuple[str, int, str, str]]:
        manifest_path = self._manifest_path(split, task_name)
        if not manifest_path.is_file():
            raise ValueError(
                f"Missing BDD-C manifest for split='{split}' task='{task_name}': {manifest_path}. "
                "Run scripts/bddc_split.py first."
            )

        records = _load_json(manifest_path)
        samples = []
        for record in records:
            category = str(record.get('label', record.get('category', '')))
            if category not in self.class_to_idx:
                continue

            image_rel = record.get('image')
            if not image_rel:
                continue
            image_path = self.root_dir / image_rel
            if not image_path.is_file():
                continue

            weather = str(record.get('weather', 'unknown'))
            timeofday = str(record.get('timeofday', 'unknown'))
            samples.append((
                str(image_path),
                self.class_to_idx[category],
                weather,
                timeofday,
            ))
        print(f"  {split}/{task_name}: loaded {len(samples)} samples from manifest")
        return samples

    def _build_dataset(self, split: str, task_name: str, transform) -> BDDCDataset:
        return BDDCDataset(
            samples=self._load_manifest_samples(split=split, task_name=task_name),
            class_names=self.class_names,
            transform=transform,
            task_name=task_name
        )

    def get_task_dataloader(
        self,
        task_id: int,
        train: bool = True,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size if train else 32

        datasets = self._train_sets if train else self._eval_sets
        if task_id >= len(datasets):
            raise ValueError(f"Task {task_id + 1} not available. Only {len(datasets)} tasks.")

        dataset = datasets[task_id]
        sampler = None
        shuffle = train

        if train and len(dataset) > 0:
            sampler, class_counts = make_inverse_frequency_sampler(
                dataset.targets,
                min_classes=self.num_classes
            )
            shuffle = False

            print(f"  Class counts: {class_counts.tolist()}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def get_all_test_dataloaders(self) -> List[DataLoader]:
        return [
            self.get_task_dataloader(task_id, train=False, batch_size=32)
            for task_id in range(len(self._eval_sets))
        ]

    def get_task_info(self, task_id: int) -> Dict:
        if task_id >= len(self._train_sets):
            raise ValueError(f"Task {task_id + 1} not available.")

        train_dataset = self._train_sets[task_id]
        class_dist = np.bincount(train_dataset.targets, minlength=self.num_classes).tolist()
        return {
            'task_id': task_id + 1,
            'task_name': self.task_names[task_id] if task_id < len(self.task_names) else f'task_{task_id + 1}',
            'num_train_samples': len(train_dataset),
            'num_test_samples': len(self._eval_sets[task_id]) if task_id < len(self._eval_sets) else 0,
            'num_classes': self.num_classes,
            'class_distribution': class_dist,
        }


def create_bddc_benchmark(config: Dict) -> BDDCBenchmark:
    """Factory function to create a BDD-C benchmark from config."""
    benchmark_config = config.get('benchmark', {})
    training_config = config.get('training', {})
    data_loading_config = config.get('data_loading', {})

    if 'data_root' not in benchmark_config:
        raise ValueError("Missing benchmark.data_root in config.")

    return BDDCBenchmark(
        root_dir=benchmark_config['data_root'],
        batch_size=training_config.get('batch_size', 32),
        num_workers=data_loading_config.get('num_workers', 4),
        prefetch_factor=data_loading_config.get('prefetch_factor', 2),
        pin_memory=data_loading_config.get('pin_memory', True),
        eval_split=benchmark_config.get('eval_split', 'val')
    )
