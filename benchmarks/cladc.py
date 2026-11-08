"""
CLAD-C Benchmark Dataset Loader.

Uses the official CLAD package from: https://github.com/VerwimpEli/CLAD
Wraps official CLAD datasets to work with our training pipeline.
"""

import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Dict, Optional
import numpy as np

from utils.constants import CLADC_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from utils.sampling import make_inverse_frequency_sampler

_CLAD_MODULE = None


@contextmanager
def _suppress_clad_detectron_message():
    if os.environ.get('CLAD_SUPPRESS_DETECTRON_MSG', '0') != '1':
        yield
        return

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        yield


def _import_clad(clad_repo_root: str):
    """Import the official CLAD package after the selected config is loaded."""
    global _CLAD_MODULE
    if _CLAD_MODULE is not None:
        return _CLAD_MODULE

    if clad_repo_root and clad_repo_root not in sys.path:
        sys.path.insert(0, clad_repo_root)

    with _suppress_clad_detectron_message():
        import clad
        _CLAD_MODULE = clad
        return _CLAD_MODULE


class CLADCDatasetWrapper(Dataset):
    """
    Wrapper around official CLAD-C dataset
    """

    def __init__(
        self,
        official_dataset,
        transform=None
    ):
        self.official_dataset = official_dataset
        self.custom_transform = transform

        self.is_concat = hasattr(official_dataset, 'datasets')

        self.resize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.official_dataset)

    def _resolve_underlying_dataset_and_obj_id(self, idx):
        """Return (underlying_dataset, obj_id) for both Concat and non-Concat datasets."""
        if self.is_concat:
            sample_idx = idx
            for d in self.official_dataset.datasets:
                if sample_idx < len(d):
                    obj_id = d.ids[sample_idx]
                    return d, obj_id
                sample_idx -= len(d)
            raise IndexError(f"Index {idx} out of range for concatenated CLAD-C dataset.")
        else:
            obj_id = self.official_dataset.ids[idx]
            return self.official_dataset, obj_id

    @staticmethod
    def _get_location_from_underlying(underlying, obj_id: int) -> str:
        """Read location metadata from official CLAD annotations for one object."""
        img_id = underlying.obj_annotations[obj_id]['image_id']
        img_ann = underlying.img_annotations[img_id]
        return img_ann.get('location', 'Unknown')

    def __getitem__(self, idx):
        underlying, obj_id = self._resolve_underlying_dataset_and_obj_id(idx)
        image = underlying._load_image(obj_id)
        label = underlying._load_target(obj_id)

        if self.custom_transform:
            image = self.custom_transform(image)
        else:
            image = self.resize_transform(image)

        return image, label

    def get_location(self, idx: int) -> str:
        """Return location domain for a sample index."""
        underlying, obj_id = self._resolve_underlying_dataset_and_obj_id(idx)
        return self._get_location_from_underlying(underlying, obj_id)

    @property
    def targets(self):
        """Return all labels for sampling purposes."""
        if self.is_concat:
            # Concatenate targets from all underlying datasets
            all_targets = []
            for d in self.official_dataset.datasets:
                all_targets.extend(d.targets)
            return all_targets
        return self.official_dataset.targets


class CLADCBenchmark:
    """
    CLAD-C Benchmark manager using official CLAD package.

    CLAD-C Configuration:
    - 6 classes (pedestrian, cyclist, car, truck, tram, tricycle)
    - 6 temporal tasks based on date/period
    - Natural chronological data stream
    - Evaluation using AMCA metric
    """

    def __init__(
        self,
        root_dir: str,
        clad_repo_root: str,
        batch_size: int = 32,
        num_workers: int = 8,
        prefetch_factor: int = 2
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.input_size = CLADC_INPUT_SIZE
        resize_after_normalize = transforms.Resize(
            (self.input_size, self.input_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=False
        )

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            resize_after_normalize
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            resize_after_normalize
        ])

        print("Loading official CLAD-C datasets")
        clad_module = _import_clad(clad_repo_root)
        with _suppress_clad_detectron_message():
            self._train_sets = clad_module.get_cladc_train(root_dir, img_size=64)
            self._val_set = clad_module.get_cladc_val(root_dir, img_size=64)
        self._cached_test_loader: Optional[DataLoader] = None
        print(f"Loaded {len(self._train_sets)} training tasks")

    def get_task_dataloader(
        self,
        task_id: int,
        train: bool = True,
        batch_size: int = None
    ) -> DataLoader:
        """
        Get DataLoader for a specific task.

        Args:
            task_id: Task identifier (0-5)
            train: If True, return training data

        Returns:
            DataLoader for the task
        """
        if batch_size is None:
            batch_size = self.batch_size if train else 32

        if train:
            if task_id >= len(self._train_sets):
                raise ValueError(f"Task {task_id} not available. Only {len(self._train_sets)} tasks.")
            official_dataset = self._train_sets[task_id]
            transform = self.train_transform
        else:
            official_dataset = self._val_set
            transform = self.test_transform

        dataset = CLADCDatasetWrapper(
            official_dataset=official_dataset,
            transform=transform
        )

        sampler = None
        shuffle = train
        if train and len(dataset) > 0:
            sampler, class_counts = make_inverse_frequency_sampler(dataset.targets)
            shuffle = False

            print(f"  Class counts: {class_counts.tolist()}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False
        )

        return dataloader

    def get_all_test_dataloaders(self) -> List[DataLoader]:
        """Get test dataloaders for all tasks (for evaluation)."""
        if self._cached_test_loader is None:
            self._cached_test_loader = self.get_task_dataloader(0, train=False, batch_size=32)
        return [self._cached_test_loader]

    def get_task_info(self, task_id: int) -> Dict:
        """Get information about a specific task."""
        if task_id >= len(self._train_sets):
            raise ValueError(f"Task {task_id} not available.")

        train_dataset = self._train_sets[task_id]

        train_labels = train_dataset.targets
        num_classes = len(set(train_labels))

        class_dist = np.bincount(train_labels, minlength=7).tolist()

        return {
            'task_id': task_id,
            'num_train_samples': len(train_dataset),
            'num_test_samples': len(self._val_set),
            'num_classes': num_classes,
            'class_distribution': class_dist
        }


def create_cladc_benchmark(config: Dict) -> CLADCBenchmark:
    """
    Factory function to create CLAD-C benchmark from config.

    Args:
        config: Configuration dictionary

    Returns:
        CLADCBenchmark instance
    """
    benchmark_config = config.get('benchmark', {})
    training_config = config.get('training', {})
    data_loading_config = config.get('data_loading', {})

    if 'data_root' not in benchmark_config:
        raise ValueError("Missing benchmark.data_root in config.")
    
    benchmark = CLADCBenchmark(
        root_dir=benchmark_config['data_root'],
        clad_repo_root=benchmark_config.get('clad_repo_root', ''),
        batch_size=training_config.get('batch_size', 32),
        num_workers=data_loading_config.get('num_workers', 4),
        prefetch_factor=data_loading_config.get('prefetch_factor', 2)
    )

    return benchmark
