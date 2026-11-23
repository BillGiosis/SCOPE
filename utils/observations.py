"""Run provenance, phase costs and evaluation observations."""

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
import time
import uuid

import numpy as np
import torch


def process_memory_stats():
    """Linux parent-process RSS; the high-water mark is lifetime, not per phase."""
    if not sys.platform.startswith('linux'):
        return None
    try:
        import resource
        resident_pages = int(Path('/proc/self/statm').read_text().split()[1])
        return {'current_rss_bytes': resident_pages * os.sysconf('SC_PAGE_SIZE'),
                'lifetime_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                'scope': 'parent process; excludes live data-loader worker RSS'}
    except (OSError, ValueError, IndexError):
        return None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def observation_context(model, loader=None):
    """Preserve parent RNG, loader generators and module modes during extra evaluation.

    Use deterministic evaluation transforms. Worker-local random augmentation state
    cannot be restored by this context.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = {module: module.training for module in model.modules()}
    generators = {}
    for owner in (loader, getattr(loader, 'sampler', None)):
        generator = getattr(owner, 'generator', None)
        if generator is not None:
            generators[generator] = generator.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for generator, state in generators.items():
            generator.set_state(state)
        for module, training in modes.items():
            module.training = training


def _git(path, *arguments):
    try:
        result = subprocess.run(['git', '-C', str(path), *arguments],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ordered_ids_digest(dataset):
    if hasattr(dataset, 'datasets'):
        identifiers = [_ordered_ids_digest(child) for child in dataset.datasets]
    elif hasattr(dataset, 'ids'):
        identifiers = [str(identifier) for identifier in dataset.ids]
    else:
        return None
    return hashlib.sha256(json.dumps(identifiers, separators=(',', ':')).encode()).hexdigest()


def _metadata_targets(dataset):
    if hasattr(dataset, 'datasets'):
        children = [_metadata_targets(child) for child in dataset.datasets]
        return np.concatenate(children) if all(child is not None for child in children) and children else None
    targets = getattr(dataset, 'targets', None)
    if targets is None:
        return None
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    return np.asarray(targets, dtype=np.int64)


def start_run(config, seed, device, config_path, benchmark, model):
    """Write the effective config, code/data fingerprints and runtime before training."""
    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    config.setdefault('run', {}).update({'id': run_id, 'seed': int(seed)})
    source = {}
    for directory in ('models', 'trainers', 'buffers', 'benchmarks', 'utils', 'experiments', 'scripts'):
        for path in sorted((root / directory).glob('*.py')):
            source[str(path.relative_to(root))] = _sha256(path)
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'torch-pruning', 'numpy', 'PyYAML'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    data_files = {}
    manifest_root = getattr(benchmark, 'manifest_root', None)
    if manifest_root is not None:
        for path in sorted(Path(manifest_root).rglob('*.json')):
            data_files[str(path.relative_to(manifest_root))] = _sha256(path)
    # Metadata only: do not load images or iterate randomized training loaders.
    datasets = {}
    for name in ('_train_sets', '_eval_sets'):
        entries = []
        for dataset in getattr(benchmark, name, []):
            targets = _metadata_targets(dataset)
            entries.append({'name': getattr(dataset, 'task_name', ''), 'samples': len(dataset),
                            'label_counts': np.bincount(targets).tolist() if targets is not None else None,
                            'ordered_object_ids_sha256': _ordered_ids_digest(dataset),
                            'ordered_labels_sha256': (hashlib.sha256(targets.tobytes()).hexdigest()
                                                      if targets is not None else None)})
        datasets[name] = entries
    if hasattr(benchmark, '_val_set'):
        validation = benchmark._val_set
        datasets['common_validation_samples'] = len(validation)
        targets = _metadata_targets(validation)
        datasets['common_validation_ordered_labels_sha256'] = (
            hashlib.sha256(targets.tobytes()).hexdigest() if targets is not None else None
        )
        datasets['common_validation_ordered_object_ids_sha256'] = _ordered_ids_digest(validation)
    clad_root = config.get('benchmark', {}).get('clad_repo_root')
    cuda = str(device).startswith('cuda') and torch.cuda.is_available()
    pretrained = getattr(model, 'pretrained_metadata', None)
    manifest = {
        'run_id': run_id, 'seed': int(seed), 'config_path': str(Path(config_path).resolve()),
        'effective_config': copy.deepcopy(config), 'git_commit': _git(root, 'rev-parse', 'HEAD'),
        'git_status': _git(root, 'status', '--short'), 'source_sha256': source,
        'data_manifest_sha256': data_files, 'dataset_metadata': datasets,
        'clad_git_commit': _git(clad_root, 'rev-parse', 'HEAD') if clad_root else None,
        'python': sys.version, 'packages': versions, 'device': str(device),
        'gpu': torch.cuda.get_device_name(torch.device(device)) if cuda else None,
        'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'runtime': {'matmul_precision': torch.get_float32_matmul_precision(),
                    'seeded_rngs': ['torch CPU', 'torch CUDA when available'],
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'torch_compile_requested': config.get('training', {}).get('use_torch_compile', True)},
        'pretrained': pretrained,
        'limitations': 'Fingerprints cover code and available manifests/labels, not all image bytes. '
                      'Recording a seed does not guarantee bitwise CUDA reproducibility.',
    }
    path = Path(config['logging']['save_dir']) / f'run_{run_id}' / 'manifest.json'
    config['run']['artifact_dir'] = str(path.parent)
    manifest['effective_config'] = copy.deepcopy(config)
    write_json(path, manifest)
    return str(path.parent)


class RunObservations:
    """Collect per-run resource measurements and model observations."""

    def __init__(self, config, device):
        self.config = config
        self.directory = config.get('run', {}).get('artifact_dir')
        logging = config.get('logging', {})
        self.phase_snapshots = bool(logging.get('phase_snapshots', False))
        self.save_phase_models = bool(logging.get('save_phase_models', False))
        self.profile_cuda = bool(logging.get('profile_cuda', False)) and str(device).startswith('cuda')
        self.device = device
        self.events = []
        self.phases = []
        self.snapshots = []
        self._cuda_phase_stack = []

    def _accumulate_cuda_peak(self, entry):
        entry['peak_cuda_allocated_bytes'] = max(entry.get('peak_cuda_allocated_bytes', 0),
                                                torch.cuda.max_memory_allocated(self.device))
        entry['peak_cuda_reserved_bytes'] = max(entry.get('peak_cuda_reserved_bytes', 0),
                                               torch.cuda.max_memory_reserved(self.device))

    @contextmanager
    def phase(self, task_id, name):
        if self.profile_cuda:
            torch.cuda.synchronize(self.device)
            if self._cuda_phase_stack:
                self._accumulate_cuda_peak(self._cuda_phase_stack[-1])
            torch.cuda.reset_peak_memory_stats(self.device)
            self._cuda_phase_stack.append({})
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.profile_cuda:
                torch.cuda.synchronize(self.device)
            entry = {'task_id': int(task_id), 'phase': name, 'wall_seconds': time.perf_counter() - start,
                     'cuda_synchronized': self.profile_cuda,
                     'parent_process_memory': process_memory_stats()}
            if self.profile_cuda:
                peak = self._cuda_phase_stack.pop()
                self._accumulate_cuda_peak(peak)
                entry.update(peak)
                if self._cuda_phase_stack:
                    parent = self._cuda_phase_stack[-1]
                    for key, value in peak.items():
                        parent[key] = max(parent.get(key, 0), value)
                    torch.cuda.reset_peak_memory_stats(self.device)
            self.phases.append(entry)

    def save_model(self, model, task_id, phase):
        if self.directory is None:
            return None
        path = Path(self.directory) / f'task_{task_id + 1}_{phase}_model.pt'
        torch.save({'model': copy.deepcopy(model).cpu(), 'task_id': task_id, 'phase': phase,
                    'config': copy.deepcopy(self.config),
                    'run_id': self.config.get('run', {}).get('id')}, path)
        return str(path)

    def flush(self, trainer):
        if self.directory is None:
            return
        summary = trainer.metrics.get_summary(max(trainer.metrics.task_times, default=0))
        write_json(Path(self.directory) / 'observations.json', {
            'run_id': trainer.config.get('run', {}).get('id'),
            'seed': trainer.config.get('run', {}).get('seed'),
            'benchmark': trainer.config['benchmark'].get('name'),
            'amca': trainer.amca_tester.compute_amca() if trainer.amca_tester else None,
            'end_task_mca': list(trainer.amca_tester.mean_class_accuracies) if trainer.amca_tester else [],
            'summary': summary, 'evaluation_records': trainer.metrics.evaluation_records,
            'accuracy_matrix': trainer.metrics.accuracy_matrix,
            'events': self.events, 'phases': self.phases, 'phase_snapshots': self.snapshots,
            'reallocation_history': trainer.reallocation_history,
        })
