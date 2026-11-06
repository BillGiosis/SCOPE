"""Export stratified success and failure examples with predicted-class Grad-CAM.

This is an offline classification diagnostic. Candidates are spread over metadata
domains/classes/tasks; success/failure selection precedes inspecting activation maps.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bddc import create_bddc_benchmark
from benchmarks.cladc import create_cladc_benchmark
from utils.constants import CLADC_CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from utils.experiment import load_config
from utils.observations import write_json


def grad_cam(model, image, target_layer, class_id):
    """Compute a normalized spatial activation map for one output class."""
    captured = []
    handle = target_layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        with torch.enable_grad():
            logits = model(image.detach().requires_grad_(True))
            features = captured[0]
            gradient = torch.autograd.grad(logits[0, class_id], features)[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (features * weights).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            cam = cam.detach()
            return ((cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)).cpu().numpy()
    finally:
        handle.remove()


def export_overlay(image, cam, path):
    pixels = image.detach().cpu().float().permute(1, 2, 0).numpy()
    pixels = np.clip(pixels * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN), 0, 1)
    heat = np.stack([cam, np.zeros_like(cam), 1 - cam], axis=-1)
    original = (pixels * 255).astype(np.uint8)
    overlay = (np.clip(.65 * pixels + .35 * heat, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([original, overlay], axis=1)).save(path)


def stratified_candidates(loaders, group, limit):
    """Interleave metadata strata with deterministic sample positions."""
    strata = defaultdict(list)
    for loader_id, loader in enumerate(loaders):
        dataset = loader.dataset
        for index, label in enumerate(dataset.targets):
            if hasattr(dataset, 'get_group_value'):
                domain = dataset.get_group_value(index, group)
            elif group == 'location' and hasattr(dataset, 'get_location'):
                domain = dataset.get_location(index)
            else:
                raise ValueError(f'Dataset does not expose the requested group: {group}')
            strata[(str(domain or 'unknown'), int(label), loader_id)].append(index)
    if not strata:
        return
    quota = max(1, int(np.ceil(limit / len(strata))))
    # Interleave domains rather than exhausting alphabetically early groups.
    # Rotate each class's first task to cover day/night tasks within the small
    # qualitative quota, while retaining deterministic evenly spaced indices.
    by_domain = defaultdict(lambda: defaultdict(list))
    for key in sorted(strata):
        by_domain[key[0]][key[1]].append(key)
    domain_keys = {}
    for domain, labels in sorted(by_domain.items()):
        order = []
        for turn in range(max(len(keys) for keys in labels.values())):
            for label_position, label in enumerate(sorted(labels)):
                keys = labels[label]
                if turn < len(keys):
                    order.append(keys[(label_position + turn) % len(keys)])
        domain_keys[domain] = iter(order)
    ordered_keys = []
    while domain_keys:
        for domain in list(domain_keys):
            try:
                ordered_keys.append(next(domain_keys[domain]))
            except StopIteration:
                del domain_keys[domain]
    queues = {}
    for key in ordered_keys:
        indices = strata[key]
        positions = np.linspace(0, len(indices) - 1, min(quota, len(indices)), dtype=int)
        queues[key] = iter([indices[position] for position in positions])
    yielded = 0
    while queues and yielded < limit:
        for key in list(queues):
            try:
                index = next(queues[key])
            except StopIteration:
                del queues[key]
                continue
            domain, label, loader_id = key
            yield loader_id, index, domain, label
            yielded += 1
            if yielded >= limit:
                break


def fixed_candidates(loaders, selection, group):
    """Reuse image identities without reselecting on another model's predictions."""
    if selection.get('group') != group:
        raise ValueError('Selection and requested metadata group do not match')
    examples = selection.get('examples', [])
    if not examples:
        raise ValueError('The supplied selection contains no examples')
    candidates, seen = [], set()
    for example in examples:
        loader_id, index = int(example['loader_id']), int(example['dataset_index'])
        if not 0 <= loader_id < len(loaders):
            raise ValueError('Selected loader is absent from the current benchmark')
        dataset = loaders[loader_id].dataset
        if not 0 <= index < len(dataset):
            raise ValueError('Selected image index is absent from the current dataset')
        if (loader_id, index) in seen:
            raise ValueError('Selection contains a duplicate image')
        seen.add((loader_id, index))
        label = int(dataset.targets[index])
        if hasattr(dataset, 'get_group_value'):
            domain = dataset.get_group_value(index, group)
        elif group == 'location' and hasattr(dataset, 'get_location'):
            domain = dataset.get_location(index)
        else:
            raise ValueError(f'Dataset does not expose the requested group: {group}')
        domain = str(domain or 'unknown')
        task = getattr(dataset, 'task_name', 'common_validation')
        if label != int(example['true_class']) or domain != example['domain'] or task != example['task_name']:
            raise ValueError('Selected image metadata differs from the original selection')
        candidates.append((loader_id, index, domain, label))
    return candidates


def main(args):
    if args.max_samples < 1 or args.max_examples < 1 or args.per_stratum < 1:
        raise ValueError('Selection limits must be positive')
    config = load_config(args.config)
    config['data_loading']['num_workers'] = 0
    name = config['benchmark']['name'].lower()
    benchmark = create_bddc_benchmark(config) if name == 'bdd-c' else create_cladc_benchmark(config)
    saved = torch.load(args.model_artifact, map_location='cpu', weights_only=False)
    saved_benchmark = saved.get('config', {}).get('benchmark', {})
    if saved_benchmark.get('name', config['benchmark']['name']) != config['benchmark']['name']:
        raise ValueError('Model artifact and selected benchmark config do not match')
    class_names = list(getattr(benchmark, 'class_names', CLADC_CLASS_NAMES))
    if saved_benchmark.get('class_names', class_names) != class_names:
        raise ValueError('Model artifact and dataset class mappings do not match')
    model = saved['model'].to(args.device).eval()
    layer = dict(model.named_modules())[args.layer]
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    selected, selected_inputs, raw_crops, examined = [], [], [], 0
    loaders = benchmark.get_all_test_dataloaders()
    selection_path = getattr(args, 'selection_json', None)
    supplied = json.loads(Path(selection_path).read_text()) if selection_path else None
    original_examples = {}
    if supplied is not None:
        if supplied.get('benchmark', config['benchmark']['name']) != config['benchmark']['name']:
            raise ValueError('Selection and benchmark do not match')
        if supplied.get('class_names', class_names) != class_names:
            raise ValueError('Selection and benchmark class mappings do not match')
        candidates = fixed_candidates(loaders, supplied, args.group)
        original_examples = {(int(row['loader_id']), int(row['dataset_index'])): row
                             for row in supplied['examples']}
    else:
        candidates = stratified_candidates(loaders, args.group, args.max_samples)
    for loader_id, index, domain, label in candidates:
        if supplied is None and len(selected) >= args.max_examples:
            break
        dataset = loaders[loader_id].dataset
        image, loaded_label = dataset[index]
        if int(loaded_label) != label:
            raise ValueError('Dataset label and metadata label disagree')
        image_hash = hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        expected_hash = original_examples.get((loader_id, index), {}).get('input_sha256')
        if expected_hash is not None and image_hash != expected_hash:
            raise ValueError('Selected image pixels/preprocessing differ from the original selection')
        input_image = image[None].to(args.device, dtype=torch.float32)
        with torch.inference_mode():
            probabilities = model(input_image).softmax(dim=1)[0]
            if probabilities.numel() != len(class_names):
                raise ValueError('Model output dimension and dataset class mapping do not match')
            prediction = int(probabilities.argmax())
            confidence = float(probabilities[prediction])
        examined += 1
        correct = prediction == label
        key = (domain, label, loader_id, correct)
        if supplied is None and counts[key] >= args.per_stratum:
            continue
        counts[key] += 1
        cam = grad_cam(model, input_image, layer, prediction)
        filename = f'example_{len(selected):04d}.png'
        export_overlay(image, cam, directory / filename)
        selected.append({'loader_id': loader_id, 'dataset_index': index, 'domain': domain,
                         'true_class': label, 'predicted_class': prediction, 'confidence': confidence,
                         'true_class_name': class_names[label], 'predicted_class_name': class_names[prediction],
                         'correct': correct, 'image': filename,
                         'input_sha256': image_hash,
                         'task_name': getattr(dataset, 'task_name', 'common_validation')})
        if getattr(args, 'export_input_bundle', None):
            selected_inputs.append(image.detach().cpu().clone())
            if hasattr(dataset, '_resolve_underlying_dataset_and_obj_id'):
                underlying, obj_id = dataset._resolve_underlying_dataset_and_obj_id(index)
                raw_image = underlying._load_image(obj_id)
            elif hasattr(dataset, 'samples'):
                with Image.open(dataset.samples[index][0]) as source_image:
                    raw_image = source_image.convert('RGB')
            else:
                raise ValueError('Cannot recover the raw object crop for shared diagnostics')
            raw_pixels = torch.from_numpy(np.asarray(raw_image.convert('RGB')).copy())
            raw_crops.append(raw_pixels)
            selected[-1]['raw_crop_sha256'] = hashlib.sha256(raw_pixels.numpy().tobytes()).hexdigest()
    write_json(directory / 'selection.json', {
        'source_artifact': str(Path(args.model_artifact).resolve()), 'group': args.group, 'layer': args.layer,
        'benchmark': config['benchmark']['name'], 'class_names': class_names,
        'source_run_id': saved.get('run_id'),
        'selection': (
            'fixed image list; predictions and Grad-CAM recomputed for this model'
            if supplied is not None else
            'domain-interleaved, task-rotated, evenly spaced class/task candidates; '
            'first successes/failures per domain/class/task stratum'
        ),
        'selection_source': str(Path(selection_path).resolve()) if selection_path else None,
        'cam_target': 'predicted class', 'examined_samples': examined,
        'max_samples': args.max_samples, 'max_examples': args.max_examples, 'per_stratum': args.per_stratum,
        'examples': selected,
    })
    if getattr(args, 'export_input_bundle', None):
        if not selected_inputs:
            raise ValueError('No images selected for the diagnostic input bundle')
        bundle_path = Path(args.export_input_bundle)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = bundle_path.with_suffix(bundle_path.suffix + '.tmp')
        torch.save({
            'images': torch.stack(selected_inputs), 'raw_crops': raw_crops, 'examples': selected,
            'class_names': class_names, 'benchmark': config['benchmark']['name'],
            'group': args.group, 'source_artifact': str(Path(args.model_artifact).resolve()),
            'source_run_id': saved.get('run_id'),
            'selection': 'SCOPE reference-selected shared qualitative inputs; not an accuracy sample',
        }, temporary)
        temporary.replace(bundle_path)
    print(f'Exported {len(selected)} examples from {examined} examined samples to {directory}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-artifact', required=True, help='Trusted *_model.pt artifact')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--group', required=True, choices=['weather', 'timeofday', 'location'])
    parser.add_argument('--layer', default='layer4')
    parser.add_argument('--per-stratum', type=int, default=2)
    parser.add_argument('--max-examples', type=int, default=100)
    parser.add_argument('--max-samples', type=int, default=1000)
    parser.add_argument(
        '--selection-json',
        help=('Reuse every image in a previous selection.json, in its original order; '
              'selection limits apply only to new selections')
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--export-input-bundle',
        help='Optional shared normalized image tensors for offline baseline comparisons'
    )
    main(parser.parse_args())
