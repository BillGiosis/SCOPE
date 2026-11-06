"""Measure inference latency, tensor storage and partial MACs of a saved model."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import compute_model_size
from utils.observations import process_memory_stats, write_json


def profile(model, shape, device, warmup=10, iterations=100, amp=False):
    """Time resident-input forward passes separately from loading and transfer."""
    if iterations < 1 or warmup < 0 or shape[0] < 1:
        raise ValueError('Invalid batch size, warmup or iterations')
    cuda = str(device).startswith('cuda')
    model.to(device).eval()
    if cuda:
        model.to(memory_format=torch.channels_last)
    # Use zeros to avoid RNG-dependent inputs; timing excludes input transfer.
    images = torch.zeros(shape, device=device)
    if cuda:
        images = images.contiguous(memory_format=torch.channels_last)
    macs = [0]
    handles = []

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            macs[0] += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            macs[0] += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=cuda and amp):
            output = model(images)
        if not torch.isfinite(output).all():
            raise RuntimeError('Model output is non-finite')
    finally:
        for handle in handles:
            handle.remove()
    samples = []
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=cuda and amp):
        for _ in range(warmup):
            model(images)
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iterations):
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(images)
            if cuda:
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - start)
    result = {'device': str(device), 'input_shape': list(shape), 'amp': bool(cuda and amp),
              'execution': 'eager, resident inputs, forward only', 'warmup': warmup, 'iterations': iterations,
              'mean_batch_ms': 1000 * float(np.mean(samples)),
              'median_batch_ms': 1000 * float(np.median(samples)),
              'p95_batch_ms': 1000 * float(np.quantile(samples, .95)),
              'images_per_second': shape[0] / float(np.mean(samples)),
              'conv_linear_macs_per_image': int(macs[0] / shape[0]), 'model_storage': compute_model_size(model),
              'mac_convention': 'one multiply-accumulate is one MAC; only Conv2d/Linear, excludes other operations',
              'torch_version': torch.__version__, 'cpu_threads': torch.get_num_threads(),
              'parent_process_memory': process_memory_stats(),
              'gpu': torch.cuda.get_device_name(torch.device(device)) if cuda else None}
    if cuda:
        result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-artifact', required=True,
        help='Trusted *_model.pt produced with save_inference_model/save_phase_models'
    )
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    saved = torch.load(args.model_artifact, map_location='cpu', weights_only=False)
    result = profile(saved['model'], (args.batch_size, 3, args.image_size, args.image_size), args.device,
                     args.warmup, args.iterations, args.amp)
    result['source_artifact'] = str(Path(args.model_artifact).resolve())
    result['source_run_id'] = saved.get('run_id')
    write_json(args.output, result)
    print(f'Mean batch latency: {result["mean_batch_ms"]:.3f} ms; images/s: {result["images_per_second"]:.2f}')
