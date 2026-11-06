# Structured Compression and Parameter Expansion (SCOPE)

**Structured Compression and Parameter Expansion** (SCOPE) is a continual-learning model for object classification in autonomous driving scenarios.

This repository contains training code for two object-classification benchmarks:

- **CLAD-C**: the continual object-classification stream from the official CLAD package.
- **BDD-C**: a continual benchmark derived from BDD100K object crops, split by weather and time-of-day tasks.

The implementation uses a ResNet50D backbone with lightweight task-start adapter expansion, replay-buffer rehearsal, optional class-balanced loss, optional distillation components, and structural parameter reallocation.

## Repository Layout

```text
benchmarks/          Dataset wrappers and dataloaders for CLAD-C and BDD-C
buffers/             Replay buffer with optional soft targets and distillation
config/              Experiment configs for CLAD-C and BDD-C
experiments/         Training entry points
models/              Expandable ResNet50D model with adapter modules
scripts/             Data preparation, profiling and Grad-CAM
trainers/            Continual-learning trainer and reallocation safeguards
utils/               Metrics, losses, reports, experiment helpers, constants
requirements.txt     Python dependencies
```

Generated checkpoints and result files are written under the configured `logging.save_dir`, currently:

```text
checkpoints/cladc/
checkpoints/bddc/
```

## Environment Setup
The code is intended for CUDA training. CPU execution is possible through `--device cpu`, but full training will be slow.

Important dependency notes:

- `timm` is used to create the pretrained `resnet50d` backbone. The first run may download pretrained weights unless they are already cached.
- `torch-pruning` is required for structural whole-model parameter reallocation.
- CLAD-C requires the official CLAD repository to be available locally and referenced in `config/cladc_config.yaml`.

Use a CUDA-enabled PyTorch build compatible with the training GPU, then install
the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependency file specifies minimum versions rather than a locked environment.
Each run records its installed package versions.

## Data Preparation

### CLAD-C

CLAD-C is loaded through the official CLAD package. The config must point to:

```yaml
benchmark:
  data_root: "/path/to/CLAD-C"
  clad_repo_root: "/path/to/CLAD"
```

### BDD-C

BDD-C uses cropped object images and JSON manifests generated from BDD100K-style detection annotations.

Recommended source:

- Dataset Ninja BDD100K: Images 100K: https://datasetninja.com/bdd100k

Check that the downloaded export includes the detection annotations and
image-level attributes required below.

The current BDD-C preprocessing script expects BDD100K-style detection JSON files, not cropped images that have already been grouped into classes. After downloading, arrange or convert the downloaded data into this raw input layout:

```text
<BASE_DATASET_DIR>/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each label JSON must contain image-level `attributes.weather` and `attributes.timeofday`, plus object annotations under `frames[0].objects` with `box2d` and `category` fields. If the export is in Supervisely, COCO, YOLO or classification-only format, convert it to this BDD100K-style layout before running the split script.

The preprocessing script is:

```bash
python scripts/bddc_split.py
```

Before running it, edit the constants at the top of [scripts/bddc_split.py](scripts/bddc_split.py) if your paths differ:

```python
BASE_DATASET_DIR = "/workspace/data/bdd"
OUTPUT_BASE_DIR = "/workspace/data/BDD-C"
```

The script expects the raw input layout above and creates:

```text
<OUTPUT_BASE_DIR>/
  train/
    task_1_clear_day/
    task_2_adverse_day/
    task_3_dawn_dusk/
    task_4_clear_night/
    task_5_adverse_night/
  val/
    task_1_clear_day/
    ...
  manifests/
    metadata.json
    train/<task_name>.json
    val/<task_name>.json
```

Then set:

```yaml
benchmark:
  data_root: "/path/to/BDD-C"
  eval_split: "val"
```

## Running Experiments

Run commands from the repository root.

### CLAD-C

```bash
python experiments/train_cladc.py \
  --config config/cladc_config.yaml --ablation reference --seed 42 --device cuda
```

### BDD-C

```bash
python experiments/train_bddc.py \
  --config config/bddc_config.yaml --ablation reference --seed 42 --device cuda
```

Each run writes:

- `results_<timestamp>.txt`
- `final_model_<timestamp>.pt`
- `final_model.pt`

under the dataset-specific checkpoint directory. Named ablations isolate outputs
at `checkpoints/<dataset>/ablations/<condition>/seed_<seed>/`. Omitting
`--ablation` uses the base configuration and its `logging.save_dir` directly.

Each execution uses a unique UTC time/random ID in report and timestamped checkpoint names.
Each also writes `run_<id>/manifest.json` (effective config, Torch seed, code/data
fingerprints and runtime) and `run_<id>/observations.json` (raw evaluation counts,
task matrix where available, pruning/capacity records and phase costs).

The supplied configs enable phase evaluations, phase-model export, synchronized
CUDA cost measurements and final inference-model export. These produce extra
evaluations and model artifacts. Their switches are `logging.phase_snapshots`,
`save_phase_models`, `profile_cuda` and `save_inference_model`.

Compaction reduces replay occupancy while retaining the allocated backing
tensors. Freezing historical adapters does not freeze predictions from a
changing shared backbone.

## Ablations

Set `--ablation` to one of the following controls. Unspecified settings are
inherited from the benchmark YAML.

| Condition | Change from the reference |
| --- | --- |
| `reference` | Keep the reference recipe; isolate outputs by seed. |
| `no_recovery` | Disable recovery training. |
| `no_compaction` | Disable end-of-task replay compaction. |
| `expansion_only` | Disable parameter reallocation; recovery does not run without reallocation. |
| `joint_adapters` | Train historical adapters during main training and recovery. |
| `no_kd` | Disable KD in training and Taylor scoring; retain replay CE, stored targets and compaction. |
| `fixed_architecture` | Disable expansion, reallocation and recovery. |
| `no_recovery_no_compaction` | Disable recovery and compaction together. |
| `tight_param_cap` | Set cap ratios to 1.025 for CLAD-C and 1.05 for BDD-C. |
| `small_replay` | Set capacities to 500 for CLAD-C and 2,500 for BDD-C. |

Use seeds 42–46 for the five-run protocol. For example:

```bash
for seed in 42 43 44 45 46; do
  python experiments/train_bddc.py --config config/bddc_config.yaml \
    --ablation no_kd --seed "$seed" --device cuda || break
done
```

The joint-adapter control keeps the reference structural-scoring trainability
policy. Structural pruning amounts are channel ratios, while whole-adapter
deletion uses a scalar-parameter target and can overshoot by one whole adapter.

## Result Files

The result report includes:

- experiment hyperparameters,
- replay-buffer settings,
- parameter reallocation settings,
- AMCA summary,
- per-task overall accuracy,
- per-class accuracy,
- grouped metrics where available,
- final model parameter count,
- total training time,
- reallocation history.

Final inference artifacts (`run_<id>/task_<N>_final_model.pt`) contain the actual
compressed architecture and weights. They are separate from training-state
checkpoints and do not provide a full training-resume workflow.

AMCA averages primary task-end mean class accuracies. Diagnostic snapshots do
not enter that history. BDD-C also records a task evaluation matrix for
forgetting and backward transfer. CLAD-C uses a common evaluation set, so its
checkpoint trajectory does not provide a task-specific forgetting matrix. Its
loader obtains official 64×64 crops before resizing normalized tensors to
224×224; class zero is unused.

## Profiling and Grad-CAM

The three scripts cover dataset preparation (`bddc_split.py`), inference
profiling (`profile_inference.py`) and qualitative inspection
(`qualitative_examples.py`). Training uses the entry points in `experiments/`.

Profile a saved final architecture and weights without training:

```bash
python scripts/profile_inference.py \
  --model-artifact /path/to/run_ID/task_5_final_model.pt \
  --device cuda --amp --batch-size 1 --image-size 224 \
  --warmup 50 --iterations 500 --output outputs/inference.json
```

The profiler measures resident-input forward execution, excluding loading,
preprocessing and transfer. Compare models on the same idle GPU with matching
settings. MAC counts cover Conv2d/Linear modules only, not total model FLOPs.
Model tensor storage and peak CUDA allocation are separate measurements.

Export metadata-stratified Grad-CAM examples:

```bash
python scripts/qualitative_examples.py --config config/bddc_config.yaml \
  --model-artifact /path/to/run_ID/task_5_final_model.pt \
  --device cuda --group weather --layer layer4 \
  --max-examples 24 --max-samples 1000 --overlay-alpha 0.6 \
  --save-cam-arrays --output-dir outputs/examples
```

For CLAD-C, use its config, final task 6 and `--group location`. Use
`--selection-json /path/to/selection.json` to reuse the same images for another
SCOPE configuration; labels, metadata and input hashes are checked. Grad-CAM
targets each model's predicted class. Selected examples illustrate behavior and
are not an accuracy sample.

Use `--cam-target ground_truth` with the same selection to compare the same
class across models. The default blue/cyan/yellow/red scale uses a heatmap blend
weight of 0.6; `--overlay-alpha` changes display only. `--save-cam-arrays` retains
the normalized maps so colours can be changed without recomputing gradients.

Recorded seeds and source fingerprints support provenance; they do not
guarantee bitwise reproducibility across hardware and package versions.
