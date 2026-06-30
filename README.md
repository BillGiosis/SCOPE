# Structured Compression and Parameter Expansion (SCOPE)

**Structured Compression and Parameter Expansion** (SCOPE) is a continual-learning model for object classification in Autonomous Driving scenarios.

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
scripts/             Dataset preparation scripts
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

Dataset Ninja provides the full BDD100K 100K image set with annotations and image-level attributes.

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

Each label JSON must contain image-level `attributes.weather` and `attributes.timeofday`, plus object annotations under `frames[0].objects` with `box2d` and `category` fields. If the Dataset Ninja export is in Supervisely format, export or convert it to this BDD100K-style layout before running the split script.

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
  --config config/cladc_config.yaml \
```

### BDD-C

```bash
python experiments/train_bddc.py \
  --config config/bddc_config.yaml \
```

Each run writes:

- `results_<timestamp>.txt`
- `final_model_<timestamp>.pt`
- `final_model.pt`

under the dataset-specific checkpoint directory.

## Main Configuration Fields

Both configs follow the same high-level structure.

### `benchmark`

Defines dataset paths, task counts, evaluation split, and parameter-budget ratio.

### `model`

```yaml
adapt_downsample_blocks: true | false
```

### `training`

Controls epochs, optimizer, learning rate, AMP, gradient accumulation, and class-balanced loss.

Useful ablation fields:

```yaml
use_class_balanced_loss: true | false
```

CLAD-C currently uses gradient accumulation by config. BDD-C disables gradient accumulation inside `experiments/train_bddc.py` so the configured batch size is used directly.

### `replay_buffer`

Controls rehearsal memory, optional stored soft targets, and optional distillation loss.

Useful ablation fields:

```yaml
use_soft_targets: true | false
use_distillation_loss: true | false
auto_distill_end_of_task: true | false
```

`use_soft_targets` controls whether soft targets are computed and stored when adding examples to the buffer. `use_distillation_loss` controls whether replay batches use the KL distillation term during training.

### `network_expansion`

Controls adapter expansion at the start of each task.


The current strategy expands adapters in layer2 and layer3 at task start.
There is also the option of expanding all layers at task start.

### `parameter_reallocation`

Controls end-of-task parameter reallocation.

Supported options are:

- `whole_model`
- `older_adapters_only`
- `all_adapters`
- `hybrid`

Whole-model reallocation is structural only.

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
