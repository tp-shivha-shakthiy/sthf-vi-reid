# ST-HF-VVI-ReID

Faithful reproduction and extension of **Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification**.

This repository implements a PyTorch-based reproduction pipeline for STHF on the HITSZ-VCM dataset, including a baseline video ReID model, fixed spatial-temporal high-frequency filtering, and an adaptive ST-HPF extension.

## Status

**Code implementation: mostly complete**
**Full experiment results: pending**

Current validation confirms that the core codebase is stable:

* Baseline, Fixed STHF, and Adaptive STHF model sanity checks pass.
* Forward contract tests pass.
* Loss tests pass.
* Dataset, sampler, metric, checkpoint, and evaluation protocol tests pass.
* Debug training runs successfully for all three model configs.
* Evaluation and ST-HPF visualization entry points are functional.
* Seed control and checkpoint resume support are available.

Full reproduction is still pending because real 200-epoch training and final evaluation have not yet been completed.

## Current Focus

* Running real HITSZ-VCM training for:

  * Baseline ResNet-50 video ReID
  * Fixed STHF
  * Adaptive STHF
* Generating:

  * Trained checkpoints
  * `metrics.json` files
  * Main comparison table
  * Cutoff ablation table
  * Final report assets

## Implemented Components

### Models

* ResNet-50 video backbone
* Baseline video ReID model
* Fixed ST-HPF module
* Adaptive ST-HPF module with weak, paper, and strong frequency branches
* Modality Purifier
* Spatial-Temporal Discriminative Component
* Disentangled Structural Refinement module

### Training

* ID loss
* Batch-hard triplet loss
* Intermediate ID and triplet losses for STHF
* Combined ReID loss builder
* Training loop with debug and real-data modes
* Adam optimizer
* Cosine scheduler
* Seed control
* Checkpoint saving
* Resume support

### Evaluation

* Feature extraction
* CMC / Rank-k evaluation
* mAP evaluation
* IR-to-RGB and RGB-to-IR protocol support
* Metrics JSON export
* Main result table export

### Visualization

* Fixed ST-HPF visualization
* Adaptive ST-HPF visualization

## Tech Stack

Python, PyTorch, TorchVision, NumPy, YAML, pytest

## Repository Structure

```text
configs/     Experiment configurations
data/        Dataset loader, collate function, sampler, transforms, CAJ augmentation
docs/        Reproduction notes and report assets
engine/      Training, checkpointing, logging, seed utilities
losses/      ID loss, triplet loss, combined ReID loss
metrics/     CMC, mAP, and evaluator logic
models/      Backbone, baseline, STHF, ST-HPF, adaptive ST-HPF, SDC, DSR
results/     Tables, figures, and final report assets
scripts/     Sanity checks, training, evaluation, visualization, feature extraction
tests/       Unit and integration tests
```

## Validation

Run the full test suite:

```bash
pytest tests/ -q
```

Expected current status:

```text
116 passed, 1 skipped
```

Run model sanity checks:

```bash
python scripts/sanity_check_model.py --config configs/baseline_hitsz.yaml
python scripts/sanity_check_model.py --config configs/sthf_fixed_hitsz.yaml
python scripts/sanity_check_model.py --config configs/sthf_adaptive_hitsz.yaml
```

Run debug training:

```bash
python scripts/train.py --config configs/baseline_hitsz.yaml --debug
python scripts/train.py --config configs/sthf_fixed_hitsz.yaml --debug
python scripts/train.py --config configs/sthf_adaptive_hitsz.yaml --debug
```

## Colab Dataset Setup

The HITSZ-VCM dataset is not stored in GitHub. To train on real data in Colab, download and extract the dataset to:

```text
/content/data/hitsz_vcm
```

Then create symlinks inside the repository:

```python
%cd /content/ST-HF-VVI-ReID

!rm -rf data/hitsz_vcm
!mkdir -p data/hitsz_vcm

!ln -s /content/data/hitsz_vcm/Train/Train data/hitsz_vcm/train
!ln -s /content/data/hitsz_vcm/Test/Test data/hitsz_vcm/test
!ln -s /content/data/hitsz_vcm/info data/hitsz_vcm/info

!ls -l data/hitsz_vcm
```

Verify the dataset:

```bash
python scripts/sanity_check_dataset.py --config configs/baseline_hitsz.yaml
```

## Smoke Training on Real Data

Before running full training, run short real-data smoke tests:

```bash
python scripts/train.py --config configs/baseline_hitsz.yaml --real-data --epochs 5 --max-batches 2
python scripts/train.py --config configs/sthf_fixed_hitsz.yaml --real-data --epochs 5 --max-batches 2
python scripts/train.py --config configs/sthf_adaptive_hitsz.yaml --real-data --epochs 5 --max-batches 2
```

Runs below 200 epochs are preliminary smoke tests and should not be reported as full reproduction results.

## Full Training

Train the baseline model:

```bash
python scripts/train.py --config configs/baseline_hitsz.yaml --real-data --epochs 200
```

Train the fixed STHF model:

```bash
python scripts/train.py --config configs/sthf_fixed_hitsz.yaml --real-data --epochs 200
```

Train the adaptive STHF model:

```bash
python scripts/train.py --config configs/sthf_adaptive_hitsz.yaml --real-data --epochs 200
```

Resume training from a checkpoint:

```bash
python scripts/train.py --config configs/sthf_fixed_hitsz.yaml --real-data \
  --resume experiments/hitsz_sthf_fixed/checkpoint_last.pth
```

## Evaluation

Evaluate trained checkpoints:

```bash
python scripts/evaluate.py --config configs/baseline_hitsz.yaml \
  --checkpoint experiments/hitsz_baseline/checkpoint_last.pth

python scripts/evaluate.py --config configs/sthf_fixed_hitsz.yaml \
  --checkpoint experiments/hitsz_sthf_fixed/checkpoint_last.pth

python scripts/evaluate.py --config configs/sthf_adaptive_hitsz.yaml \
  --checkpoint experiments/hitsz_sthf_adaptive/checkpoint_last.pth
```

Generate the main comparison table:

```bash
python scripts/export_results_table.py
```

## Visualization

Generate ST-HPF visualizations:

```bash
python scripts/visualize_sthpf.py --mode fixed
python scripts/visualize_sthpf.py --mode adaptive
```

Expected outputs:

```text
results/figures/fixed_sthpf_visualization.png
results/figures/adaptive_sthpf_visualization.png
```

## Notes

* The dataset is excluded from the repository.
* Full reproduction requires real HITSZ-VCM training.
* Runs below 200 epochs are smoke tests only.
* `train.save_dir` controls checkpoint output paths.
* `--epochs` can override config epochs for quick tests.
* `--seed` controls reproducibility.
* `--resume` can continue training from a checkpoint.
