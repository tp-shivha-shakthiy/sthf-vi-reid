# ST-HF-VVI-ReID

Faithful  reproduction and extension of Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.

Status: In progress

Current focus:
- Baseline ResNet-50 video ReID model
- Fixed spatial-temporal high-frequency filtering module
- HITSZ-style training/evaluation pipeline
- Reproducible experiments and ablations

Tech stack:
Python, PyTorch, TorchVision, NumPy, YAML, pytest

Repository structure:
- models/ — backbone, baseline, STHF model modules
- losses/ — ID loss, triplet loss, combined ReID loss
- engine/ — training loop
- scripts/ — sanity checks, training, evaluation entry points
- configs/ — baseline/fixed/adaptive experiment configs
- tests/ — forward-pass and loss contract tests

Current validation:
- Model sanity checks pass for baseline/fixed/adaptive configs
- Forward contract tests pass
- Loss tests pass
- Debug training step runs successfully
- Full training pipeline enabled (see Colab setup below)

## Colab Dataset Setup

The HITSZ-VCM dataset is not stored in GitHub. To train on real data in Colab:

1. Download from Kaggle and extract to `/content/data/hitsz_vcm`
2. Create symlinks inside the repo:

```python
%cd /content/ST-HF-VVI-ReID
!rm -rf data/hitsz_vcm
!mkdir -p data/hitsz_vcm
!ln -s /content/data/hitsz_vcm/Train/Train data/hitsz_vcm/train
!ln -s /content/data/hitsz_vcm/Test/Test data/hitsz_vcm/test
!ln -s /content/data/hitsz_vcm/info data/hitsz_vcm/info
!ls -l data/hitsz_vcm
```

3. Verify: `python scripts/sanity_check_dataset.py --config configs/baseline_hitsz.yaml`
4. Train: `python scripts/train.py --config configs/baseline_hitsz.yaml` (add `--epochs N` for smoke runs)

## Notes

- Runs below 200 epochs are preliminary/smoke, not full reproduction.
- Config `train.save_dir` controls checkpoint output path.
- Use `--epochs` CLI flag to override config epochs for quick tests.