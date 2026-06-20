# Experiment Log

## baseline_hitsz (debug — no checkpoint)

| Direction        | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP    |
|------------------|--------|--------|---------|---------|--------|
| IR → RGB         | 100.0% | 100.0% | 100.0%  | 100.0%  | 100.0% |
| RGB → IR         | 100.0% | 100.0% | 100.0%  | 100.0%  | 100.0% |

## sthf_fixed_hitsz (debug — no checkpoint)

| Direction        | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP    |
|------------------|--------|--------|---------|---------|--------|
| IR → RGB         | 100.0% | 100.0% | 100.0%  | 100.0%  | 100.0% |
| RGB → IR         | 100.0% | 100.0% | 100.0%  | 100.0%  | 100.0% |

## sthf_fixed_hitsz (evaluated with checkpoint)

- **Config:** `configs/sthf_fixed_hitsz.yaml`
- **Checkpoint:** `experiments/hitsz_sthf_fixed/last.pth`
- **Date:** 2026-06-19

| Direction        | Rank-1  | Rank-5  | Rank-10 | mAP     |
|------------------|---------|---------|---------|---------|
| IR → RGB         | 100.00% | 100.00% | 100.00% | 100.00% |
| RGB → IR         | 100.00% | 100.00% | 100.00% | 100.00% |

> **Note:** Evaluation ran on synthetic debug batch (no real dataset mounted).
> 100% scores are an artifact of the tiny synthetic test set.
> Real evaluation requires mounting the HITSZ-VCM dataset at `data/hitsz_vcm` and using `--real-data`.

## sthf_adaptive_hitsz (evaluated — no trained checkpoint)

- **Config:** `configs/sthf_adaptive_hitsz.yaml`
- **Checkpoint:** `experiments/hitsz_sthf_adaptive/last.pth`
- **Date:** 2026-06-19
- **Status:** Checkpoint not available (dataset not mounted for training). Evaluation ran with random weights on synthetic data.

| Direction        | Rank-1  | Rank-5  | Rank-10 | mAP     |
|------------------|---------|---------|---------|---------|
| IR → RGB         | 100.00% | 100.00% | 100.00% | 100.00% |
| RGB → IR         | 100.00% | 100.00% | 100.00% | 100.00% |

> **Note:** Placeholder results. Requires training with `python scripts/train.py --config configs/sthf_adaptive_hitsz.yaml --real-data` on a machine with the HITSZ-VCM dataset, followed by re-evaluation.
