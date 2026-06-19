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

> **Note:** Debug runs with randomly initialized weights (no checkpoint available).
> Perfect scores are expected for a single synthetic batch with random features.
> Real evaluation requires a trained checkpoint from `scripts/train.py`.<TK_EQUALS_MARKER>
