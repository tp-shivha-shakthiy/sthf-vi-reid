# Paper Settings

This document records all hyperparameter choices used to reproduce the
ST-HF paper on the HITSZ-VCM dataset.

---

## Dataset Settings

| Setting           | Value       | Explanation                                    |
|-------------------|-------------|------------------------------------------------|
| Sequence Length   | 6           | Number of frames sampled per video clip.       |
| Image Resolution  | 288 x 144   | Height and width of each frame after resize.   |

A sequence length of 6 provides enough temporal context for clip-level
feature extraction while keeping GPU memory manageable. The 288 x 144
resolution preserves spatial detail without excessive compute cost.

---

## Sampling Settings

| Setting          | Value | Explanation                                     |
|------------------|-------|-------------------------------------------------|
| Batch Size       | 16    | Total number of clips per training step.        |
| IDs per Batch    | 4     | Number of unique identities in each batch.      |
| Clips per ID     | 4     | Number of clips sampled per identity.           |

Identity-balanced sampling (4 IDs x 4 clips = 16) ensures every batch
contains multiple views of the same person, which is crucial for
learning discriminative video-level features.

---

## Optimization

| Setting    | Value    | Explanation                               |
|------------|----------|-------------------------------------------|
| Optimizer  | Adam     | Adaptive momentum-based gradient descent. |
| Learning Rate | 0.00035 | Initial learning rate for Adam.       |
| Scheduler  | Cosine   | Cosine annealing without restarts.        |

Adam with a relatively low learning rate (3.5e-4) provides stable
training. The cosine schedule smoothly decays the LR to zero over the
training duration, removing the need for manual milestone tuning.

---

## ST-HF Fixed Cutoffs

| Parameter           | Value | Explanation                                    |
|---------------------|-------|------------------------------------------------|
| Spatial Cutoff (fs) | 10    | Number of spatial frequency bins retained.     |
| Temporal Cutoff (ft)| 2    | Number of temporal frequency bins retained.    |

The spatial cutoff fs = 10 preserves low-to-mid frequency spatial
patterns while filtering high-frequency noise. The temporal cutoff
ft = 2 retains only the lowest temporal frequencies, emphasizing
slow-changing motion patterns across frames.
