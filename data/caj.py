import torch
import torch.nn as nn


class ChannelAugmentedJoint(nn.Module):
    """Channel-Augmented Joint Learning (CAJ) augmentation.

    Paper: mixes channel-wise mean/std statistics between paired RGB and IR
    samples from the same identity within a mini-batch, creating augmented
    cross-modality views.

    Process (per pair of matched RGB/IR samples):
        1. Compute channel mean/std for each modality.
        2. Sample random mixing ratio λ ~ Beta(α, α) or Uniform.
        3. Create augmented RGB: scale+shift using mixed statistics.
        4. Create augmented IR:  scale+shift using mixed statistics.

    This module operates on a full batch and requires modality labels.
    If no same-identity RGB/IR pairs exist in the batch, it is a no-op.

    Args:
        alpha: Beta distribution parameter for mixing ratio.
            Higher values concentrate mixing near 0.5. Default: 1.0 (uniform).
        prob: probability of applying CAJ to each matched pair. Default: 0.5.
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob

    def forward(self, frames: torch.Tensor, pids: torch.Tensor,
                modalities: list) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W] — batch of video frames.
            pids: [B] — identity labels.
            modalities: list of B strings, each "rgb" or "ir".

        Returns:
            augmented frames: [B, T, C, H, W].
        """
        if not self.training:
            return frames

        B = frames.shape[0]
        device = frames.device
        dtype = frames.dtype

        # Find RGB and IR indices
        rgb_idx = [i for i, m in enumerate(modalities) if m == "rgb"]
        ir_idx = [i for i, m in enumerate(modalities) if m == "ir"]

        if not rgb_idx or not ir_idx:
            return frames

        # Match same-identity pairs
        rgb_pids = {i: pids[i].item() for i in rgb_idx}
        ir_pids = {i: pids[i].item() for i in ir_idx}

        # Group by pid
        pid_to_rgb = {}
        for i, pid in rgb_pids.items():
            pid_to_rgb.setdefault(pid, []).append(i)
        pid_to_ir = {}
        for i, pid in ir_pids.items():
            pid_to_ir.setdefault(pid, []).append(i)

        # Find common pids
        common_pids = set(pid_to_rgb.keys()) & set(pid_to_ir.keys())
        if not common_pids:
            return frames

        augmented = frames.clone()

        for pid in common_pids:
            if torch.rand(1).item() > self.prob:
                continue

            # Pick one random sample from each modality
            ri = pid_to_rgb[pid][torch.randint(len(pid_to_rgb[pid]), (1,)).item()]
            ii = pid_to_ir[pid][torch.randint(len(pid_to_ir[pid]), (1,)).item()]

            rgb_frames = frames[ri]  # [T, C, H, W]
            ir_frames = frames[ii]   # [T, C, H, W]

            # Sample mixing ratio
            if self.alpha > 0:
                lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
            else:
                lam = torch.rand(1).item()

            # Compute channel statistics: mean and std over T, H, W
            rgb_mean = rgb_frames.mean(dim=(0, 2, 3))  # [C]
            rgb_std = rgb_frames.std(dim=(0, 2, 3))    # [C]
            ir_mean = ir_frames.mean(dim=(0, 2, 3))    # [C]
            ir_std = ir_frames.std(dim=(0, 2, 3))      # [C]

            # Mixed statistics
            mixed_mean = lam * rgb_mean + (1 - lam) * ir_mean
            mixed_std = lam * rgb_std + (1 - lam) * ir_std

            # Augment RGB: normalize with RGB stats, apply mixed stats
            eps = 1e-5
            rgb_aug = (rgb_frames - rgb_mean.view(1, -1, 1, 1)) / (rgb_std.view(1, -1, 1, 1) + eps)
            rgb_aug = rgb_aug * mixed_std.view(1, -1, 1, 1) + mixed_mean.view(1, -1, 1, 1)

            # Augment IR: normalize with IR stats, apply mixed stats
            ir_aug = (ir_frames - ir_mean.view(1, -1, 1, 1)) / (ir_std.view(1, -1, 1, 1) + eps)
            ir_aug = ir_aug * mixed_std.view(1, -1, 1, 1) + mixed_mean.view(1, -1, 1, 1)

            augmented[ri] = rgb_aug.to(dtype)
            augmented[ii] = ir_aug.to(dtype)

        return augmented
