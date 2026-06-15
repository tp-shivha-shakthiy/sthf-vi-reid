import torch
import torch.nn as nn

from models.sthpf import FixedSTHPF


class AdaptiveSTHPF(nn.Module):
    def __init__(self):
        super().__init__()

        self.weak_branch = FixedSTHPF(fs=5, ft=1)
        self.paper_branch = FixedSTHPF(fs=10, ft=2)
        self.strong_branch = FixedSTHPF(fs=15, ft=2)

        # Placeholder for future learnable gating (Day 8)
        self.gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weak_out = self.weak_branch(x)
        paper_out = self.paper_branch(x)
        strong_out = self.strong_branch(x)

        fused = (weak_out + paper_out + strong_out) / 3.0
        return fused
