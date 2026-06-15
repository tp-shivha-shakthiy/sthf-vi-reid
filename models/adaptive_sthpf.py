import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sthpf import FixedSTHPF


class AdaptiveSTHPF(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 32):
        super().__init__()

        self.weak_branch = FixedSTHPF(fs=5, ft=1)
        self.paper_branch = FixedSTHPF(fs=10, ft=2)
        self.strong_branch = FixedSTHPF(fs=15, ft=2)

        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        self.latest_gate_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weak_out = self.weak_branch(x)
        paper_out = self.paper_branch(x)
        strong_out = self.strong_branch(x)

        if x.ndim == 5:
            B, T, C = x.shape[:3]
            global_feat = x.mean(dim=(1, 3, 4))
        elif x.ndim == 4:
            B, C = x.shape[:2]
            global_feat = x.mean(dim=(2, 3))
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.ndim}D")

        logits = self.gate(global_feat)
        weights = F.softmax(logits, dim=-1)
        self.latest_gate_weights = weights.detach()

        if x.ndim == 5:
            w1 = weights[:, 0].view(B, 1, 1, 1, 1)
            w2 = weights[:, 1].view(B, 1, 1, 1, 1)
            w3 = weights[:, 2].view(B, 1, 1, 1, 1)
        else:
            w1 = weights[:, 0].view(B, 1, 1, 1)
            w2 = weights[:, 1].view(B, 1, 1, 1)
            w3 = weights[:, 2].view(B, 1, 1, 1)

        fused = w1 * weak_out + w2 * paper_out + w3 * strong_out
        return fused
