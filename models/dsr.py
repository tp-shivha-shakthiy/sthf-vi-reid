import torch
import torch.nn as nn


class DSR(nn.Module):
    def __init__(self, feature_dim: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim

        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        original_seq_feat: torch.Tensor,
        high_freq_seq_feat: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([original_seq_feat, high_freq_seq_feat], dim=-1)
        fused = fused.permute(0, 2, 1)
        delta = self.temporal_fusion(fused)
        refined = delta.permute(0, 2, 1)
        refined = original_seq_feat + refined
        return refined
