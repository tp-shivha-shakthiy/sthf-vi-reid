import torch
import torch.nn as nn


class SDC(nn.Module):
    def __init__(self, feature_dim: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim
        self.align = nn.Identity()

    def forward(self, original_feat: torch.Tensor, high_freq_feat: torch.Tensor):
        aligned = high_freq_feat
        if original_feat.shape[1] != high_freq_feat.shape[1]:
            in_c = high_freq_feat.shape[1]
            out_c = original_feat.shape[1]
            ndim = high_freq_feat.ndim
            if ndim == 2:
                self.align = nn.Linear(in_c, out_c, bias=False).to(
                    high_freq_feat.device, high_freq_feat.dtype
                )
            elif ndim == 4:
                self.align = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False).to(
                    high_freq_feat.device, high_freq_feat.dtype
                )
            aligned = self.align(high_freq_feat)
        compensated_feat = original_feat + aligned
        return compensated_feat
