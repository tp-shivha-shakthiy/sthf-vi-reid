import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention module."""

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        hidden = max(num_channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ModalityPurifier(nn.Module):
    """Suppresses modality-specific style noise via Instance Normalization
    and Channel Attention.

    Supports both:
      - 4D input: [B*T, C, H, W]  (batch–time flattened feature maps)
      - 5D input: [B, T, C, H, W] (temporal feature maps)

    Output shape always matches input shape.
    """

    def __init__(self, num_channels=2048):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_channels, affine=True)
        self.channel_attention = ChannelAttention(num_channels)

    def forward(self, x):
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.instance_norm(x)
            x = self.channel_attention(x)
            return x.view(B, T, C, H, W)
        elif x.ndim == 4:
            return self.channel_attention(self.instance_norm(x))
        else:
            raise ValueError(
                f"Expected 4D or 5D input, got {x.ndim}D with shape {x.shape}"
            )
