import torch
import torch.nn as nn
from torchvision import models


class ResNet50VideoBackbone(nn.Module):
    """
    ResNet-50 wrapper for video sequence input.

    Input:
        frames: [B, T, C, H, W]

    Output:
        features: [B, feature_dim]

    Current behavior:
        - Flatten B and T.
        - Extract frame-level features using ResNet-50.
        - Average features over time.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.feature_dim = resnet.fc.in_features

        # Remove final classification layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames with shape [B, T, C, H, W], got {frames.shape}"
            )

        b, t, c, h, w = frames.shape

        x = frames.reshape(b * t, c, h, w)
        x = self.backbone(x)
        x = x.flatten(1)
        x = x.reshape(b, t, -1)
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames with shape [B, T, C, H, W], got {frames.shape}"
            )

        b, t, c, h, w = frames.shape

        # [B, T, C, H, W] -> [B*T, C, H, W]
        x = frames.reshape(b * t, c, h, w)

        # [B*T, 2048, 1, 1]
        x = self.backbone(x)

        # [B*T, 2048]
        x = x.flatten(1)

        # [B, T, 2048]
        x = x.reshape(b, t, -1)

        # Temporal average pooling: [B, 2048]
        x = x.mean(dim=1)

        return x
