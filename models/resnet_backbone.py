import torch
import torch.nn as nn
from torchvision import models


class ResNet50VideoBackbone(nn.Module):
    """
    ResNet-50 backbone that exposes individual layers for SDC/DSR insertion.

    Exposes:
        - stem:    conv1 + bn1 + relu + maxpool  → output: 64-ch feature maps
        - layer1:  3 bottleneck blocks             → output: 256-ch feature maps
        - layer2:  4 bottleneck blocks             → output: 512-ch feature maps
        - layer3:  6 bottleneck blocks             → output: 1024-ch feature maps
        - layer4:  3 bottleneck blocks             → output: 2048-ch feature maps
        - pool:    AdaptiveAvgPool2d(1)
        - bn:      BatchNorm2d(2048)

    Input:
        frames: [B, T, C, H, W]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.feature_dim = resnet.fc.in_features  # 2048

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.pool = resnet.avgpool
        self.final_bn = nn.BatchNorm2d(self.feature_dim)

    def forward_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Run full backbone and return per-frame feature vectors.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            sequence: [B, T, 2048]
        """
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames with shape [B, T, C, H, W], got {frames.shape}"
            )

        b, t, c, h, w = frames.shape

        x = frames.reshape(b * t, c, h, w)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.final_bn(x)
        x = x.flatten(1)
        x = x.reshape(b, t, -1)
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Full forward: return temporally-averaged feature vector.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            features: [B, 2048]
        """
        seq = self.forward_sequence(frames)
        return seq.mean(dim=1)
