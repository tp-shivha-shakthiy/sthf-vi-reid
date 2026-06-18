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

    The forward_feature_maps() method runs stem+layer1 and returns 4D feature maps.
    The forward_full() method runs all layers and returns the final pooled feature.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.feature_dim = resnet.fc.in_features  # 2048

        # Expose individual layers for SDC/DSR insertion
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # 256 channels, 64x36 -> 64x36
        self.layer2 = resnet.layer2  # 512 channels, 64x36 -> 32x18
        self.layer3 = resnet.layer3  # 1024 channels, 32x18 -> 16x9
        self.layer4 = resnet.layer4  # 2048 channels, 16x9 -> 8x5 (approx)

<<<<<<< Updated upstream
    def forward_sequence(self, frames: torch.Tensor) -> torch.Tensor:
=======
        # Global average pooling + final batch norm (paper: GAP → BN → feature)
        self.pool = resnet.avgpool
        self.final_bn = nn.BatchNorm2d(self.feature_dim)

    def _forward_stem(self, x):
        """Run stem layers. Input: [N, 3, H, W]. Output: [N, 64, H/4, W/4]."""
        return self.stem(x)

    def forward_shallow(self, frames: torch.Tensor):
        """Run stem + layer1 on all frames.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            feat_maps: [B*T, 256, H', W'] — shallow feature maps for SDC.
        """
>>>>>>> Stashed changes
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames with shape [B, T, C, H, W], got {frames.shape}"
            )
<<<<<<< Updated upstream

        b, t, c, h, w = frames.shape

        x = frames.reshape(b * t, c, h, w)
        x = self.backbone(x)
        x = x.flatten(1)
        x = x.reshape(b, t, -1)
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
=======
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        x = self.stem(x)       # [B*T, 64, H/4, W/4]
        x = self.layer1(x)     # [B*T, 256, H/4, W/4]
        return x

    def forward_deep_from_shallow(self, shallow_feat: torch.Tensor):
        """Continue from shallow features through layer2 → layer4 → pool → BN.

        Args:
            shallow_feat: [B*T, 256, H', W'] — output of forward_shallow / SDC.

        Returns:
            features: [B, 2048] — final pooled feature vector.
        """
        x = self.layer2(shallow_feat)  # [N, 512, H'/2, W'/2]
        x = self.layer3(x)            # [N, 1024, H'/4, W'/4]
        x = self.layer4(x)            # [N, 2048, H'/8, W'/8]
        x = self.pool(x)              # [N, 2048, 1, 1]
        # Apply BN: need to reshape for BatchNorm2d which expects [N, C, 1, 1]
        x = self.final_bn(x)
        x = x.flatten(1)             # [N, 2048]
        return x

    def forward_deep_features(self, shallow_feat: torch.Tensor):
        """Run layer2 → layer3 → layer4 (before pooling) for DSR.

        Args:
            shallow_feat: [N, 256, H', W'] — output of layer1 / SDC.

        Returns:
            deep_feat: [N, 2048, H'', W''] — deep feature maps before pooling.
        """
        x = self.layer2(shallow_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_pool_bn(self, deep_feat: torch.Tensor):
        """Run GAP + BN on deep features.

        Args:
            deep_feat: [N, 2048, H'', W'']

        Returns:
            features: [N, 2048]
        """
        x = self.pool(deep_feat)
        x = self.final_bn(x)
        x = x.flatten(1)
        return x

    def forward_feature_maps(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract feature maps before global pooling.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            feature_maps: [B*T, C', H', W'] after conv layers, before avgpool.
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
        return x

<<<<<<< Updated upstream
=======
    def forward_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Run full backbone and return per-frame feature vectors.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            sequence: [B, T, 2048]
        """
>>>>>>> Stashed changes
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

>>>>>>> Stashed changes
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Full forward: return temporally-averaged feature vector.

        Args:
            frames: [B, T, C, H, W]

        Returns:
            features: [B, 2048]
        """
        seq = self.forward_sequence(frames)
        return seq.mean(dim=1)
