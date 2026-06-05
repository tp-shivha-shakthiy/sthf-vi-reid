import torch
import torch.nn as nn

from models.resnet_backbone import ResNet50VideoBackbone


class BaselineModel(nn.Module):
    """
    Two-stream-compatible baseline skeleton.

    For now, this model uses a shared ResNet-50 video backbone.
    Later, if needed, modality-specific branches can be added.

    Input:
        frames: [B, T, C, H, W]

    Output:
        dictionary with:
            features
            logits
            int_features = None
            int_logits = None
            extra
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feature_dim: int = 2048,
    ):
        super().__init__()

        self.backbone = ResNet50VideoBackbone(pretrained=pretrained)

        actual_dim = self.backbone.feature_dim
        if actual_dim != feature_dim:
            raise ValueError(
                f"Configured feature_dim={feature_dim}, but backbone outputs {actual_dim}"
            )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, frames: torch.Tensor, modalities=None):
        features = self.backbone(frames)
        logits = self.classifier(features)

        return {
            "features": features,
            "logits": logits,
            "int_features": None,
            "int_logits": None,
            "extra": {
                "model_type": "baseline"
            },
        }
