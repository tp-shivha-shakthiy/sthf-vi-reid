import torch
import torch.nn as nn

from models.resnet_backbone import ResNet50VideoBackbone


class STHFModel(nn.Module):
    """
    Skeleton for fixed/adaptive STHF model.

    Final model will contain:
        - original video branch
        - high-frequency branch
        - ST-HPF module
        - Modality Purifier
        - SDC
        - DSR

    Day 1 version:
        - Keeps the correct interface.
        - Uses two ResNet-50 backbones:
            original branch
            intermediate/high-frequency branch placeholder
        - Does not yet apply ST-HPF, SDC, or DSR.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feature_dim: int = 2048,
        sthpf_type: str = "fixed",
    ):
        super().__init__()

        self.sthpf_type = sthpf_type

        self.original_backbone = ResNet50VideoBackbone(pretrained=pretrained)
        self.intermediate_backbone = ResNet50VideoBackbone(pretrained=pretrained)

        actual_dim = self.original_backbone.feature_dim
        if actual_dim != feature_dim:
            raise ValueError(
                f"Configured feature_dim={feature_dim}, but backbone outputs {actual_dim}"
            )

        self.classifier = nn.Linear(feature_dim, num_classes)
        self.int_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, frames: torch.Tensor, modalities=None):
        """
        Args:
            frames: [B, T, C, H, W]
            modalities: optional list[str]

        Returns:
            model output dictionary
        """

        # Original branch placeholder.
        features = self.original_backbone(frames)
        logits = self.classifier(features)

        # Intermediate branch placeholder.
        # Later this will receive ST-HPF output, not raw frames.
        int_features = self.intermediate_backbone(frames)
        int_logits = self.int_classifier(int_features)

        return {
            "features": features,
            "logits": logits,
            "int_features": int_features,
            "int_logits": int_logits,
            "extra": {
                "model_type": f"sthf_{self.sthpf_type}",
                "sthpf_type": self.sthpf_type,
            },
        }
