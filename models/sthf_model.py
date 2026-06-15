import torch
import torch.nn as nn

from models.resnet_backbone import ResNet50VideoBackbone
from models.sdc import SDC
from models.sthpf import FixedSTHPF


class STHFModel(nn.Module):
    """
    Fixed/adaptive STHF model.

    Contains:
        - original video branch
        - high-frequency branch with ST-HPF
        - classifiers for both branches

    The ST-HPF module is applied to frames before the intermediate backbone.
    Modality Purifier, SDC, and DSR will be added in later phases.
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

        if sthpf_type == "fixed":
            self.sthpf = FixedSTHPF(fs=10, ft=2)
        else:
            self.sthpf = None

        self.original_backbone = ResNet50VideoBackbone(pretrained=pretrained)
        self.intermediate_backbone = ResNet50VideoBackbone(pretrained=pretrained)

        actual_dim = self.original_backbone.feature_dim
        if actual_dim != feature_dim:
            raise ValueError(
                f"Configured feature_dim={feature_dim}, but backbone outputs {actual_dim}"
            )

        self.classifier = nn.Linear(feature_dim, num_classes)
        self.int_classifier = nn.Linear(feature_dim, num_classes)
        self.sdc = SDC(feature_dim=actual_dim)

    def forward(self, frames: torch.Tensor, modalities=None):
        features = self.original_backbone(frames)

        if self.sthpf is not None:
            high_freq_frames = self.sthpf(frames)
            int_features = self.intermediate_backbone(high_freq_frames)
        else:
            int_features = self.intermediate_backbone(frames)

        compensated_feat = self.sdc(features, int_features)

        logits = self.classifier(compensated_feat)
        int_logits = self.int_classifier(compensated_feat)

        extra = {
            "model_type": f"sthf_{self.sthpf_type}",
            "sthpf_type": self.sthpf_type,
        }

        return {
            "features": features,
            "logits": logits,
            "int_features": int_features,
            "int_logits": int_logits,
            "extra": extra,
        }
