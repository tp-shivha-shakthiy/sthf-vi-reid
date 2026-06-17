import torch
import torch.nn as nn

from models.dsr import DSR
from models.resnet_backbone import ResNet50VideoBackbone
from models.sdc import SDC
from models.sthpf import FixedSTHPF


class STHFModel(nn.Module):
    """
    Fixed/adaptive STHF model.

    Contains:
        - original video branch
        - high-frequency branch with ST-HPF
        - SDC for feature compensation
        - DSR for temporal structural refinement
        - classifiers for both branches

    The ST-HPF module is applied to frames before the intermediate backbone.
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
        self.dsr = DSR(feature_dim=actual_dim)

    def forward(self, frames: torch.Tensor, modalities=None):
        original_seq = self.original_backbone.forward_sequence(frames)
        features = original_seq.mean(dim=1)

        if self.sthpf is not None:
            high_freq_frames = self.sthpf(frames)
            hf_seq = self.intermediate_backbone.forward_sequence(high_freq_frames)
        else:
            hf_seq = self.intermediate_backbone.forward_sequence(frames)

        int_features = hf_seq.mean(dim=1)

        compensated_feat = self.sdc(features, int_features)

        refined_seq = self.dsr(original_seq, hf_seq)
        refined = refined_seq.mean(dim=1)

        logits = self.classifier(refined)
        int_logits = self.int_classifier(refined)

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
