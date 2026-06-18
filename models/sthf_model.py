import torch
import torch.nn as nn

from models.dsr import DSR
from models.sdc import SDC
from models.resnet_backbone import ResNet50VideoBackbone
from models.sthpf import FixedSTHPF
from models.adaptive_sthpf import AdaptiveSTHPF


class STHFModel(nn.Module):
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
        elif sthpf_type == "adaptive":
            self.sthpf = AdaptiveSTHPF(in_channels=3, hidden_dim=32)
        else:
            self.sthpf = None

        self.backbone = ResNet50VideoBackbone(pretrained=pretrained)
        actual_dim = self.backbone.feature_dim
        if actual_dim != feature_dim:
            raise ValueError(
                f"Configured feature_dim={feature_dim}, but backbone outputs {actual_dim}"
            )

        self.sdc = SDC(in_channels=256)
        self.dsr = DSR(feature_dim=actual_dim)

        self.classifier = nn.Linear(feature_dim, num_classes)
        self.int_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, frames: torch.Tensor, modalities=None):
        b, t, c, h, w = frames.shape
        backbone = self.backbone

        if self.sthpf is not None:
            high_freq_frames = self.sthpf(frames)
        else:
            high_freq_frames = frames

        # --- High-frequency branch (full pass) ---
        hf_x = high_freq_frames.reshape(b * t, c, h, w)
        hf_shallow = backbone.stem(hf_x)
        hf_after_layer1 = backbone.layer1(hf_shallow)
        hf_after_layer2 = backbone.layer2(hf_after_layer1)
        hf_after_layer3 = backbone.layer3(hf_after_layer2)
        hf_after_layer4 = backbone.layer4(hf_after_layer3)
        hf_pooled = backbone.pool(hf_after_layer4)
        hf_pooled = backbone.final_bn(hf_pooled)
        hf_seq = hf_pooled.flatten(1).reshape(b, t, -1)
        int_features = hf_seq.mean(dim=1)

        # --- Original branch with SDC compensation ---
        orig_x = frames.reshape(b * t, c, h, w)
        orig_shallow = backbone.stem(orig_x)
        orig_after_layer1 = backbone.layer1(orig_shallow)
        orig_comp1 = self.sdc(orig_after_layer1, hf_after_layer1)
        orig_after_layer2 = backbone.layer2(orig_comp1)
        orig_after_layer3 = backbone.layer3(orig_after_layer2)
        orig_after_layer4 = backbone.layer4(orig_after_layer3)
        orig_pooled = backbone.pool(orig_after_layer4)
        orig_pooled = backbone.final_bn(orig_pooled)
        orig_seq = orig_pooled.flatten(1).reshape(b, t, -1)

        # --- DSR temporal refinement ---
        refined_seq = self.dsr(orig_seq, hf_seq)
        features = refined_seq.mean(dim=1)

        logits = self.classifier(features)
        int_logits = self.int_classifier(int_features)

        extra = {
            "model_type": f"sthf_{self.sthpf_type}",
            "sthpf_type": self.sthpf_type,
        }

        if (
            self.sthpf_type == "adaptive"
            and isinstance(self.sthpf, AdaptiveSTHPF)
            and self.sthpf.latest_gate_weights is not None
        ):
            extra["filter_weights"] = self.sthpf.latest_gate_weights

        return {
            "features": features,
            "logits": logits,
            "int_features": int_features,
            "int_logits": int_logits,
            "extra": extra,
        }
