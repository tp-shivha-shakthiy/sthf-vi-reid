import torch
import torch.nn as nn

<<<<<<< Updated upstream
from models.dsr import DSR
=======
<<<<<<< Updated upstream
>>>>>>> Stashed changes
from models.resnet_backbone import ResNet50VideoBackbone
=======
from models.adaptive_sthpf import AdaptiveSTHPF
from models.dsr import DSR
>>>>>>> Stashed changes
from models.sdc import SDC
from models.resnet_backbone import ResNet50VideoBackbone
from models.sthpf import FixedSTHPF


class STHFModel(nn.Module):
    """Paper-exact STHF model following Fig. 3 of the paper.

<<<<<<< Updated upstream
    Contains:
        - original video branch
        - high-frequency branch with ST-HPF
        - SDC for feature compensation
        - DSR for temporal structural refinement
        - classifiers for both branches

    The ST-HPF module is applied to frames before the intermediate backbone.
<<<<<<< Updated upstream
=======
    Modality Purifier, SDC, and DSR will be added in later phases.
=======
    Architecture:
        - ST-HPF creates a high-frequency intermediate modality from input frames.
        - Two ResNet-50 branches process:
            1. Original sequence
            2. High-frequency sequence
        - SDC is inserted after layer1 AND layer2 (shallow blocks).
          It operates on 4D feature maps [B*T, C, H, W].
        - DSR is inserted after layer3 AND layer4 (deep blocks).
          It operates on 5D sequence feature maps [B, C, T, H', W'].
        - Final feature comes after DSR2 → GAP → BN.
        - Main classifier/loss works on original branch final feature.
        - Intermediate classifier/loss works on high-frequency branch final feature.

    Loss:
        Ltotal = LID + LTri + λ1 * LID_int + λ2 * LTri_int

    No pooled-vector shortcut.  No features = compensated + refined.
>>>>>>> Stashed changes
>>>>>>> Stashed changes
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

        # --- ST-HPF (high-frequency extractor) ---
        if sthpf_type == "fixed":
            self.sthpf = FixedSTHPF(fs=10, ft=2)
        elif sthpf_type == "adaptive":
            self.sthpf = AdaptiveSTHPF(in_channels=3, hidden_dim=32)
        else:
            self.sthpf = None

        # --- Single ResNet-50 backbone (shared architecture, two forward paths) ---
        self.backbone = ResNet50VideoBackbone(pretrained=pretrained)
        actual_dim = self.backbone.feature_dim
        if actual_dim != feature_dim:
            raise ValueError(
                f"Configured feature_dim={feature_dim}, but backbone outputs {actual_dim}"
            )

        # --- SDC: shallow blocks 1-2 ---
        self.sdc1 = SDC(in_channels=256)   # after layer1 (256 channels)
        self.sdc2 = SDC(in_channels=512)   # after layer2 (512 channels)

        # --- DSR: deep blocks 3-4 ---
        self.dsr1 = DSR(in_channels=1024)  # after layer3 (1024 channels)
        self.dsr2 = DSR(in_channels=2048)  # after layer4 (2048 channels)

        # --- Classifiers ---
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.int_classifier = nn.Linear(feature_dim, num_classes)
<<<<<<< Updated upstream
        self.sdc = SDC(feature_dim=actual_dim)
        self.dsr = DSR(feature_dim=actual_dim)

    def forward(self, frames: torch.Tensor, modalities=None):
<<<<<<< Updated upstream
        original_seq = self.original_backbone.forward_sequence(frames)
        features = original_seq.mean(dim=1)
=======
        features = self.original_backbone(frames)
=======
>>>>>>> Stashed changes

    def forward(self, frames: torch.Tensor, modalities=None):
        """
        Args:
            frames: [B, T, C, H, W] input video frames
            modalities: optional modality labels (unused in forward)
>>>>>>> Stashed changes

        Returns:
            dict with keys:
                features:    [B, 2048] — original branch final feature
                logits:      [B, num_classes] — main classifier output
                int_features: [B, 2048] — high-frequency branch final feature
                int_logits:  [B, num_classes] — intermediate classifier output
                extra:       dict with model metadata
        """
        b, t, c, h, w = frames.shape

        backbone = self.backbone

        # ============================================
        # High-Frequency Branch (runs first to get fh for SDC/DSR)
        # ============================================
        if self.sthpf is not None:
<<<<<<< Updated upstream
            high_freq_frames = self.sthpf(frames)
            hf_seq = self.intermediate_backbone.forward_sequence(high_freq_frames)
        else:
<<<<<<< Updated upstream
            hf_seq = self.intermediate_backbone.forward_sequence(frames)

        int_features = hf_seq.mean(dim=1)
=======
            int_features = self.intermediate_backbone(frames)
=======
            high_freq_frames = self.sthpf(frames)  # [B, T, 3, H, W]
        else:
            high_freq_frames = frames
>>>>>>> Stashed changes

        hf_x = high_freq_frames.reshape(b * t, c, h, w)  # [B*T, 3, H, W]
        hf_shallow = backbone.stem(hf_x)                  # [B*T, 64, H/4, W/4]
        hf_shallow = backbone.layer1(hf_shallow)          # [B*T, 256, H/4, W/4]
        hf_after_layer2 = backbone.layer2(hf_shallow)     # [B*T, 512, ...]
        hf_after_layer3 = backbone.layer3(hf_after_layer2) # [B*T, 1024, ...]
        hf_after_layer4 = backbone.layer4(hf_after_layer3) # [B*T, 2048, ...]
        hf_pooled = backbone.pool(hf_after_layer4)         # [B*T, 2048, 1, 1]
        hf_pooled = backbone.final_bn(hf_pooled)           # [B*T, 2048, 1, 1]
        int_features = hf_pooled.flatten(1).reshape(b, t, -1).mean(dim=1)  # [B, 2048]
>>>>>>> Stashed changes

<<<<<<< Updated upstream
        refined_seq = self.dsr(original_seq, hf_seq)
        refined = refined_seq.mean(dim=1)

        logits = self.classifier(refined)
        int_logits = self.int_classifier(refined)
=======
        # ============================================
        # Original Branch with SDC1, SDC2, DSR1, DSR2
        # ============================================
        orig_x = frames.reshape(b * t, c, h, w)  # [B*T, 3, H, W]
        orig_shallow = backbone.stem(orig_x)       # [B*T, 64, H/4, W/4]

<<<<<<< Updated upstream
        logits = self.classifier(compensated_feat)
        int_logits = self.int_classifier(compensated_feat)
=======
        # --- layer1 → SDC1 (shallow block 1) ---
        orig_after_layer1 = backbone.layer1(orig_shallow)   # [B*T, 256, H/4, W/4]
        orig_comp1 = self.sdc1(orig_after_layer1, hf_shallow)  # [B*T, 256, H/4, W/4]

        # --- layer2 → SDC2 (shallow block 2) ---
        orig_after_layer2 = backbone.layer2(orig_comp1)     # [B*T, 512, ...]
        orig_comp2 = self.sdc2(orig_after_layer2, hf_after_layer2)  # [B*T, 512, ...]

        # --- layer3 → DSR1 (deep block 3) ---
        orig_after_layer3 = backbone.layer3(orig_comp2)     # [B*T, 1024, ...]
        deep_c3 = orig_after_layer3.shape[1]
        deep_h3 = orig_after_layer3.shape[2]
        deep_w3 = orig_after_layer3.shape[3]
        orig_5d_3 = orig_after_layer3.reshape(b, t, deep_c3, deep_h3, deep_w3).permute(0, 2, 1, 3, 4)
        hf_5d_3 = hf_after_layer3.reshape(b, t, deep_c3, deep_h3, deep_w3).permute(0, 2, 1, 3, 4)
        orig_refined_5d_3 = self.dsr1(orig_5d_3, hf_5d_3)  # [B, 1024, T, H', W']
        orig_refined_4d_3 = orig_refined_5d_3.permute(0, 2, 1, 3, 4).reshape(
            b * t, deep_c3, deep_h3, deep_w3
        )

        # --- layer4 → DSR2 (deep block 4) ---
        orig_after_layer4 = backbone.layer4(orig_refined_4d_3)  # [B*T, 2048, ...]
        deep_c4 = orig_after_layer4.shape[1]
        deep_h4 = orig_after_layer4.shape[2]
        deep_w4 = orig_after_layer4.shape[3]
        orig_5d_4 = orig_after_layer4.reshape(b, t, deep_c4, deep_h4, deep_w4).permute(0, 2, 1, 3, 4)
        hf_5d_4 = hf_after_layer4.reshape(b, t, deep_c4, deep_h4, deep_w4).permute(0, 2, 1, 3, 4)
        orig_refined_5d_4 = self.dsr2(orig_5d_4, hf_5d_4)  # [B, 2048, T, H', W']

        # --- Final: GAP → BN → feature ---
        orig_refined_4d_4 = orig_refined_5d_4.permute(0, 2, 1, 3, 4).reshape(
            b * t, deep_c4, deep_h4, deep_w4
        )
        orig_pooled = backbone.pool(orig_refined_4d_4)    # [B*T, 2048, 1, 1]
        orig_pooled = backbone.final_bn(orig_pooled)       # [B*T, 2048, 1, 1]
        features = orig_pooled.flatten(1).reshape(b, t, -1).mean(dim=1)  # [B, 2048]

        # ============================================
        # Classifiers and outputs
        # ============================================
        logits = self.classifier(features)
        int_logits = self.int_classifier(int_features)
>>>>>>> Stashed changes
>>>>>>> Stashed changes

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
