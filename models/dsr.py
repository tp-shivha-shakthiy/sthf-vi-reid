<<<<<<< Updated upstream
import torch
import torch.nn as nn


class DSR(nn.Module):
    def __init__(self, feature_dim: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim

        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        original_seq_feat: torch.Tensor,
        high_freq_seq_feat: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([original_seq_feat, high_freq_seq_feat], dim=-1)
        fused = fused.permute(0, 2, 1)
        delta = self.temporal_fusion(fused)
        refined = delta.permute(0, 2, 1)
        refined = original_seq_feat + refined
        return refined
=======
<<<<<<< Updated upstream
class DSR:
    def __init__(self, feature_dim=2048):
        self.feature_dim = feature_dim

    def forward(self, features):
        return features
=======
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSR(nn.Module):
    """Dynamic Spatial Resampling (DSR) — paper-faithful implementation.

    Operates on high-level 5D sequence feature maps [B, T, C, H, W] from deep
    ResNet blocks (layer3 output). Uses 3D 1x1 convolutions for Q, K, V
    projections and computes spatial-temporal affinity.

    Process:
        1. Apply Modality Purifier to original sequence feature maps → Fp
        2. Use 3D 1x1 convolutions to generate Q, K, V:
             phi_q(Fh) = Q_conv(Fh)     [B, P, T, H, W]
             phi_k(Fp) = K_conv(Fp)     [B, P, T, H, W]
             phi_v(Fp) = V_conv(Fp)     [B, C, T, H, W]
        3. Compute spatial-temporal affinity:
             Mst = softmax(phi_q(Fh) × phi_k(Fp)^T)   [B, T*H*W, T*H*W]
        4. Refine:
             Fst = Fp + wst · (phi_v(Fp) × Mst)       [B, C, T, H, W]

    Args:
        in_channels: channels in the input feature maps (e.g. 1024 from layer3).
        proj_channels: projection dimension for Q, K.  Defaults to in_channels // 8.
    """

    def __init__(self, in_channels: int = 1024, proj_channels: int = None):
        super().__init__()
        if proj_channels is None:
            proj_channels = max(in_channels // 8, 32)
        self.in_channels = in_channels
        self.proj_channels = proj_channels

        # Modality Purifier (inline to avoid circular imports)
        self.mp = ModalityPurifierInline3D(num_channels=in_channels)

        # 3D 1x1 convolutions for Q, K, V
        # kernel_size=(1,1,1) keeps T, H, W unchanged, mixes channels
        self.phi_q = nn.Conv3d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.phi_k = nn.Conv3d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.phi_v = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

        self.wst = nn.Parameter(torch.zeros(1))  # scalar gate, init to 0

    def forward(
        self, F_orig: torch.Tensor, F_hf: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            F_orig: original high-level sequence feature maps [B, C, T, H, W]
                    (note: channels-first for Conv3d)
            F_hf:   high-frequency high-level sequence feature maps [B, C, T, H, W]

        Returns:
            Fst: refined sequence feature maps [B, C, T, H, W]
        """
        B, C, T, H, W = F_orig.shape

        # 1. Modality Purifier on original
        Fp = self.mp(F_orig)

        # 2. 3D 1x1 convolutions for Q, K, V
        Q_hf = self.phi_q(F_hf)     # [B, P, T, H, W]
        K_orig = self.phi_k(Fp)     # [B, P, T, H, W]
        V_orig = self.phi_v(Fp)     # [B, C, T, H, W]

        # 3. Spatial-temporal affinity
        # Reshape to [B, P, T*H*W] for Q and K
        P = self.proj_channels
        THW = T * H * W

        q_flat = Q_hf.reshape(B, P, THW)       # [B, P, THW]
        k_flat = K_orig.reshape(B, P, THW)      # [B, P, THW]

        # Mst = softmax(q^T × k)  →  [B, THW, THW]
        Mst = torch.bmm(q_flat.transpose(1, 2), k_flat)  # [B, THW, P] × [B, P, THW] → [B, THW, THW]
        Mst = F.softmax(Mst, dim=-1)

        # 4. Refine: Fst = Fp + wst · (V × Mst)
        v_flat = V_orig.reshape(B, C, THW)     # [B, C, THW]
        refined = torch.bmm(v_flat, Mst).reshape(B, C, T, H, W)  # [B, C, T, H, W]

        Fst = Fp + self.wst * refined

        return Fst


class ModalityPurifierInline3D(nn.Module):
    """Minimal MP for DSR — operates on 5D [B, C, T, H, W] (channels-first).

    fp = mc * F + (1 - mc) * IN(F)
    mc = sigmoid(W2(ReLU(W1(GAP(F)))))
    """

    def __init__(self, num_channels: int = 1024, reduction: int = 16):
        super().__init__()
        hidden = max(num_channels // reduction, 16)
        self.instance_norm = nn.InstanceNorm3d(num_channels, affine=True)
        self.fc1 = nn.Linear(num_channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W] (channels-first for Conv3d/InstanceNorm3d)."""
        N, C = x.shape[:2]
        x_in = self.instance_norm(x)
        # GAP over T, H, W → [N, C]
        gap = x.mean(dim=(2, 3, 4))
        mc = torch.sigmoid(self.fc2(self.relu(self.fc1(gap))))  # [N, C]
        mc = mc.view(N, C, 1, 1, 1)
        return mc * x + (1.0 - mc) * x_in
>>>>>>> Stashed changes
>>>>>>> Stashed changes
