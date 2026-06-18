import torch
import torch.nn as nn
import torch.nn.functional as F


class SDC(nn.Module):
    """Spatial-Deep Compensation (SDC) — paper-faithful implementation.

    Operates on low-level 4D feature maps [N, C, H, W] from shallow ResNet
    blocks (layer1). Injects spatially-resolved cross-modality information
    into the original branch feature maps so later layers can use it.

    Process:
        1. Apply Modality Purifier to original feature map → fp
        2. Channel information compensation:
             Mc = softmax(q1(fp) × k1(fh)^T)      [N, H*W, H*W]
             fc = fp + wc · (v1(fh) × Mc)           [N, C, H, W]
        3. Spatial information compensation:
             Ms = softmax(q2(fc) × k2(fh)^T)       [N, H*W, H*W]
             fcs = fc + ws · (v2(fh) × Ms)          [N, C, H, W]

    Args:
        in_channels: number of channels in the feature maps (e.g. 256 from layer1).
        proj_channels: projection dimension for Q, K, V.  Defaults to in_channels // 8.
    """

    def __init__(self, in_channels: int = 256, proj_channels: int = None):
        super().__init__()
        if proj_channels is None:
            proj_channels = max(in_channels // 8, 32)
        self.in_channels = in_channels
        self.proj_channels = proj_channels

        # Modality Purifier applied to original feature map
        self.mp = ModalityPurifierInline(num_channels=in_channels)

        # Channel compensation projections: q1, k1, v1  (1x1 convolutions)
        self.q1 = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.k1 = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.wc = nn.Parameter(torch.zeros(1))  # scalar gate, init to 0

        # Spatial compensation projections: q2, k2, v2
        self.q2 = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.k2 = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.v2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.ws = nn.Parameter(torch.zeros(1))  # scalar gate, init to 0

    def forward(
        self, f: torch.Tensor, fh: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f:  original low-level feature maps   [N, C, H, W]
            fh: high-frequency low-level feature maps [N, C, H, W]

        Returns:
            fcs: compensated feature maps [N, C, H, W]
        """
        N, C, H, W = f.shape

        # 1. Modality Purifier on original
        fp = self.mp(f)

        # --- Channel information compensation ---
        # q1(fp): [N, P, H, W], k1(fh): [N, P, H, W]
        q1_fp = self.q1(fp)  # [N, P, H, W]
        k1_fh = self.k1(fh)  # [N, P, H, W]
        v1_fh = self.v1(fh)  # [N, C, H, W]

        # Mc = softmax(q1(fp) × k1(fh)^T)  — channel attention map
        # Reshape to [N, P, H*W] for matrix multiply
        q1_flat = q1_fp.reshape(N, self.proj_channels, H * W)   # [N, P, HW]
        k1_flat = k1_fh.reshape(N, self.proj_channels, H * W)   # [N, P, HW]
        # Mc: [N, H*W, H*W] — but this is too large. Use attention over channels.
        # Paper: Mc = softmax(q1(fp) × k1(fh)^T) where the dot product
        # is over the channel dimension: [N, H*W, P] × [N, P, H*W] → [N, H*W, H*W]
        # This is standard self-attention. With P small, it's efficient.
        Mc = torch.bmm(q1_flat.transpose(1, 2), k1_flat)  # [N, HW, P] × [N, P, HW] → [N, HW, HW]
        Mc = F.softmax(Mc, dim=-1)  # [N, HW, HW]

        # v1(fh) × Mc:  [N, C, HW] × [N, HW, HW] → [N, C, HW]
        v1_flat = v1_fh.reshape(N, C, H * W)  # [N, C, HW]
        channel_comp = torch.bmm(v1_flat, Mc).reshape(N, C, H, W)  # [N, C, H, W]

        fc = fp + self.wc * channel_comp

        # --- Spatial information compensation ---
        q2_fc = self.q2(fc)   # [N, P, H, W]
        k2_fh = self.k2(fh)  # [N, P, H, W]
        v2_fh = self.v2(fh)  # [N, C, H, W]

        q2_flat = q2_fc.reshape(N, self.proj_channels, H * W)  # [N, P, HW]
        k2_flat = k2_fh.reshape(N, self.proj_channels, H * W)  # [N, P, HW]

        Ms = torch.bmm(q2_flat.transpose(1, 2), k2_flat)  # [N, HW, HW]
        Ms = F.softmax(Ms, dim=-1)

        v2_flat = v2_fh.reshape(N, C, H * W)  # [N, C, HW]
        spatial_comp = torch.bmm(v2_flat, Ms).reshape(N, C, H, W)  # [N, C, H, W]

        fcs = fc + self.ws * spatial_comp

        return fcs


class ModalityPurifierInline(nn.Module):
    """Minimal MP for SDC — inline version to avoid circular imports.

    fp = mc * f + (1 - mc) * IN(f)
    mc = sigmoid(W2(ReLU(W1(GAP(f)))))
    """

    def __init__(self, num_channels: int = 256, reduction: int = 16):
        super().__init__()
        hidden = max(num_channels // reduction, 16)
        self.instance_norm = nn.InstanceNorm2d(num_channels, affine=True)
        self.fc1 = nn.Linear(num_channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_in = self.instance_norm(x)
        gap = x.mean(dim=(2, 3))  # [N, C]
        mc = torch.sigmoid(self.fc2(self.relu(self.fc1(gap))))  # [N, C]
        mc = mc.view(N, C, 1, 1)
        return mc * x + (1.0 - mc) * x_in
