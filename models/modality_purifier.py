import torch
import torch.nn as nn


class ModalityPurifier(nn.Module):
    """Modality Purifier (MP) — paper formula:

        fp = mc * f + (1 - mc) * IN(f)

    where:
        mc = sigmoid(W2(ReLU(W1(GAP(f)))))

    Supports both:
      - 4D input: [N, C, H, W]   (single-frame feature maps)
      - 5D input: [B, T, C, H, W] (temporal feature maps, processed per-frame)

    Output shape always matches input shape.
    """

    def __init__(self, num_channels: int = 2048, reduction: int = 16):
        super().__init__()
        hidden = max(num_channels // reduction, 16)
        self.instance_norm = nn.InstanceNorm2d(num_channels, affine=True)
        # W1: C → hidden,  W2: hidden → C
        self.fc1 = nn.Linear(num_channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
            out = self._forward_4d(x_flat)
            return out.reshape(B, T, C, H, W)
        elif x.ndim == 4:
            return self._forward_4d(x)
        else:
            raise ValueError(
                f"Expected 4D or 5D input, got {x.ndim}D with shape {x.shape}"
            )

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        """Core MP computation on 4D [N, C, H, W] tensors.

        fp = mc * f + (1 - mc) * IN(f)
        mc = sigmoid(W2(ReLU(W1(GAP(f)))))
        """
        N, C, H, W = x.shape

        # IN(f)
        x_in = self.instance_norm(x)

        # GAP(f) → [N, C]
        gap = x.mean(dim=(2, 3))

        # mc = sigmoid(W2(ReLU(W1(GAP(f))))) → [N, C]
        mc = self.fc2(self.relu(self.fc1(gap)))
        mc = torch.sigmoid(mc)

        # Reshape mc for broadcasting: [N, C, 1, 1]
        mc = mc.view(N, C, 1, 1)

        # fp = mc * f + (1 - mc) * IN(f)
        fp = mc * x + (1.0 - mc) * x_in
        return fp
