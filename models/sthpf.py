import torch
import torch.nn as nn


class FixedSTHPF(nn.Module):
    """
    Fixed Spatial-Temporal High-Pass Filter.

    Input:
        video: [B, T, C, H, W]

    Output:
        high_freq_video: [B, T, C, H, W]

    Applies FFT over temporal and spatial dimensions only:
        T, H, W

    Does not apply FFT over:
        B or C
    """

    def __init__(self, fs: int = 10, ft: int = 2):
        super().__init__()
        self.fs = fs
        self.ft = ft

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"Expected video shape [B, T, C, H, W], got {video.shape}"
            )

        if self.ft < 0 or self.fs < 0:
            raise ValueError("Cutoff frequencies fs and ft must be non-negative.")

        b, t, c, h, w = video.shape

        x = torch.fft.fftn(video, dim=(1, 3, 4))
        x = torch.fft.fftshift(x, dim=(1, 3, 4))

        mask = torch.ones((t, h, w), device=video.device, dtype=x.dtype)

        t_center = t // 2
        h_center = h // 2
        w_center = w // 2

        t_start = max(t_center - self.ft, 0)
        t_end = min(t_center + self.ft + 1, t)

        h_start = max(h_center - self.fs, 0)
        h_end = min(h_center + self.fs + 1, h)

        w_start = max(w_center - self.fs, 0)
        w_end = min(w_center + self.fs + 1, w)

        mask[t_start:t_end, h_start:h_end, w_start:w_end] = 0

        mask = mask.view(1, t, 1, h, w)

        x_filtered = x * mask

        x_filtered = torch.fft.ifftshift(x_filtered, dim=(1, 3, 4))
        high_freq = torch.fft.ifftn(x_filtered, dim=(1, 3, 4)).real

        return high_freq.contiguous()


# Backward-compatible alias.
STHPF = FixedSTHPF
