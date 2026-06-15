import os

import matplotlib.pyplot as plt
import torch

from models.adaptive_sthpf import AdaptiveSTHPF


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdaptiveSTHPF(in_channels=3, hidden_dim=32).to(device)
    model.eval()

    x = torch.randn(1, 6, 3, 288, 144, device=device)

    with torch.no_grad():
        weak_out = model.weak_branch(x)
        paper_out = model.paper_branch(x)
        strong_out = model.strong_branch(x)
        adaptive_out = model(x)
        weights = model.latest_gate_weights

    outs = {
        "Weak\n(fs=5, ft=1)": weak_out,
        "Paper\n(fs=10, ft=2)": paper_out,
        "Strong\n(fs=15, ft=2)": strong_out,
        "Adaptive\nw=[{:.3f}, {:.3f}, {:.3f}]".format(
            weights[0, 0].item(), weights[0, 1].item(), weights[0, 2].item()
        ): adaptive_out,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (title, out) in zip(axes.flat, outs.items()):
        frame = out[0, 0].cpu()
        frame = frame - frame.min()
        frame = frame / frame.max()
        frame = frame.permute(1, 2, 0).numpy()
        ax.imshow(frame)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/adaptive_sthpf_visualization.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.sthpf import FixedSTHPF


def _make_synthetic_batch(seq_len=6, height=288, width=144):
    """Create synthetic frames with spatial structure and temporal motion."""
    b, t, c, h, w = 1, seq_len, 3, height, width
    frames = torch.zeros(b, t, c, h, w)

    for ti in range(t):
        for ci in range(c):
            x = torch.linspace(-1, 1, w).view(1, -1).expand(h, -1)
            y = torch.linspace(-1, 1, h).view(-1, 1).expand(-1, w)
            gradient = (x + y) / 2.0
            frames[0, ti, ci, :, :] = gradient

    rect_size = 20
    for ti in range(t):
        offset = int(ti * 5)
        start_y = height // 2 - rect_size // 2 + offset
        start_x = width // 2 - rect_size // 2 + offset
        end_y = min(start_y + rect_size, height)
        end_x = min(start_x + rect_size, width)
        for ci in range(c):
            frames[0, ti, ci, start_y:end_y, start_x:end_x] = 0.9

    return frames


def _tensor_to_img(tensor):
    """Convert a normalized tensor [C, H, W] to displayable [H, W, C] uint8."""
    img = tensor.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype("uint8")
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize FixedST-HPF filter effect")
    parser.add_argument("--config", help="Config path (for img_size, currently unused)")
    parser.add_argument("--output", default="results/figures/fixed_sthpf_visualization.png",
                        help="Output figure path")
    parser.add_argument("--fs", type=int, default=10, help="Spatial cutoff")
    parser.add_argument("--ft", type=int, default=2, help="Temporal cutoff")
    args = parser.parse_args()

    sthpf = FixedSTHPF(fs=args.fs, ft=args.ft)

    batch = _make_synthetic_batch(seq_len=6, height=288, width=144)
    with torch.no_grad():
        high_freq = sthpf(batch)

    frame_idx = 0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    rgb_frame = batch[0, frame_idx]
    hf_rgb_frame = high_freq[0, frame_idx]

    ir_frame = batch[0, frame_idx].clone()
    ir_frame[1:, :, :] = ir_frame[:1, :, :] * 0.3
    with torch.no_grad():
        ir_batch = batch.clone()
        ir_batch[0, :, 1:, :, :] = ir_batch[0, :, :1, :, :] * 0.3
        hf_ir_batch = sthpf(ir_batch)
        hf_ir_frame = hf_ir_batch[0, frame_idx]

    rgb_img = _tensor_to_img(rgb_frame)
    hf_rgb_img = _tensor_to_img(hf_rgb_frame)
    ir_img = _tensor_to_img(ir_frame)
    hf_ir_img = _tensor_to_img(hf_ir_frame)

    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("Original RGB Frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(hf_rgb_img)
    axes[0, 1].set_title("High-Passed RGB Frame")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(ir_img)
    axes[1, 0].set_title("Original IR Frame")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hf_ir_img)
    axes[1, 1].set_title("High-Passed IR Frame")
    axes[1, 1].axis("off")

    fig.suptitle(f"FixedST-HPF Visualization (fs={args.fs}, ft={args.ft})",
                 fontsize=14)

    plt.tight_layout()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"FixedST-HPF visualization saved to {args.output}")
    print(f"  Parameters: fs={args.fs}, ft={args.ft}")


if __name__ == "__main__":
    main()
