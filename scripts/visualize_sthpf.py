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


if __name__ == "__main__":
    main()
