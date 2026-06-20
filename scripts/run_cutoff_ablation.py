"""Cutoff ablation pipeline for ST-HF.

Iterates over ablation variants defined in the sweep config,
runs a debug forward pass and loss computation for each variant
to verify the pipeline runs without crashing.

Usage:
    python scripts/run_cutoff_ablation.py --config configs/cutoff_ablation_hitsz.yaml
    python scripts/run_cutoff_ablation.py --config configs/cutoff_ablation_hitsz.yaml --debug
"""

import argparse
import os
import sys
import yaml

import torch
import torch.optim as optim

from models.sthf_model import STHFModel
from losses.build_loss import build_loss


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ST-HF cutoff ablation pipeline")
    parser.add_argument("--config", required=True, help="Ablation config path")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: synthetic batch, no training")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    sweep_cfg = config.get("sweep", {})
    variants = sweep_cfg.get("variants", [])

    seq_len = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))
    batch_size = 2
    num_classes = config.get("model", {}).get("num_classes", 500)
    lr = train_cfg.get("lr", 3.5e-4)

    print("=" * 60)
    print("ST-HF Cutoff Ablation Pipeline")
    print("=" * 60)
    print(f"Variants: {[v['name'] for v in variants]}")
    print()

    for variant in variants:
        name = variant["name"]
        sthpf_type = variant["sthpf_type"]
        print(f"\n--- Variant: {name} (type={sthpf_type}) ---")

        model = STHFModel(
            num_classes=num_classes,
            pretrained=False,
            feature_dim=config["model"]["feature_dim"],
            sthpf_type=sthpf_type,
        )
        model.eval()

        criterion = build_loss(config)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        dummy_batch = {
            "frames": torch.randn(batch_size, seq_len, 3, img_size[0], img_size[1]),
            "pids": torch.arange(batch_size),
            "camids": torch.zeros(batch_size, dtype=torch.long),
            "modalities": ["rgb", "ir"],
            "track_ids": torch.zeros(batch_size, dtype=torch.long),
        }

        with torch.no_grad():
            outputs = model(dummy_batch["frames"],
                            modalities=dummy_batch.get("modalities"))
            losses = criterion(outputs, dummy_batch["pids"])

        print(f"  Output keys:          {list(outputs.keys())}")
        print(f"  Features shape:       {outputs['features'].shape}")
        print(f"  Logits shape:         {outputs['logits'].shape}")
        print(f"  loss_total:           {losses['loss_total'].item():.4f}")
        print(f"  loss_id:              {losses['loss_id'].item():.4f}")
        print(f"  loss_tri:             {losses['loss_tri'].item():.4f}")

        if sthpf_type == "adaptive" and hasattr(model.sthpf, "latest_gate_weights"):
            fw = model.sthpf.latest_gate_weights
            if fw is not None:
                print(f"  gate_weights mean:    {fw.mean(dim=0).tolist()}")

        assert torch.isfinite(outputs["features"]).all(), "NaN in features"
        assert torch.isfinite(losses["loss_total"]), "NaN in loss_total"
        print(f"  Status:               OK")

    print("\n" + "=" * 60)
    print("All ablation variants passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
