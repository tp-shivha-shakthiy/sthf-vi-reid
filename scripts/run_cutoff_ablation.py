"""Cutoff ablation pipeline for ST-HF.

Iterates over ablation variants, trains (or debug-forwards) and evaluates
each, then records metrics to results/tables/cutoff_ablation.csv.

Usage:
    # Structural check with synthetic data:
    python scripts/run_cutoff_ablation.py --config configs/cutoff_ablation_hitsz.yaml

    # Full training + evaluation (requires HITSZ-VCM dataset):
    python scripts/run_cutoff_ablation.py --config configs/cutoff_ablation_hitsz.yaml --real-data
"""

import argparse
import csv
import os
import sys
import yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.sthf_model import STHFModel
from data.hitsz_vcm import HITSZVCM
from data.transforms import get_video_transforms, build_caj
from data.collate import collate_video_fn
from data.video_sampler import VideoSampler
from losses.build_loss import build_loss
from engine.trainer import Trainer
from metrics.evaluator import Evaluator


ABLATION_CSV = "results/tables/cutoff_ablation.csv"
ABLATION_COLUMNS = [
    "method", "cutoff_parameters",
    "ir_to_rgb_rank1", "ir_to_rgb_rank5", "ir_to_rgb_rank10", "ir_to_rgb_rank20", "ir_to_rgb_mAP",
    "rgb_to_ir_rank1", "rgb_to_ir_rank5", "rgb_to_ir_rank10", "rgb_to_ir_rank20", "rgb_to_ir_mAP",
]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _cutoff_label(variant):
    if variant["sthpf_type"] == "adaptive":
        return "learned_dynamic_weights"
    return f"fs={variant['fs']},ft={variant['ft']}"


def _variant_save_dir(base_dir, name):
    return os.path.join(base_dir, f"ablation_{name}")


def _extract_features_debug(model, batch_size=2, seq_len=6):
    """Extract synthetic features for debug structural check."""
    model.eval()
    half = batch_size // 2
    dummy = {
        "frames": torch.randn(batch_size, seq_len, 3, 64, 32),
        "pids": torch.arange(batch_size),
        "camids": torch.zeros(batch_size, dtype=torch.long),
        "modalities": ["rgb"] * half + ["ir"] * (batch_size - half),
        "track_ids": torch.zeros(batch_size, dtype=torch.long),
    }
    with torch.no_grad():
        outputs = model(dummy["frames"], modalities=dummy["modalities"])
    return {
        "features": outputs["features"].cpu(),
        "pids": dummy["pids"].cpu(),
        "camids": dummy["camids"].cpu(),
        "modalities": dummy["modalities"],
    }


def _extract_features_real(model, loader, device):
    """Extract features from real dataset loader."""
    model.eval()
    all_features, all_pids, all_camids, all_modalities = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            outputs = model(frames, modalities=batch.get("modalities"))
            all_features.append(outputs["features"].cpu())
            all_pids.append(batch["pids"].cpu())
            all_camids.append(batch["camids"].cpu())
            all_modalities.extend(batch["modalities"])
    return {
        "features": torch.cat(all_features, dim=0),
        "pids": torch.cat(all_pids, dim=0),
        "camids": torch.cat(all_camids, dim=0),
        "modalities": all_modalities,
    }


def _evaluate(data):
    evaluator = Evaluator(metric="cosine")
    results = evaluator.evaluate_all_directions(
        data["features"], data["pids"], data["camids"], data["modalities"],
        topk=(1, 5, 10, 20),
    )
    return results


def _flatten_results(variant_name, variant, results):
    cutoff = _cutoff_label(variant)
    row = {"method": variant_name, "cutoff_parameters": cutoff}
    for direction in ("ir_to_rgb", "rgb_to_ir"):
        for key, val in results.get(direction, {}).items():
            row[f"{direction}_{key}"] = round(val, 2) if isinstance(val, float) else val
    return row


def _upsert_csv(row):
    os.makedirs(os.path.dirname(ABLATION_CSV), exist_ok=True)
    rows = []
    if os.path.isfile(ABLATION_CSV):
        with open(ABLATION_CSV, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            updated = False
            for r in reader:
                if not updated and r.get("method") == row["method"]:
                    rows.append(row)
                    updated = True
                elif r.get("method") != row["method"]:
                    rows.append(r)
            if not updated:
                rows.append(row)
    else:
        fieldnames = ABLATION_COLUMNS
        rows.append(row)
    with open(ABLATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results written to {ABLATION_CSV}")


def main():
    parser = argparse.ArgumentParser(description="ST-HF cutoff ablation pipeline")
    parser.add_argument("--config", required=True, help="Ablation config path")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real HITSZ-VCM dataset (train + eval)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    sweep_cfg = config.get("sweep", {})
    variants = sweep_cfg.get("variants", [])
    seq_len = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))
    num_classes = config.get("model", {}).get("num_classes", 500)
    batch_size = data_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 3.5e-4)
    epochs = train_cfg.get("epochs", 150)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = data_cfg.get("root", "./data/hitsz_vcm")
    base_save_dir = train_cfg.get("save_dir", "experiments/hitsz_cutoff_ablation")

    print("=" * 60)
    print("ST-HF Cutoff Ablation Pipeline")
    print("=" * 60)
    print(f"Variants: {[v['name'] for v in variants]}")
    print(f"Real data: {args.real_data}")
    print()

    for variant in variants:
        name = variant["name"]
        sthpf_type = variant["sthpf_type"]
        save_dir = _variant_save_dir(base_save_dir, name)
        print(f"\n{'=' * 60}")
        print(f"  Variant: {name}  (type={sthpf_type})")
        print(f"{'=' * 60}")

        # Build model
        model = STHFModel(
            num_classes=num_classes,
            pretrained=True,
            feature_dim=config["model"]["feature_dim"],
            sthpf_type=sthpf_type,
        ).to(device)
        print(f"  Model built (pretrained=True, sthpf_type={sthpf_type})")

        if args.real_data:
            if not os.path.isdir(root):
                print(f"Dataset not found at: {os.path.abspath(root)}")
                sys.exit(1)

            # Training
            criterion = build_loss(config)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            caj = build_caj(config.get("augmentation", {}).get("caj"))

            transform = get_video_transforms(mode="train", img_size=img_size)
            dataset = HITSZVCM(root=root, seq_len=seq_len, transform=transform, split="train")
            sampling_cfg = config.get("sampling", {})
            sampler = VideoSampler(dataset, num_ids=sampling_cfg.get("num_ids", 4),
                                   clips_per_id=sampling_cfg.get("clips_per_id", 4), shuffle=True)
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_video_fn)

            trainer = Trainer(model, criterion, optimizer, scheduler, config, caj=caj)
            trainer.fit(loader, None, epochs, max_batches=100)

            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))
            print(f"  Checkpoint saved to {save_dir}/last.pth")

            # Evaluation features
            eval_transform = get_video_transforms(mode="test", img_size=img_size)
            eval_dataset = HITSZVCM(root=root, seq_len=seq_len, transform=eval_transform, split="test")
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_video_fn)
            data = _extract_features_real(model, eval_loader, device)
        else:
            # Debug: synthetic forward pass
            print("  Debug mode: synthetic feature extraction (no training)")
            with torch.no_grad():
                data = _extract_features_debug(model, batch_size=2, seq_len=seq_len)

        # Evaluate
        results = _evaluate(data)
        for direction in ("ir_to_rgb", "rgb_to_ir"):
            d = results[direction]
            print(f"  {direction:12s}  rank1={d['rank1']:.2f}%  rank5={d['rank5']:.2f}%  "
                  f"rank10={d['rank10']:.2f}%  mAP={d['mAP']:.2f}%")

        # Save per-variant metrics.json
        evaluator = Evaluator()
        evaluator.save_metrics_json(results, f"ablation_{name}", output_dir=save_dir)

        # Upsert to ablation CSV
        row = _flatten_results(name, variant, results)
        _upsert_csv(row)

    # Final summary
    print(f"\n{'=' * 60}")
    print("Ablation complete. Final table:")
    print(f"{'=' * 60}")
    if os.path.isfile(ABLATION_CSV):
        with open(ABLATION_CSV) as f:
            print(f.read())


if __name__ == "__main__":
    main()
