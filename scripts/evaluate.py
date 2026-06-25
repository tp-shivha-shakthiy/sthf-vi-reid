"""
Evaluation entry point for Fixed STHF checkpoint.

Supports:
1) Pre-extracted features (--features)
2) Real model inference (--config + --checkpoint)
"""

import argparse
import os
import sys
import torch
import yaml

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metrics.evaluator import Evaluator
from scripts.export_results_table import append_experiment
from scripts.extract_features import build_model as build_feature_model


# -----------------------------
# FIXED checkpoint loader
# -----------------------------
def load_checkpoint(checkpoint_path):
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print("[WARNING] No checkpoint found — using random weights")
        return None

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Case 1: wrapped checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]

    # Case 2: raw state_dict
    if isinstance(ckpt, dict):
        return ckpt

    raise ValueError("Invalid checkpoint format")


# -----------------------------
# REAL feature extraction (FIXED)
# -----------------------------
def extract_features(config, checkpoint_path):
    from data.hitsz_vcm import HITSZVCM
    from data.transforms import get_video_transforms
    from torch.utils.data import DataLoader
    from data.collate import collate_video_fn

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    root = data_cfg.get("root", "./data/hitsz_vcm")
    seq_len = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))
    batch_size = data_cfg.get("batch_size", 16)
    num_classes = model_cfg.get("num_classes", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_feature_model(config, num_classes=num_classes)

    state_dict = load_checkpoint(checkpoint_path)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

    model.to(device)
    model.eval()

    transform = get_video_transforms(mode="test", img_size=img_size)

    dataset = HITSZVCM(
        root=root,
        seq_len=seq_len,
        transform=transform,
        split="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_video_fn
    )

    all_features, all_pids, all_camids, all_modalities = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)

            outputs = model(frames, modalities=batch.get("modalities"))
            feats = outputs["features"].cpu()

            all_features.append(feats)
            all_pids.append(batch["pids"])
            all_camids.append(batch["camids"])
            all_modalities.extend(batch["modalities"])

    return {
        "features": torch.cat(all_features, dim=0),
        "pids": torch.cat(all_pids, dim=0),
        "camids": torch.cat(all_camids, dim=0),
        "modalities": all_modalities,
    }


# -----------------------------
# Utilities
# -----------------------------
def get_experiment_name(config_path, checkpoint_path):
    if config_path:
        return os.path.splitext(os.path.basename(config_path))[0]
    if checkpoint_path:
        return os.path.basename(os.path.dirname(checkpoint_path))
    return "unknown"


def print_results(results):
    for direction in ("ir_to_rgb", "rgb_to_ir"):
        print("=" * 50)
        print(f"{direction.upper().replace('_', ' ')}")
        print("=" * 50)
        for k in ("rank1", "rank5", "rank10", "mAP"):
            print(f"{k:>8s}: {results[direction][k]:.2f}%")
    print("=" * 50)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", help="Pre-extracted features")
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--checkpoint", help="Model checkpoint")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    if args.features:
        data = torch.load(args.features, map_location="cpu")
    else:
        if args.config is None:
            raise ValueError("Provide --features or --config")

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        data = extract_features(config, args.checkpoint)

    # -----------------------------
    # Evaluate
    # -----------------------------
    evaluator = Evaluator(metric=args.metric)

    results = evaluator.evaluate_all_directions(
        data["features"],
        data["pids"],
        data["camids"],
        data["modalities"],
        topk=(1, 5, 10, 20),
    )

    print("\nFixed STHF Evaluation\n")
    print_results(results)

    # -----------------------------
    # Save results
    # -----------------------------
    experiment_name = get_experiment_name(args.config, args.checkpoint)

    evaluator.save_metrics_json(
        results,
        experiment_name,
        output_dir=args.output_dir
    )

    if not args.no_csv:
        append_experiment(experiment_name, results)


if __name__ == "__main__":
    main()