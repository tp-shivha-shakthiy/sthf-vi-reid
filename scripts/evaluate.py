"""Evaluation entry point for Fixed STHF checkpoint.

Usage:
    python scripts/evaluate.py --features experiments/run/features/features.pt

    python scripts/evaluate.py \\
        --config configs/sthf_fixed_hitsz.yaml \\
        --checkpoint experiments/hitsz_sthf_fixed/last.pth
"""

import argparse
import os
import sys

import torch
import yaml

from metrics.evaluator import Evaluator
from scripts.export_results_table import append_experiment
from scripts.extract_features import build_model as build_feature_model


def _extract_features_debug(config, checkpoint_path):
    """Extract features using a model with a debug (synthetic) batch."""
    from data.collate import collate_video_fn

    data_cfg = config.get("data", {})
    num_classes = config.get("model", {}).get("num_classes", 10)
    seq_len = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_feature_model(config, num_classes=num_classes)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {state.get('epoch', '?')})")
    else:
        print("No checkpoint found — using random weights (debug mode)")

    model.to(device)
    model.eval()

    batch_size = min(data_cfg.get("batch_size", 16), 2)
    half = batch_size // 2
    dummy_frames = torch.randn(batch_size, seq_len, 3, img_size[0], img_size[1])
    shared_pids = list(range(half))
    pids = torch.tensor(shared_pids + shared_pids[:batch_size - half])
    dummy = {
        "frames": dummy_frames,
        "pids": pids,
        "camids": torch.zeros(batch_size, dtype=torch.long),
        "modalities": ["rgb"] * half + ["ir"] * (batch_size - half),
        "track_ids": torch.zeros(batch_size, dtype=torch.long),
    }

    all_features = []
    all_pids = []
    all_camids = []
    all_modalities = []

    with torch.no_grad():
        frames = dummy["frames"].to(device)
        outputs = model(frames, modalities=dummy.get("modalities"))
        feats = outputs["features"].cpu()
        all_features.append(feats)
        all_pids.append(dummy["pids"].cpu())
        all_camids.append(dummy["camids"].cpu())
        all_modalities.extend(dummy["modalities"])

    return {
        "features": torch.cat(all_features, dim=0),
        "pids": torch.cat(all_pids, dim=0),
        "camids": torch.cat(all_camids, dim=0),
        "modalities": all_modalities,
    }


def _get_experiment_name(config_path, checkpoint_path):
    if config_path:
        base = os.path.splitext(os.path.basename(config_path))[0]
        return base.replace("configs/", "").replace(".yaml", "")
    if checkpoint_path:
        parts = checkpoint_path.split(os.sep)
        if len(parts) >= 2:
            return parts[-3]
    return "unknown"


def _print_results(results):
    for direction in ("ir_to_rgb", "rgb_to_ir"):
        label = direction.replace("_", " -> ").upper()
        print("=" * 50)
        print(f"  {label}")
        print("=" * 50)
        for key in ("rank1", "rank5", "rank10", "rank20", "mAP"):
            print(f"    {key:>8s}: {results[direction][key]:.2f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Cross-modality ReID evaluation")
    parser.add_argument("--features", help="Path to features.pt (pre-extracted)")
    parser.add_argument("--config", help="Config path (for extraction or naming)")
    parser.add_argument("--checkpoint", help="Checkpoint .pth path")
    parser.add_argument("--output-dir", help="Override output directory for metrics.json")
    parser.add_argument("--metric", default="cosine", choices=("cosine", "euclidean"))
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    args = parser.parse_args()

    if args.features:
        data = torch.load(args.features, map_location="cpu", weights_only=True)
    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        data = _extract_features_debug(config, args.checkpoint)
    else:
        print("Provide --features, or --config (+ optional --checkpoint).")
        sys.exit(1)

    evaluator = Evaluator(metric=args.metric)
    results = evaluator.evaluate_all_directions(
        data["features"],
        data["pids"],
        data["camids"],
        data["modalities"],
        topk=(1, 5, 10, 20),
    )

    _print_results(results)

    experiment_name = _get_experiment_name(args.config, args.checkpoint)
    output_dir = args.output_dir

    if output_dir is None and args.config:
        try:
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            save_dir = cfg.get("train", {}).get("save_dir", "")
            if save_dir:
                output_dir = save_dir
        except Exception:
            pass

    evaluator.save_metrics_json(
        results, experiment_name, output_dir=output_dir,
    )

    if not args.no_csv:
        append_experiment(experiment_name, results)


if __name__ == "__main__":
    main()
