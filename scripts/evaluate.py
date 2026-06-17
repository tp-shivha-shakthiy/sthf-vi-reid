"""Lightweight evaluation entry point.

Supports cross-modality evaluation (IR→RGB and RGB→IR).

Usage:
    python scripts/evaluate.py --features experiments/run/features/features.pt
"""

import argparse
import os
import sys

import torch
import yaml

from metrics.evaluator import Evaluator
from scripts.export_results_table import append_experiment


def _get_experiment_name(config_path, features_path):
    if config_path:
        base = os.path.splitext(os.path.basename(config_path))[0]
        name = base.replace("configs/", "").replace(".yaml", "")
        return name
    if features_path:
        parts = features_path.split(os.sep)
        if len(parts) >= 2:
            return parts[-3]
    return "unknown"


def _print_results(results):
    for direction in ("ir_to_rgb", "rgb_to_ir"):
        label = direction.replace("_", " → ").upper()
        print("=" * 50)
        print(f"  {label}")
        print("=" * 50)
        for key in ("rank1", "rank5", "rank10", "rank20", "mAP"):
            print(f"    {key:>8s}: {results[direction][key]:.2f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Cross-modality ReID evaluation")
    parser.add_argument("--features", required=True, help="Path to features.pt")
    parser.add_argument("--config", help="Config path (for experiment naming)")
    parser.add_argument("--output-dir", help="Override output directory for metrics.json")
    parser.add_argument("--metric", default="cosine", choices=("cosine", "euclidean"))
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    args = parser.parse_args()

    data = torch.load(args.features, map_location="cpu", weights_only=True)

    evaluator = Evaluator(metric=args.metric)
    results = evaluator.evaluate_all_directions(
        data["features"],
        data["pids"],
        data["camids"],
        data["modalities"],
        topk=(1, 5, 10, 20),
    )

    _print_results(results)

    experiment_name = _get_experiment_name(args.config, args.features)
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

    metrics_path = evaluator.save_metrics_json(
        results, experiment_name, output_dir=output_dir,
    )

    if not args.no_csv:
        append_experiment(experiment_name, results)


if __name__ == "__main__":
    main()
