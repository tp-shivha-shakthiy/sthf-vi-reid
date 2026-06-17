"""Lightweight evaluation entry point.

Supports two modes:

1. Protocol-aware mode (single features.pt):
    python scripts/evaluate.py \\
        --features experiments/run/features/features.pt \\
        --protocol ir_to_rgb \\
        --config configs/baseline_hitsz.yaml \\
        --checkpoint experiments/baseline/checkpoint_last.pth

2. Legacy mode (separate query/gallery files):
    python scripts/evaluate.py \\
        --query-features path/to/q_feat.pt \\
        --query-pids     path/to/q_pids.pt \\
        --query-camids   path/to/q_camids.pt \\
        --gallery-features path/to/g_feat.pt \\
        --gallery-pids     path/to/g_pids.pt \\
        --gallery-camids   path/to/g_camids.pt
"""

import argparse
import json
import os
import sys
import yaml

import torch

from metrics.evaluator import Evaluator


def _split_by_modality(data, protocol):
    modalities = data["modalities"]
    features = data["features"]
    pids = data["pids"]
    camids = data["camids"]

    if protocol == "ir_to_rgb":
        q_mask = [m == "ir" for m in modalities]
        g_mask = [m == "rgb" for m in modalities]
    elif protocol == "rgb_to_ir":
        q_mask = [m == "rgb" for m in modalities]
        g_mask = [m == "ir" for m in modalities]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    q_idx = [i for i, m in enumerate(q_mask) if m]
    g_idx = [i for i, m in enumerate(g_mask) if m]

    q_feat = features[q_idx]
    q_pids = pids[q_idx]
    q_camids = camids[q_idx]
    g_feat = features[g_idx]
    g_pids = pids[g_idx]
    g_camids = camids[g_idx]

    q_count = len(q_idx)
    g_count = len(g_idx)

    shared = set(q_pids.tolist()) & set(g_pids.tolist())
    shared_count = len(shared)

    return q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, q_count, g_count, shared_count


def _run_and_print(evaluator, q_feat, q_pids, q_camids, g_feat, g_pids, g_camids,
                   q_count, g_count, shared_count, protocol, metric, topk=(1, 5, 10, 20)):

    if shared_count == 0:
        raise ValueError(
            f"Zero shared identities between query ({q_count} samples)"
            f" and gallery ({g_count} samples). Cannot evaluate."
        )

    results = evaluator.evaluate(
        q_feat, q_pids, q_camids,
        g_feat, g_pids, g_camids,
        topk=topk,
    )

    print("=" * 50)
    print(f"Protocol: {protocol}")
    print(f"Metric:   {metric}")
    print(f"Query:    {q_count} samples")
    print(f"Gallery:  {g_count} samples")
    print(f"Shared:   {shared_count} identities")
    print("-" * 50)
    for key in ("rank1", "rank5", "rank10", "rank20", "mAP"):
        print(f"  {key:>8s}: {results[key]:.2f}%")
    print("=" * 50)

    return results


def _save_metrics_json(results, protocol, q_count, g_count, shared_count,
                       config_path, checkpoint_path, output_dir):
    if output_dir is None:
        return

    if protocol == "all_search":
        metrics = {
            "protocol": "all_search",
            "ir_to_rgb": results["ir_to_rgb"],
            "rgb_to_ir": results["rgb_to_ir"],
            "query_count": q_count,
            "gallery_count": g_count,
            "shared_identity_count": shared_count,
        }
    else:
        metrics = {**results,
                   "protocol": protocol,
                   "query_count": q_count,
                   "gallery_count": g_count,
                   "shared_identity_count": shared_count}

    if config_path:
        metrics["config"] = os.path.abspath(config_path)
    if checkpoint_path:
        metrics["checkpoint"] = os.path.abspath(checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="ReID evaluation")

    # Group 1: Protocol-aware evaluation (single features.pt)
    parser.add_argument("--features", help="Path to features.pt (single structured file)")
    parser.add_argument("--protocol", choices=["ir_to_rgb", "rgb_to_ir", "all_search"],
                        help="Evaluation protocol (requires --features)")
    parser.add_argument("--config", help="Config path (for metrics metadata)")

    # Group 2: Legacy evaluation (separate query/gallery files)
    parser.add_argument("--query-features", help="Query features .pt file")
    parser.add_argument("--query-pids", help="Query PIDs .pt file")
    parser.add_argument("--query-camids", help="Query camera IDs .pt file")
    parser.add_argument("--gallery-features", help="Gallery features .pt file")
    parser.add_argument("--gallery-pids", help="Gallery PIDs .pt file")
    parser.add_argument("--gallery-camids", help="Gallery camera IDs .pt file")

    parser.add_argument("--checkpoint", help="Checkpoint path (for metrics metadata)")
    parser.add_argument("--output-dir", help="Directory to save metrics.json")
    parser.add_argument("--metric", default="cosine", choices=("cosine", "euclidean"))
    args = parser.parse_args()

    evaluator = Evaluator(metric=args.metric)
    config_path = args.config
    checkpoint_path = args.checkpoint

    # Determine output directory: explicit --output-dir wins, else derive from config
    output_dir = args.output_dir
    if output_dir is None and config_path is not None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            save_dir = cfg.get("train", {}).get("save_dir", "")
            if save_dir:
                output_dir = save_dir
        except Exception:
            pass

    if args.features:
        data = torch.load(args.features, map_location="cpu", weights_only=True)
        protocol = args.protocol
        if protocol is None:
            print("--protocol is required when using --features")
            sys.exit(1)

        if protocol == "all_search":
            results = {}
            for sub_proto in ("ir_to_rgb", "rgb_to_ir"):
                q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, qc, gc, sc = \
                    _split_by_modality(data, sub_proto)
                sub_results = _run_and_print(
                    evaluator, q_feat, q_pids, q_camids, g_feat, g_pids, g_camids,
                    qc, gc, sc, sub_proto, args.metric,
                )
                results[sub_proto] = sub_results

            # Counts are from ir_to_rgb (representative)
            _, _, _, _, _, _, qc, gc, sc = _split_by_modality(data, "ir_to_rgb")
            _save_metrics_json(results, "all_search", qc, gc, sc,
                               config_path, checkpoint_path, output_dir)
        else:
            q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, qc, gc, sc = \
                _split_by_modality(data, protocol)
            results = _run_and_print(
                evaluator, q_feat, q_pids, q_camids, g_feat, g_pids, g_camids,
                qc, gc, sc, protocol, args.metric,
            )
            _save_metrics_json(results, protocol, qc, gc, sc,
                               config_path, checkpoint_path, output_dir)

    else:
        if not (args.query_features and args.gallery_features):
            print("Provide --features + --protocol, or --query-features + --gallery-features")
            sys.exit(1)

        q_feat = torch.load(args.query_features, weights_only=True)
        q_pids = torch.load(args.query_pids, weights_only=True)
        q_camids = torch.load(args.query_camids, weights_only=True)
        g_feat = torch.load(args.gallery_features, weights_only=True)
        g_pids = torch.load(args.gallery_pids, weights_only=True)
        g_camids = torch.load(args.gallery_camids, weights_only=True)

        qc = q_pids.shape[0]
        gc = g_pids.shape[0]
        sc = len(set(q_pids.tolist()) & set(g_pids.tolist()))

        results = _run_and_print(
            evaluator, q_feat, q_pids, q_camids,
            g_feat, g_pids, g_camids,
            qc, gc, sc, "manual", args.metric,
        )


if __name__ == "__main__":
    main()
