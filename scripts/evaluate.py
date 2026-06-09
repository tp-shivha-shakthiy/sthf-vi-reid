"""Lightweight evaluation entry point.

Loads pre-computed feature tensors and metadata from disk, runs ReID
evaluation, and prints results.

Usage:
    python scripts/evaluate.py \\
        --query-features path/to/q_feat.pt \\
        --query-pids     path/to/q_pids.pt \\
        --query-camids   path/to/q_camids.pt \\
        --gallery-features path/to/g_feat.pt \\
        --gallery-pids     path/to/g_pids.pt \\
        --gallery-camids   path/to/g_camids.pt
"""

import argparse
import torch
from metrics.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="ReID evaluation from saved features")
    parser.add_argument("--query-features", required=True)
    parser.add_argument("--query-pids", required=True)
    parser.add_argument("--query-camids", required=True)
    parser.add_argument("--gallery-features", required=True)
    parser.add_argument("--gallery-pids", required=True)
    parser.add_argument("--gallery-camids", required=True)
    parser.add_argument("--metric", default="cosine", choices=("cosine", "euclidean"))
    args = parser.parse_args()

    q_feat = torch.load(args.query_features, weights_only=True)
    q_pids = torch.load(args.query_pids, weights_only=True)
    q_camids = torch.load(args.query_camids, weights_only=True)
    g_feat = torch.load(args.gallery_features, weights_only=True)
    g_pids = torch.load(args.gallery_pids, weights_only=True)
    g_camids = torch.load(args.gallery_camids, weights_only=True)

    evaluator = Evaluator(metric=args.metric)
    results = evaluator.evaluate(q_feat, q_pids, q_camids, g_feat, g_pids, g_camids)

    print("=" * 40)
    print("ReID Evaluation Results")
    print("=" * 40)
    for key, val in results.items():
        print(f"  {key:>8s}: {val:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
