"""Structural sanity check for the HITSZ-VCM dataset pipeline.

Usage:
    python scripts/sanity_check_dataset.py --config configs/baseline_hitsz.yaml

Convention: All dataset configuration (root, seq_len, img_size) is read
from the ``data`` section in the YAML config.
"""

import argparse
import os
import sys
import yaml

import torch
from torch.utils.data import DataLoader

from data.hitsz_vcm import HITSZVCM
from data.transforms import get_video_transforms
from data.collate import collate_video_fn


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check(condition, msg):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    return condition


def main():
    parser = argparse.ArgumentParser(description="Sanity check HITSZ-VCM dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    root = data_cfg.get("root", "./data/hitsz_vcm")
    seq_len = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))

    print("=" * 60)
    print("HITSZ-VCM Dataset Sanity Check")
    print("=" * 60)
    print(f"  Config file: {os.path.abspath(args.config)}")
    print(f"  Data root:   {os.path.abspath(root)}")
    print(f"  Seq len:     {seq_len}")
    print(f"  Image size:  {img_size}")
    print(f"  Root exists: {os.path.isdir(root)}")
    print()

    # ------------------------------------------------------------------
    # 1. Check root existence
    # ------------------------------------------------------------------
    if not os.path.isdir(root):
        print(f"  [FAIL] Dataset not found at: {os.path.abspath(root)}")
        print()
        print(f"  Mount or copy HITSZ-VCM here, then re-run.")
        print(f"  Expected structure: {root}/train/pid_XXXX/...")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Instantiate dataset
    # ------------------------------------------------------------------
    transform = get_video_transforms(mode="train", img_size=img_size)
    dataset = HITSZVCM(root=root, seq_len=seq_len,
                       transform=transform, split="train")
    print(f"  Dataset loaded: {len(dataset)} tracklets found")
    print()

    # ------------------------------------------------------------------
    # 3. Dataset-level stats
    # ------------------------------------------------------------------
    unique_pids = set()
    for s in dataset.samples:
        unique_pids.add(s["pid"])
    print(f"  Unique identities (pids): {len(unique_pids)}")
    print(f"  Sample modalities: {set(s['modality'] for s in dataset.samples[:20])}")
    print()

    # ------------------------------------------------------------------
    # 4. Sample-level validation
    # ------------------------------------------------------------------
    n_samples = min(len(dataset), 5)
    print(f"  Validating {n_samples} samples...")
    all_pass = True

    for i in range(n_samples):
        sample = dataset[i]
        frames = sample["frames"]

        t_ok = check(frames.ndim == 4, f"Sample {i}: frames.ndim == 4")
        t, c, h, w = frames.shape if frames.ndim == 4 else (-1, -1, -1, -1)

        if frames.ndim == 4:
            check(t == seq_len, f"Sample {i}: T == {seq_len} (got {t})")
            check(c == 3, f"Sample {i}: C == 3 (got {c})")
            check((h, w) == img_size, f"Sample {i}: HxW == {img_size} (got {(h, w)})")

        check("pid" in sample and isinstance(sample["pid"], int),
              f"Sample {i}: pid = {sample.get('pid')}")
        check("camid" in sample and isinstance(sample["camid"], int),
              f"Sample {i}: camid = {sample.get('camid')}")
        check("modality" in sample and sample["modality"] in ("rgb", "ir"),
              f"Sample {i}: modality = {sample.get('modality')}")
        check("track_id" in sample and isinstance(sample["track_id"], int),
              f"Sample {i}: track_id = {sample.get('track_id')}")

        all_pass &= t_ok

    print()

    # ------------------------------------------------------------------
    # 5. Batch-level validation via DataLoader
    # ------------------------------------------------------------------
    print("  Testing DataLoader with collate_video_fn...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=range(min(4, len(dataset))),
        collate_fn=collate_video_fn,
    )
    batch = next(iter(loader))

    b_frames = batch["frames"]
    b_ok = check(b_frames.ndim == 5,
                 f"Batch frames shape = {tuple(b_frames.shape)}")
    all_pass &= b_ok

    if b_frames.ndim == 5:
        b, t, c, h, w = b_frames.shape
        check(b == min(4, len(dataset)),
              f"Batch dim B = {b} (expected {min(4, len(dataset))})")
        check(t == seq_len, f"Batch dim T = {t} (expected {seq_len})")
        check(c == 3, f"Batch dim C = {c} (expected 3)")
        check((h, w) == img_size,
              f"Batch HxW = {(h, w)} (expected {img_size})")

    check("pids" in batch and batch["pids"].ndim == 1,
          f"Batch pids shape = {tuple(batch['pids'].shape)}")
    check("camids" in batch and batch["camids"].ndim == 1,
          f"Batch camids shape = {tuple(batch['camids'].shape)}")
    check("modalities" in batch and len(batch["modalities"]) == b,
          f"Batch modalities = {batch.get('modalities', [])}")
    check("track_ids" in batch and batch["track_ids"].ndim == 1,
          f"Batch track_ids shape = {tuple(batch['track_ids'].shape)}")

    print()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("-" * 60)
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED")
    print("-" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
