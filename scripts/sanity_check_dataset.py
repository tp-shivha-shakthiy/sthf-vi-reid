"""Structural sanity check for the HITSZ-VCM dataset pipeline.

Usage:
    python scripts/sanity_check_dataset.py --config configs/baseline_hitsz.yaml
"""

import argparse
import sys
import yaml

import torch
from data.hitsz_vcm import HITSZVCM
from data.transforms import get_video_transforms
from data.collate import collate_video_fn


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check(condition, msg, pass_msg=None):
    if condition:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"  [{status}] {msg}")
    return condition


def main():
    parser = argparse.ArgumentParser(description="Sanity check HITSZ-VCM dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "./data/hitsz_vcm")
    img_size = tuple(cfg.get("dataset", {}).get("image_size",
                      data_cfg.get("img_size", [288, 144])))
    seq_len = cfg.get("dataset", {}).get("sequence_length",
                                          data_cfg.get("seq_len", 6))

    print("=" * 60)
    print("HITSZ-VCM Dataset Sanity Check")
    print("=" * 60)
    print(f"  Data root:  {root}")
    print(f"  Seq len:    {seq_len}")
    print(f"  Image size: {img_size}")
    print()

    # ------------------------------------------------------------------
    # 1. Instantiate dataset
    # ------------------------------------------------------------------
    try:
        transform = get_video_transforms(mode="train", img_size=img_size)
        dataset = HITSZVCM(root=root, seq_len=seq_len,
                           transform=transform, split="train")
        print(f"  Dataset loaded: {len(dataset)} tracklets found")
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")
        print("\n  Skipping sample-level checks (no data on disk).")
        print("  Code structure validated successfully.")
        sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] Failed to instantiate dataset: {e}")
        sys.exit(1)

    n_samples = min(len(dataset), 10)

    # ------------------------------------------------------------------
    # 2. Sample-level validation
    # ------------------------------------------------------------------
    print(f"\n  Validating {n_samples} samples...")
    all_pass = True
    shapes_ok = True
    metas_ok = True

    for i in range(n_samples):
        try:
            sample = dataset[i]
        except Exception as e:
            print(f"  [FAIL] Sample {i} raised exception: {e}")
            all_pass = False
            continue

        # Check frames
        frames = sample.get("frames")
        frames_ok = check(
            isinstance(frames, torch.Tensor),
            f"Sample {i}: frames is a Tensor",
        )
        all_pass &= frames_ok
        shapes_ok &= frames_ok

        if frames is not None:
            shape_ok = check(
                frames.ndim == 4,
                f"Sample {i}: frames.ndim == 4",
            )
            all_pass &= shape_ok
            shapes_ok &= shape_ok

            if frames.ndim == 4:
                t, c, h, w = frames.shape
                t_ok = check(
                    t == seq_len,
                    f"Sample {i}: frames.shape[0] == T ({t} == {seq_len})",
                )
                c_ok = check(
                    c == 3,
                    f"Sample {i}: frames.shape[1] == C ({c} == 3)",
                )
                hw_ok = check(
                    (h, w) == img_size,
                    f"Sample {i}: frames.shape[2:4] == HxW ({(h, w)} == {img_size})",
                )
                all_pass &= t_ok & c_ok & hw_ok
                shapes_ok &= t_ok & c_ok & hw_ok

        # Check metadata
        pid_ok = check(
            "pid" in sample and isinstance(sample["pid"], int),
            f"Sample {i}: pid exists and is int",
        )
        camid_ok = check(
            "camid" in sample and isinstance(sample["camid"], int),
            f"Sample {i}: camid exists and is int",
        )
        mod_ok = check(
            "modality" in sample and sample["modality"] in ("rgb", "ir"),
            f"Sample {i}: modality is 'rgb' or 'ir'",
        )
        tid_ok = check(
            "track_id" in sample and isinstance(sample["track_id"], int),
            f"Sample {i}: track_id exists and is int",
        )
        all_pass &= pid_ok & camid_ok & mod_ok & tid_ok
        metas_ok &= pid_ok & camid_ok & mod_ok & tid_ok

    # ------------------------------------------------------------------
    # 3. Batch-level validation
    # ------------------------------------------------------------------
    print(f"\n  Validating batch collation...")
    try:
        batch_samples = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collate_video_fn(batch_samples)
        b_frames = batch.get("frames")

        b_frames_ok = check(
            isinstance(b_frames, torch.Tensor) and b_frames.ndim == 5,
            f"Batch frames shape = {tuple(b_frames.shape)} (expected 5D)",
        )
        b_pids_ok = check(
            "pids" in batch and batch["pids"].ndim == 1,
            f"Batch pids shape = {tuple(batch['pids'].shape)} (expected 1D)",
        )
        b_camids_ok = check(
            "camids" in batch and batch["camids"].ndim == 1,
            f"Batch camids shape = {tuple(batch['camids'].shape)} (expected 1D)",
        )
        b_mods_ok = check(
            "modalities" in batch and len(batch["modalities"]) == len(batch_samples),
            f"Batch modalities length = {len(batch.get('modalities', []))}",
        )
        b_tids_ok = check(
            "track_ids" in batch and batch["track_ids"].ndim == 1,
            f"Batch track_ids shape = {tuple(batch['track_ids'].shape)} (expected 1D)",
        )
        all_pass &= b_frames_ok & b_pids_ok & b_camids_ok & b_mods_ok & b_tids_ok

    except Exception as e:
        print(f"  [FAIL] Batch collation raised exception: {e}")
        all_pass = False

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        if not shapes_ok:
            print("  RESULT: SHAPE VALIDATION FAILED")
        if not metas_ok:
            print("  RESULT: METADATA VALIDATION FAILED")
        print("  RESULT: SOME CHECKS FAILED")
    print("-" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
