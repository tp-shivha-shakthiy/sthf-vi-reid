import os
import glob
import random

import torch
from PIL import Image
import torchvision.transforms.functional as F

from .base_video_dataset import BaseVideoDataset


class HITSZVCM(BaseVideoDataset):
    """HITSZ-VCM video-based visible-infrared person ReID dataset.

    Expected directory structure (configurable via path patterns):

        root/
            train/
                pid_XXXX/
                    rgb/
                        cam_XX/
                            track_YYYY/
                                frame_000001.jpg
                                frame_000002.jpg
                                ...
                    ir/
                        cam_XX/
                            track_YYYY/
                                ...
            test/
                pid_XXXX/
                    ...

    Each tracklet (video sequence) is stored as a directory of ordered frames.
    The scanner parses identity ID, camera ID, modality, and tracklet ID from
    the directory structure. Returns samples conforming to the BaseVideoDataset
    contract.
    """

    def __init__(self, root, seq_len=6, transform=None, split="train"):
        super().__init__(root, seq_len, transform)
        self.split = split
        self.samples = []
        self.pid2label = {}
        self.label2pid = {}
        self._scan()
        self._relabel_pids()

    def _parse_pid(self, dirname):
        """Extract numeric identity ID from a directory name."""
        digits = "".join(ch for ch in dirname if ch.isdigit())
        return int(digits) if digits else hash(dirname) % (10 ** 8)

    def _parse_camid(self, dirname):
        """Extract numeric camera ID from a directory name."""
        digits = "".join(ch for ch in dirname if ch.isdigit())
        return int(digits) if digits else 0

    def _parse_track_id(self, dirname):
        """Extract numeric tracklet ID from a directory name."""
        digits = "".join(ch for ch in dirname if ch.isdigit())
        return int(digits) if digits else 0

    def _discover_tracklets(self, cam_path, pid, camid, modality):
        """Find all video tracklets under a camera directory.

        Supports two layouts:
          - Nested tracklet subdirectories: cam/track_XXX/frame_XXX.jpg
          - Flat frame files directly under camera: cam/frame_XXX.jpg
        """
        entries = sorted(os.listdir(cam_path))
        subdirs = [e for e in entries
                   if os.path.isdir(os.path.join(cam_path, e))]

        if subdirs:
            # Each subdirectory is a separate tracklet
            for track_dir in subdirs:
                track_path = os.path.join(cam_path, track_dir)
                frames = sorted(glob.glob(os.path.join(track_path, "*.jpg")) +
                                glob.glob(os.path.join(track_path, "*.png")) +
                                glob.glob(os.path.join(track_path, "*.jpeg")))
                if len(frames) >= self.seq_len:
                    track_id = self._parse_track_id(track_dir)
                    self.samples.append({
                        "pid": pid,
                        "camid": camid,
                        "modality": modality,
                        "track_id": track_id,
                        "frames": frames,
                    })
        else:
            # Flat frame files in the camera directory
            frames = sorted(glob.glob(os.path.join(cam_path, "*.jpg")) +
                            glob.glob(os.path.join(cam_path, "*.png")) +
                            glob.glob(os.path.join(cam_path, "*.jpeg")))
            if len(frames) >= self.seq_len:
                self.samples.append({
                    "pid": pid,
                    "camid": camid,
                    "modality": modality,
                    "track_id": 0,
                    "frames": frames,
                })

    def _scan(self):
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Ensure HITSZ-VCM data is placed at {self.root}."
            )

        pid_dirs = sorted(os.listdir(split_dir))
        for pid_dir in pid_dirs:
            pid_path = os.path.join(split_dir, pid_dir)
            if not os.path.isdir(pid_path):
                continue
            if pid_dir.startswith("."):
                continue

            pid = self._parse_pid(pid_dir)

            for modality in ("rgb", "ir"):
                mod_path = os.path.join(pid_path, modality)
                if not os.path.isdir(mod_path):
                    continue

                cam_dirs = sorted(os.listdir(mod_path))
                for cam_dir in cam_dirs:
                    cam_path = os.path.join(mod_path, cam_dir)
                    if not os.path.isdir(cam_path):
                        continue
                    if cam_dir.startswith("."):
                        continue

                    camid = self._parse_camid(cam_dir)
                    self._discover_tracklets(cam_path, pid, camid, modality)

    def _relabel_pids(self):
        """Map raw person IDs to contiguous training labels starting at 0."""
        unique_pids = sorted({sample["pid"] for sample in self.samples})
        self.pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}
        self.label2pid = {idx: pid for pid, idx in self.pid2label.items()}
        for sample in self.samples:
            sample["raw_pid"] = sample["pid"]
            sample["pid"] = self.pid2label[sample["pid"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = sample["frames"]
        n_frames = len(frame_paths)

        if n_frames < self.seq_len:
            raise RuntimeError(
                f"Tracklet has {n_frames} frames, but seq_len={self.seq_len}."
            )

        start = random.randint(0, n_frames - self.seq_len)
        selected_paths = frame_paths[start:start + self.seq_len]

        frames = []
        for path in selected_paths:
            img = Image.open(path).convert("RGB")
            frames.append(img)

        if self.transform:
            frames = [self.transform(f) for f in frames]
        else:
            frames = [F.to_tensor(f) for f in frames]

        frames_tensor = torch.stack(frames, dim=0)

        return {
            "frames": frames_tensor,
            "pid": sample["pid"],
            "camid": sample["camid"],
            "modality": sample["modality"],
            "track_id": sample["track_id"],
        }
