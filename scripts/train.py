import argparse
import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.baseline import BaselineModel
from models.sthf_model import STHFModel
from data.hitsz_vcm import HITSZVCM
from data.transforms import get_video_transforms
from data.collate import collate_video_fn
from losses.build_loss import build_loss
from engine.trainer import Trainer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config, num_classes=10):
    model_name = config["model"]["name"]
    pretrained = config["model"].get("pretrained", False)
    feature_dim = config["model"].get("feature_dim", 2048)

    if model_name == "baseline":
        return BaselineModel(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
        )

    if model_name in ("sthf_fixed", "sthf_adaptive"):
        sthpf_type = model_name.replace("sthf_", "")
        return STHFModel(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
            sthpf_type=sthpf_type,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real HITSZ-VCM batches instead of dummy tensors")
    args = parser.parse_args()

    config = load_config(args.config)

    if not args.debug:
        parser.error("Only --debug mode is currently supported for training.")

    data_cfg = config.get("data", {})
    sequence_length = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))
    height, width = img_size
    batch_size = 4
    num_classes = config.get("model", {}).get("num_classes", 10)
    lr = config.get("train", {}).get("lr", 3.5e-4)

    model = build_model(config, num_classes=num_classes)
    criterion = build_loss(config)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, criterion, optimizer, config)

    root = data_cfg.get("root", "./data/hitsz_vcm")

    if args.real_data:
        if not os.path.isdir(root):
            print(f"Dataset not found at: {os.path.abspath(root)}")
            print("Mount or copy HITSZ-VCM here and re-run with --real-data.")
            sys.exit(1)

        transform = get_video_transforms(mode="train", img_size=img_size)
        dataset = HITSZVCM(root=root, seq_len=sequence_length,
                           transform=transform, split="train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_video_fn,
        )

        print(f"Real dataset loaded: {len(dataset)} tracklets")
        batch = next(iter(loader))
        print(f"First batch frames shape: {batch['frames'].shape}")
    else:
        batch = {
            "frames": torch.randn(batch_size, sequence_length, 3, height, width),
            "pids": torch.tensor([0, 0, 1, 1]),
            "camids": torch.zeros(batch_size, dtype=torch.long),
            "modalities": ["rgb"] * batch_size,
            "track_ids": torch.zeros(batch_size, dtype=torch.long),
        }

    losses = trainer.training_step(batch)

    print("Debug training step completed successfully.")
    print(f"  id_loss:           {losses['id_loss'].item():.6f}")
    print(f"  triplet_loss:      {losses['triplet_loss'].item():.6f}")
    print(f"  int_id_loss:       {losses['int_id_loss'].item():.6f}")
    print(f"  int_triplet_loss:  {losses['int_triplet_loss'].item():.6f}")
    print(f"  total:             {losses['total'].item():.6f}")


if __name__ == "__main__":
    main()
