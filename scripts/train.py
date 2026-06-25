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
from data.transforms import get_video_transforms, build_caj
from data.collate import collate_video_fn
from data.video_sampler import VideoSampler
from losses.build_loss import build_loss
from engine.trainer import Trainer
from engine.seed import set_seed


# -----------------------------
# Utils
# -----------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config, num_classes):
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
        return STHFModel(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
            sthpf_type=model_name.replace("sthf_", ""),
        )

    raise ValueError(f"Unknown model: {model_name}")


def safe_next(loader):
    """Prevents crash when loader is empty or sampler misconfigured."""
    try:
        return next(iter(loader))
    except StopIteration:
        print("[ERROR] DataLoader returned no batches.")
        sys.exit(1)


def check_cuda():
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("[WARNING] CUDA NOT available — running on CPU")
        return torch.device("cpu")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # -----------------------------
    # Seed
    # -----------------------------
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    device = check_cuda()

    # -----------------------------
    # Config
    # -----------------------------
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    root = data_cfg.get("root", "./data/hitsz_vcm")
    seq_len = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))
    batch_size = data_cfg.get("batch_size", 16)

    num_classes = config.get("model", {}).get("num_classes", 10)
    lr = train_cfg.get("lr", 3.5e-4)
    epochs = args.epochs if args.epochs else train_cfg.get("epochs", 60)

    model_name = config["model"]["name"]

    print("\n==============================")
    print(f"Experiment: {model_name}")
    print("==============================")

    # -----------------------------
    # Model / Loss / Optim
    # -----------------------------
    model = build_model(config, num_classes=num_classes).to(device)
    criterion = build_loss(config)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    caj = build_caj(config.get("augmentation", {}).get("caj"))

    trainer = Trainer(model, criterion, optimizer, scheduler, config, caj=caj)

    # -----------------------------
    # Resume
    # -----------------------------
    start_epoch = 1
    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"[ERROR] Resume not found: {args.resume}")
            sys.exit(1)

        ckpt = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

        print(f"[INFO] Resumed from {args.resume}")

    # -----------------------------
    # Dataset
    # -----------------------------
    if args.real_data:
        if not os.path.isdir(root):
            print(f"[ERROR] Dataset not found: {root}")
            sys.exit(1)

        transform = get_video_transforms(mode="train", img_size=img_size)

        dataset = HITSZVCM(
            root=root,
            seq_len=seq_len,
            transform=transform,
            split="train"
        )

        sampling_cfg = config.get("sampling", {})
        num_ids = sampling_cfg.get("num_ids", 4)
        clips_per_id = sampling_cfg.get("clips_per_id", 4)

        sampler = VideoSampler(
            dataset,
            num_ids=num_ids,
            clips_per_id=clips_per_id,
            shuffle=True,
        )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_video_fn,
            num_workers=data_cfg.get("num_workers",4),
            pin_memory=data_cfg.get("pin_memory",True),
        )

        print(f"Dataset: {len(dataset)} tracklets")
        print(f"Batch: {num_ids} IDs × {clips_per_id} clips")

        batch = safe_next(loader)
        print("Sample batch shape:", batch["frames"].shape)

    else:
        # dummy mode
        b = batch_size
        batch = {
            "frames": torch.randn(b, seq_len, 3, *img_size),
            "pids": torch.arange(b),
            "camids": torch.zeros(b, dtype=torch.long),
            "modalities": ["rgb"] * b,
            "track_ids": torch.zeros(b, dtype=torch.long),
        }

    # move to device
    batch["frames"] = batch["frames"].to(device)
    batch["pids"] = batch["pids"].to(device)

    # -----------------------------
    # Debug mode
    # -----------------------------
    if args.debug:
        print("\n[DEBUG] Single forward pass")

        model.eval()
        with torch.no_grad():
            out = model(batch["frames"], modalities=batch.get("modalities"))
            loss = criterion(out, batch["pids"])

        print("Loss breakdown:")
        for k, v in loss.items():
            print(f"  {k}: {v.item():.6f}")

        print("Features:", out["features"].shape)
        print("Logits:", out["logits"].shape)
        return

    # -----------------------------
    # Train
    # -----------------------------
    if args.real_data:
        if epochs < 50:
            print("[WARNING] Very low epochs → not meaningful training")

        print(f"\nTraining {model_name}")
        print(f"Epochs: {epochs} | LR: {lr}")

        if args.max_batches:
            print(f"[SMOKE TEST] max_batches={args.max_batches}")

        trainer.fit(
            loader,
            None,
            epochs,
            max_batches=args.max_batches,
            start_epoch=start_epoch,
        )

        save_dir = train_cfg.get("save_dir", f"experiments/{model_name}")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "last.pth")
        torch.save(model.state_dict(), save_path)

        print(f"\nModel saved to: {save_path}")
        return

    print("Use --debug or --real-data")
    sys.exit(1)


if __name__ == "__main__":
    main()