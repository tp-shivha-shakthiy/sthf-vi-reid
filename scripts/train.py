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
    parser.add_argument("--debug", action="store_true",
                        help="Single training step with dummy batch")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real HITSZ-VCM batches instead of dummy tensors")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: config value)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after N batches per epoch (smoke test)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: config value or 42)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Seed handling: CLI arg > config > default 42
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    model_name = config["model"]["name"]

    # Verify adaptive configuration
    sthpf_type = config.get("model", {}).get("sthpf", {}).get("type", "fixed")
    print(f"Experiment: {model_name}")
    print(f"STHPF Type: {sthpf_type}")
    if sthpf_type == "adaptive":
        print("AdaptiveSTHPF gating weights will be logged in metadata.")

    sequence_length = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))
    height, width = img_size
    batch_size = data_cfg.get("batch_size", 16)
    num_classes = config.get("model", {}).get("num_classes", 10)
    lr = train_cfg.get("lr", 3.5e-4)
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 60)

    model = build_model(config, num_classes=num_classes)
    criterion = build_loss(config)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    caj = build_caj(config.get("augmentation", {}).get("caj"))
    trainer = Trainer(model, criterion, optimizer, scheduler, config, caj=caj)

    # Resume logic
    start_epoch = 1
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            print(f"Resume checkpoint not found: {args.resume}")
            sys.exit(1)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Resumed model weights from {args.resume}")
        else:
            model.load_state_dict(ckpt)
            print(f"Resumed model weights from {args.resume} (raw state_dict, no epoch/optimizer)")
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("  Resumed optimizer state")
        else:
            print("  WARNING: checkpoint has no optimizer_state_dict — optimizer reset to initial state")
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"  Resuming from epoch {start_epoch}")
        else:
            print("  WARNING: checkpoint has no epoch — starting from epoch 1")

    root = data_cfg.get("root", "./data/hitsz_vcm")

    if args.real_data:
        if not os.path.isdir(root):
            print(f"Dataset not found at: {os.path.abspath(root)}")
            print("Mount or copy HITSZ-VCM here and re-run with --real-data.")
            sys.exit(1)

        transform = get_video_transforms(mode="train", img_size=img_size)
        dataset = HITSZVCM(root=root, seq_len=sequence_length,
                           transform=transform, split="train")

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
        )

        print(f"  VideoSampler: {num_ids} ids × {clips_per_id} clips = {num_ids * clips_per_id} per batch")

        print(f"Real dataset loaded: {len(dataset)} tracklets")
        batch = next(iter(loader))
        print(f"First batch frames shape: {batch['frames'].shape}")
    else:
        debug_batch_size = 1 if args.debug else batch_size
        dummy_pids = torch.arange(debug_batch_size)
        dummy_modalities = ["rgb"] if debug_batch_size == 1 else (["rgb"] * (debug_batch_size // 2) + ["ir"] * (debug_batch_size - debug_batch_size // 2))
        batch = {
            "frames": torch.randn(debug_batch_size, sequence_length, 3, height, width),
            "pids": dummy_pids,
            "camids": torch.zeros(debug_batch_size, dtype=torch.long),
            "modalities": dummy_modalities,
            "track_ids": torch.zeros(debug_batch_size, dtype=torch.long),
        }

    if args.debug:
        print("Debug mode: forward pass + loss computation (no gradient)")
        model.eval()
        with torch.no_grad():
            outputs = model(batch["frames"], modalities=batch.get("modalities"))
            losses = criterion(outputs, batch["pids"])
        print("Debug training step completed successfully.")
        for key in ("loss_total", "loss_id", "loss_tri", "loss_id_int", "loss_tri_int"):
            val = losses[key].item()
            print(f"  {key:15s}: {val:.6f}")
        if hasattr(model, "sthpf") and hasattr(model.sthpf, "latest_gate_weights"):
            fw = model.sthpf.latest_gate_weights
            if fw is not None:
                print(f"  gate_weights mean: {fw.mean(dim=0).tolist()}")
        print("Feature shape:", outputs["features"].shape)
        print("Logits shape:",  outputs["logits"].shape)
    elif args.real_data:
        if epochs < 200:
            print("WARNING: Running with fewer than 200 epochs — results are preliminary, not full reproduction.")
        print(f"Training {model_name} for {epochs} epochs (lr={lr}, batch_size={batch_size})")
        if args.max_batches is not None:
            print(f"  Smoke-test mode: max {args.max_batches} batch(es) per epoch")
        trainer.fit(loader, None, epochs, max_batches=args.max_batches,
                    start_epoch=start_epoch)

        save_dir = train_cfg.get("save_dir", f"experiments/{model_name}")
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, "last.pth")
        torch.save(model.state_dict(), final_path)
        print(f"Final model weights saved to {final_path}")
    else:
        print("Use --debug for a single step or --real-data for full training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
