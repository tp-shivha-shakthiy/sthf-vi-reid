import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

from models.baseline import BaselineModel
from models.sthf_model import STHFModel
from data.hitsz_vcm import HITSZVCM
from data.transforms import get_video_transforms
from data.collate import collate_video_fn


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
    parser = argparse.ArgumentParser(description="Extract ReID features from a trained checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pth file (unused in --debug mode)")
    parser.add_argument("--output-dir", default=".", help="Directory to save features.pt")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real HITSZ-VCM test split")
    parser.add_argument("--debug", action="store_true",
                        help="Extract features from synthetic data (no dataset required)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    num_classes = config.get("model", {}).get("num_classes", 10)
    seq_len = data_cfg.get("seq_len", 6)
    img_size = tuple(data_cfg.get("img_size", [288, 144]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, num_classes=num_classes)

    if args.debug:
        print("Debug mode: using model with random weights (no checkpoint loaded)")
    elif args.checkpoint is None:
        print("Provide --checkpoint for real feature extraction, or --debug for synthetic data.")
        sys.exit(1)
    else:
        if not os.path.isfile(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint} (epoch {state.get('epoch', '?')})")

    model.to(device)
    model.eval()

    if args.real_data:
        root = data_cfg.get("root", "./data/hitsz_vcm")
        if not os.path.isdir(root):
            print(f"Dataset not found at: {os.path.abspath(root)}")
            print("Mount or copy HITSZ-VCM here and re-run with --real-data.")
            sys.exit(1)

        transform = get_video_transforms(mode="test", img_size=img_size)
        dataset = HITSZVCM(root=root, seq_len=seq_len,
                           transform=transform, split="test")
        loader = DataLoader(
            dataset,
            batch_size=data_cfg.get("batch_size", 16),
            shuffle=False,
            collate_fn=collate_video_fn,
        )
        print(f"Real test dataset loaded: {len(dataset)} tracklets")
    elif args.debug:
        batch_size = data_cfg.get("batch_size", 16)
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
        loader = [dummy]
        print("Debug mode: using synthetic batch")
    else:
        print("Use --debug for synthetic data or --real-data for real dataset.")
        sys.exit(1)

    all_features = []
    all_pids = []
    all_camids = []
    all_modalities = []
    all_track_ids = []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            outputs = model(frames, modalities=batch.get("modalities"))
            feats = outputs["features"].cpu()
            all_features.append(feats)
            all_pids.append(batch["pids"].cpu())
            all_camids.append(batch["camids"].cpu())
            all_modalities.extend(batch["modalities"])
            all_track_ids.append(
                batch["track_ids"].cpu() if isinstance(batch["track_ids"], torch.Tensor)
                else torch.tensor(batch["track_ids"])
            )

    features = torch.cat(all_features, dim=0)
    pids = torch.cat(all_pids, dim=0)
    camids = torch.cat(all_camids, dim=0)

    track_ids_list = []
    for t in all_track_ids:
        if t.ndim == 0:
            track_ids_list.append(t.unsqueeze(0))
        else:
            track_ids_list.append(t)
    track_ids = torch.cat(track_ids_list, dim=0)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "features.pt")

    save_data = {
        "features": features,
        "pids": pids,
        "camids": camids,
        "modalities": all_modalities,
        "track_ids": track_ids,
    }

    torch.save(save_data, output_path)
    print(f"Saved features to {output_path}")
    print(f"  features shape: {features.shape}")
    print(f"  pids shape:     {pids.shape}")
    print(f"  modalities:     {len(all_modalities)} samples "
          f"({sum(1 for m in all_modalities if m == 'rgb')} rgb, "
          f"{sum(1 for m in all_modalities if m == 'ir')} ir)")


if __name__ == "__main__":
    main()
