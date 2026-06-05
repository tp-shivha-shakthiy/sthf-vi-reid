import argparse
import yaml
import torch

from models.baseline import BaselineModel
from models.sthf_model import STHFModel


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

    if model_name == "sthf_fixed":
        return STHFModel(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
            sthpf_type="fixed",
        )

    if model_name == "sthf_adaptive":
        return STHFModel(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
            sthpf_type="adaptive",
        )

    raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    sequence_length = config["dataset"].get("sequence_length", 6)
    image_size = config["dataset"].get("image_size", [288, 144])
    height, width = image_size

    model = build_model(config, num_classes=10)
    model.eval()

    frames = torch.randn(2, sequence_length, 3, height, width)

    with torch.no_grad():
        outputs = model(frames, modalities=["rgb", "ir"])

    required_keys = ["features", "logits", "int_features", "int_logits", "extra"]
    for key in required_keys:
        if key not in outputs:
            raise KeyError(f"Missing output key: {key}")

    print("Model sanity check passed.")
    print(f"Input frames: {frames.shape}")
    print(f"features: {outputs['features'].shape}")
    print(f"logits: {outputs['logits'].shape}")

    if outputs["int_features"] is None:
        print("int_features: None")
    else:
        print(f"int_features: {outputs['int_features'].shape}")

    if outputs["int_logits"] is None:
        print("int_logits: None")
    else:
        print(f"int_logits: {outputs['int_logits'].shape}")

    print(f"extra: {outputs['extra']}")


if __name__ == "__main__":
    main()
