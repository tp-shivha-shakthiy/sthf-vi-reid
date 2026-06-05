import torch

from models.baseline import BaselineModel
from models.sthf_model import STHFModel


def test_baseline_forward_contract():
    model = BaselineModel(num_classes=10, pretrained=False)
    frames = torch.randn(2, 6, 3, 288, 144)

    outputs = model(frames)

    assert "features" in outputs
    assert "logits" in outputs
    assert "int_features" in outputs
    assert "int_logits" in outputs
    assert "extra" in outputs

    assert outputs["features"].shape == (2, 2048)
    assert outputs["logits"].shape == (2, 10)
    assert outputs["int_features"] is None
    assert outputs["int_logits"] is None


def test_sthf_forward_contract():
    model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
    frames = torch.randn(2, 6, 3, 288, 144)

    outputs = model(frames)

    assert "features" in outputs
    assert "logits" in outputs
    assert "int_features" in outputs
    assert "int_logits" in outputs
    assert "extra" in outputs

    assert outputs["features"].shape == (2, 2048)
    assert outputs["logits"].shape == (2, 10)
    assert outputs["int_features"].shape == (2, 2048)
    assert outputs["int_logits"].shape == (2, 10)
    assert outputs["extra"]["sthpf_type"] == "fixed"
