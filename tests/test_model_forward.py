import torch

from models.baseline import BaselineModel
from models.sthf_model import STHFModel
from models.sthpf import FixedSTHPF


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
    assert outputs["extra"]["model_type"] == "baseline"


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


def test_sthf_uses_sthpf_module():
    model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
    assert hasattr(model, "sthpf")
    assert isinstance(model.sthpf, FixedSTHPF)


def test_sthf_int_features_differ_from_original():
    model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
    model.eval()

    torch.manual_seed(42)
    frames = torch.randn(2, 6, 3, 288, 144)

    with torch.no_grad():
        outputs = model(frames)

    orig_out = outputs["features"]
    int_out = outputs["int_features"]

    assert not torch.allclose(orig_out, int_out, atol=1e-6), (
        "Intermediate branch should produce different features from original branch"
    )


def test_sthf_model_has_sthpf_in_extra():
    model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
    frames = torch.randn(2, 6, 3, 288, 144)
    outputs = model(frames)

    assert outputs["extra"]["sthpf_type"] == "fixed"
