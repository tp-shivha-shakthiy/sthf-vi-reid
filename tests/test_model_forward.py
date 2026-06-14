import torch
import pytest

from models.baseline import BaselineModel
from models.sthf_model import STHFModel
from models.sthpf import FixedSTHPF
from models.modality_purifier import ModalityPurifier, ChannelAttention


class TestModalityPurifier:
    """Shape and behaviour verification for ModalityPurifier."""

    # ------------------------------------------------------------------
    # 4D Input
    # ------------------------------------------------------------------

    def test_4d_shape_preserved(self):
        B, C, H, W = 8, 2048, 7, 7
        x = torch.randn(B, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert out.shape == x.shape, f"4D shape mismatch: {out.shape} vs {x.shape}"

    def test_4d_forward_no_error(self):
        B, C, H, W = 4, 2048, 7, 7
        x = torch.randn(B, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_4d_dtype_preserved(self):
        x = torch.randn(4, 2048, 7, 7)
        model = ModalityPurifier(num_channels=2048)
        out = model(x)
        assert out.dtype == x.dtype, f"dtype changed: {out.dtype} vs {x.dtype}"

    def test_4d_values_modified(self):
        """Verify that the purifier actually transforms the input."""
        x = torch.randn(4, 2048, 7, 7)
        model = ModalityPurifier(num_channels=2048)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert not torch.allclose(x, out, atol=1e-4), (
            "ModalityPurifier should modify input features"
        )

    # ------------------------------------------------------------------
    # 5D Input
    # ------------------------------------------------------------------

    def test_5d_shape_preserved(self):
        B, T, C, H, W = 2, 6, 2048, 7, 7
        x = torch.randn(B, T, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert out.shape == x.shape, f"5D shape mismatch: {out.shape} vs {x.shape}"

    def test_5d_batch_temporal_unchanged(self):
        """Batch and temporal dimensions must be preserved exactly."""
        B, T, C, H, W = 2, 6, 2048, 7, 7
        x = torch.randn(B, T, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert out.shape[0] == B, f"Batch dim changed: {out.shape[0]} vs {B}"
        assert out.shape[1] == T, f"Temporal dim changed: {out.shape[1]} vs {T}"

    def test_5d_forward_no_error(self):
        B, T, C, H, W = 2, 6, 2048, 7, 7
        x = torch.randn(B, T, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_5d_dtype_preserved(self):
        x = torch.randn(2, 6, 2048, 7, 7).float()
        model = ModalityPurifier(num_channels=2048)
        out = model(x)
        assert out.dtype == x.dtype

    def test_5d_values_modified(self):
        x = torch.randn(2, 6, 2048, 7, 7)
        model = ModalityPurifier(num_channels=2048)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert not torch.allclose(x, out, atol=1e-4), (
            "ModalityPurifier should modify 5D input features"
        )

    # ------------------------------------------------------------------
    # Invalid input
    # ------------------------------------------------------------------

    def test_3d_raises(self):
        x = torch.randn(4, 2048, 7)
        model = ModalityPurifier(num_channels=2048)
        with pytest.raises(ValueError, match="4D or 5D"):
            model(x)


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
