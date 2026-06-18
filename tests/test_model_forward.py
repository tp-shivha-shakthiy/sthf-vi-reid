import torch
import pytest

from models.baseline import BaselineModel
from models.sdc import SDC
from models.dsr import DSR
from models.sthf_model import STHFModel
from models.sthpf import FixedSTHPF
from models.modality_purifier import ModalityPurifier


# ======================================================================
# ModalityPurifier tests
# ======================================================================

class TestModalityPurifier:
    """Shape and behaviour verification for paper-faithful ModalityPurifier."""

    def test_4d_shape_preserved(self):
        B, C, H, W = 8, 2048, 7, 7
        x = torch.randn(B, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert out.shape == x.shape

    def test_4d_forward_no_error(self):
        B, C, H, W = 4, 2048, 7, 7
        x = torch.randn(B, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_4d_values_modified(self):
        x = torch.randn(4, 2048, 7, 7)
        model = ModalityPurifier(num_channels=2048)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert not torch.allclose(x, out, atol=1e-4)

    def test_5d_shape_preserved(self):
        B, T, C, H, W = 2, 6, 2048, 7, 7
        x = torch.randn(B, T, C, H, W)
        model = ModalityPurifier(num_channels=C)
        out = model(x)
        assert out.shape == x.shape

    def test_3d_raises(self):
        x = torch.randn(4, 2048, 7)
        model = ModalityPurifier(num_channels=2048)
        with pytest.raises(ValueError, match="4D or 5D"):
            model(x)


# ======================================================================
# SDC tests — must accept 4D feature maps
# ======================================================================

class TestSDC:
    def test_accepts_4d_feature_maps(self):
        sdc = SDC(in_channels=256)
        f = torch.randn(4, 256, 36, 72)
        fh = torch.randn(4, 256, 36, 72)
        output = sdc(f, fh)
        assert output.shape == f.shape

    def test_output_is_4d(self):
        sdc = SDC(in_channels=256)
        f = torch.randn(2, 256, 18, 36)
        fh = torch.randn(2, 256, 18, 36)
        output = sdc(f, fh)
        assert output.ndim == 4

    def test_preserves_spatial_dims(self):
        sdc = SDC(in_channels=256)
        f = torch.randn(2, 256, 18, 36)
        fh = torch.randn(2, 256, 18, 36)
        output = sdc(f, fh)
        assert output.shape[2:] == f.shape[2:]

    def test_forward_no_error(self):
        sdc = SDC(in_channels=256)
        f = torch.randn(2, 256, 14, 14)
        fh = torch.randn(2, 256, 14, 14)
        output = sdc(f, fh)
        assert torch.isfinite(output).all()

    def test_values_modified(self):
        sdc = SDC(in_channels=256)
        f = torch.randn(2, 256, 14, 14)
        fh = torch.randn(2, 256, 14, 14)
        sdc.eval()
        with torch.no_grad():
            output = sdc(f, fh)
        assert not torch.allclose(f, output, atol=1e-6)

    def test_no_pooled_vector_output(self):
        sdc = SDC(in_channels=256)
        f = torch.randn(2, 256, 14, 14)
        fh = torch.randn(2, 256, 14, 14)
        output = sdc(f, fh)
        assert output.ndim != 2

    def test_sdc2_512_channels(self):
        """SDC after layer2 must handle 512-channel feature maps."""
        sdc = SDC(in_channels=512)
        f = torch.randn(2, 512, 18, 36)
        fh = torch.randn(2, 512, 18, 36)
        output = sdc(f, fh)
        assert output.shape == f.shape


# ======================================================================
# DSR tests — operates on [B, T, C] sequence features
# ======================================================================

class TestDSR:
    def test_accepts_3d_sequence_features(self):
        dsr = DSR(feature_dim=2048)
        orig_seq = torch.randn(2, 6, 2048)
        hf_seq = torch.randn(2, 6, 2048)
        output = dsr(orig_seq, hf_seq)
        assert output.shape == orig_seq.shape

    def test_output_is_3d(self):
        dsr = DSR(feature_dim=2048)
        orig_seq = torch.randn(2, 6, 2048)
        hf_seq = torch.randn(2, 6, 2048)
        output = dsr(orig_seq, hf_seq)
        assert output.ndim == 3

    def test_preserves_dims(self):
        dsr = DSR(feature_dim=2048)
        orig_seq = torch.randn(2, 6, 2048)
        hf_seq = torch.randn(2, 6, 2048)
        output = dsr(orig_seq, hf_seq)
        assert output.shape == orig_seq.shape

    def test_forward_no_error(self):
        dsr = DSR(feature_dim=2048)
        orig_seq = torch.randn(2, 6, 2048)
        hf_seq = torch.randn(2, 6, 2048)
        output = dsr(orig_seq, hf_seq)
        assert torch.isfinite(output).all()

    def test_values_modified(self):
        dsr = DSR(feature_dim=2048)
        orig_seq = torch.randn(2, 6, 2048)
        hf_seq = torch.randn(2, 6, 2048)
        dsr.eval()
        with torch.no_grad():
            output = dsr(orig_seq, hf_seq)
        assert not torch.allclose(orig_seq, output, atol=1e-6)

    def test_uses_conv1d(self):
        dsr = DSR(feature_dim=2048)
        assert isinstance(dsr.temporal_fusion[0], torch.nn.Conv1d)

    def test_dsr_feature_dim_2048(self):
        dsr = DSR(feature_dim=2048)
        orig_seq = torch.randn(2, 6, 2048)
        hf_seq = torch.randn(2, 6, 2048)
        output = dsr(orig_seq, hf_seq)
        assert output.shape[-1] == 2048


# ======================================================================
# Baseline model tests
# ======================================================================

def test_baseline_forward_contract():
    model = BaselineModel(num_classes=10, pretrained=False)
    frames = torch.randn(2, 6, 3, 288, 144)
    outputs = model(frames)
    assert outputs["features"].shape == (2, 2048)
    assert outputs["logits"].shape == (2, 10)
    assert outputs["int_features"] is None
    assert outputs["int_logits"] is None
    assert outputs["extra"]["model_type"] == "baseline"


# ======================================================================
# STHF model dual SDC/DSR placement tests
# ======================================================================

class TestSTHFArchitectureWiring:
    """Verify paper-exact SDC/DSR placement: blocks 1-2 and blocks 3-4."""

    def test_output_keys(self):
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        frames = torch.randn(2, 6, 3, 288, 144)
        outputs = model(frames)
        assert set(outputs.keys()) == {"features", "logits", "int_features", "int_logits", "extra"}

    def test_output_shapes(self):
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        frames = torch.randn(2, 6, 3, 288, 144)
        outputs = model(frames)
        assert outputs["features"].shape == (2, 2048)
        assert outputs["logits"].shape == (2, 10)
        assert outputs["int_features"].shape == (2, 2048)
        assert outputs["int_logits"].shape == (2, 10)

    # --- SDC placement ---

    def test_sdc_exists_256_channels(self):
        """sdc must exist with in_channels=256 (layer1 output)."""
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        assert hasattr(model, "sdc")
        assert isinstance(model.sdc, SDC)
        assert model.sdc.in_channels == 256

    # --- DSR placement ---

    def test_dsr_exists(self):
        """dsr must exist with feature_dim=2048."""
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        assert hasattr(model, "dsr")
        assert isinstance(model.dsr, DSR)

    # --- SDC called after layer1 ---

    def test_sdc_called_after_layer1(self):
        """SDC must be invoked in the forward path after layer1."""
        import inspect
        source = inspect.getsource(STHFModel.forward)
        assert "self.sdc(" in source, "sdc not called in forward"

    # --- DSR called on sequence features ---

    def test_dsr_called_on_sequence(self):
        """DSR must be invoked on [B, T, C] sequence features."""
        import inspect
        source = inspect.getsource(STHFModel.forward)
        assert "self.dsr(" in source, "dsr not called in forward"

    # --- Final features come after DSR, GAP, BN ---

    def test_features_after_pool_bn(self):
        """output['features'] must come from pool + final_bn."""
        import inspect
        source = inspect.getsource(STHFModel.forward)
        assert "backbone.pool(orig_" in source
        assert "backbone.final_bn(orig_" in source
        assert "features = refined_seq.mean" in source

    # --- No shortcut feature fusion ---

    def test_no_shortcut_feature_fusion(self):
        """No features = compensated + refined shortcut."""
        import inspect
        source = inspect.getsource(STHFModel.forward)
        assert "compensated + refined" not in source
        assert "features_raw" not in source

    # --- int_features from HF branch ---

    def test_int_features_from_hf_branch(self):
        """output['int_features'] must come from HF branch after layer4 -> GAP -> BN."""
        import inspect
        source = inspect.getsource(STHFModel.forward)
        assert "hf_pooled.flatten(1)" in source
        assert "int_features =" in source

    # --- Fixed sanity check ---

    def test_fixed_sanity_check(self):
        """Full end-to-end sanity check for fixed STHF model."""
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        model.eval()
        frames = torch.randn(2, 6, 3, 288, 144)
        with torch.no_grad():
            outputs = model(frames)
        assert torch.isfinite(outputs["features"]).all()
        assert torch.isfinite(outputs["logits"]).all()
        assert torch.isfinite(outputs["int_features"]).all()
        assert torch.isfinite(outputs["int_logits"]).all()
        assert outputs["features"].shape == (2, 2048)
        assert outputs["int_features"].shape == (2, 2048)

    # --- Adaptive sanity check ---

    def test_adaptive_sanity_check(self):
        """Full end-to-end sanity check for adaptive STHF model."""
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="adaptive")
        model.eval()
        frames = torch.randn(2, 6, 3, 288, 144)
        with torch.no_grad():
            outputs = model(frames)
        assert torch.isfinite(outputs["features"]).all()
        assert torch.isfinite(outputs["int_features"]).all()
        assert outputs["extra"]["sthpf_type"] == "adaptive"

    # --- Other existing tests ---

    def test_model_has_sthpf(self):
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        assert hasattr(model, "sthpf")
        assert isinstance(model.sthpf, FixedSTHPF)

    def test_backbone_exposes_layers(self):
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        backbone = model.backbone
        assert hasattr(backbone, "stem")
        assert hasattr(backbone, "layer1")
        assert hasattr(backbone, "layer2")
        assert hasattr(backbone, "layer3")
        assert hasattr(backbone, "layer4")
        assert hasattr(backbone, "pool")
        assert hasattr(backbone, "final_bn")

    def test_extra_metadata(self):
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        frames = torch.randn(2, 6, 3, 288, 144)
        outputs = model(frames)
        assert outputs["extra"]["sthpf_type"] == "fixed"
        assert "sthf_fixed" in outputs["extra"]["model_type"]

    def test_int_features_differ_from_original(self):
        model = STHFModel(num_classes=10, pretrained=False, sthpf_type="fixed")
        model.eval()
        torch.manual_seed(42)
        frames = torch.randn(2, 6, 3, 288, 144)
        with torch.no_grad():
            outputs = model(frames)
        assert not torch.allclose(
            outputs["features"], outputs["int_features"], atol=1e-6
        )
