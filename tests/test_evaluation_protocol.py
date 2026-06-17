import os
import json
import tempfile

import torch
import pytest

from metrics.evaluator import Evaluator


def _make_feature_data(num_samples=20, num_ids=5, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    features = torch.randn(num_samples, 64, generator=g)
    pids = torch.randint(0, num_ids, (num_samples,), generator=g)
    camids = torch.randint(0, 3, (num_samples,), generator=g)
    track_ids = torch.randint(0, 10, (num_samples,), generator=g)
    return features, pids, camids, track_ids


def _make_modality_data(num_samples=20):
    return ["rgb"] * (num_samples // 2) + ["ir"] * (num_samples - num_samples // 2)


class TestProtocolSplit:
    def test_ir_to_rgb_split(self):
        from scripts.evaluate import _split_by_modality
        feats, pids, camids, _ = _make_feature_data(20, 5)
        mods = _make_modality_data(20)
        data = {"features": feats, "pids": pids, "camids": camids, "modalities": mods}

        q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, qc, gc, sc = \
            _split_by_modality(data, "ir_to_rgb")

        assert qc == 10
        assert gc == 10
        assert tuple(q_feat.shape) == (10, 64)
        assert tuple(g_feat.shape) == (10, 64)

    def test_rgb_to_ir_split(self):
        from scripts.evaluate import _split_by_modality
        feats, pids, camids, _ = _make_feature_data(20, 5)
        mods = _make_modality_data(20)
        data = {"features": feats, "pids": pids, "camids": camids, "modalities": mods}

        q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, qc, gc, sc = \
            _split_by_modality(data, "rgb_to_ir")

        assert qc == 10
        assert gc == 10
        assert tuple(q_feat.shape) == (10, 64)
        assert tuple(g_feat.shape) == (10, 64)

    def test_shared_identity_counted(self):
        from scripts.evaluate import _split_by_modality
        feats = torch.randn(12, 64)
        pids = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
        camids = torch.zeros(12, dtype=torch.long)
        mods = ["rgb"] * 6 + ["ir"] * 6
        data = {"features": feats, "pids": pids, "camids": camids, "modalities": mods}

        _, _, _, _, _, _, qc, gc, sc = _split_by_modality(data, "ir_to_rgb")
        assert qc == 6
        assert gc == 6
        assert sc >= 1

    def test_unknown_protocol_raises(self):
        from scripts.evaluate import _split_by_modality
        data = {"features": torch.randn(4, 64), "pids": torch.zeros(4, dtype=torch.long),
                "camids": torch.zeros(4, dtype=torch.long), "modalities": ["rgb"] * 4}
        with pytest.raises(ValueError, match="Unknown protocol"):
            _split_by_modality(data, "invalid")


class TestZeroSharedIdentities:
    def test_raises_error(self):
        from scripts.evaluate import _run_and_print
        evaluator = Evaluator(metric="cosine")
        q_feat = torch.randn(5, 64)
        g_feat = torch.randn(5, 64)
        q_pids = torch.tensor([99, 98, 97, 96, 95])
        g_pids = torch.tensor([0, 1, 2, 3, 4])
        q_camids = torch.zeros(5, dtype=torch.long)
        g_camids = torch.ones(5, dtype=torch.long)

        with pytest.raises(ValueError, match="Zero shared identities"):
            _run_and_print(evaluator, q_feat, q_pids, q_camids, g_feat, g_pids, g_camids,
                           5, 5, 0, "test", "cosine")


class TestMetricsJSON:
    def test_save_and_load(self):
        from scripts.evaluate import _save_metrics_json
        results = {"rank1": 85.0, "rank5": 95.0, "rank10": 98.0, "rank20": 99.0, "mAP": 72.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_metrics_json(results, "ir_to_rgb", 10, 50, 5,
                               "config.yaml", "checkpoint.pth", tmpdir)
            path = os.path.join(tmpdir, "metrics.json")
            assert os.path.isfile(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["protocol"] == "ir_to_rgb"
            assert loaded["rank1"] == 85.0
            assert loaded["mAP"] == 72.5
            assert loaded["query_count"] == 10
            assert loaded["gallery_count"] == 50
            assert loaded["shared_identity_count"] == 5
            assert "config" in loaded
            assert "checkpoint" in loaded

    def test_all_search_save(self):
        from scripts.evaluate import _save_metrics_json
        results = {
            "ir_to_rgb": {"rank1": 80.0, "rank5": 90.0, "mAP": 65.0},
            "rgb_to_ir": {"rank1": 75.0, "rank5": 88.0, "mAP": 60.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_metrics_json(results, "all_search", 10, 50, 5,
                               None, None, tmpdir)
            path = os.path.join(tmpdir, "metrics.json")
            assert os.path.isfile(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["protocol"] == "all_search"
            assert loaded["ir_to_rgb"]["rank1"] == 80.0
            assert loaded["rgb_to_ir"]["rank1"] == 75.0


class TestFeatureExtractionCompat:
    def test_model_build_and_forward(self):
        from scripts.extract_features import build_model
        config = {
            "model": {"name": "baseline", "pretrained": False, "feature_dim": 2048},
        }
        model = build_model(config, num_classes=10)
        frames = torch.randn(2, 6, 3, 288, 144)
        outputs = model(frames)
        assert outputs["features"].shape == (2, 2048)

    def test_features_dictionary_keys(self):
        from scripts.extract_features import build_model
        config = {
            "model": {"name": "baseline", "pretrained": False, "feature_dim": 2048},
        }
        model = build_model(config, num_classes=10)
        frames = torch.randn(2, 6, 3, 288, 144)
        outputs = model(frames)
        assert set(outputs.keys()) == {"features", "logits", "int_features", "int_logits", "extra"}


class TestVisualizationSmoke:
    def test_smoke_no_dataset(self):
        from scripts.visualize_sthpf import _make_synthetic_batch
        batch = _make_synthetic_batch(seq_len=6, height=288, width=144)
        assert batch.shape == (1, 6, 3, 288, 144)
        assert torch.isfinite(batch).all()

        from models.sthpf import FixedSTHPF
        sthpf = FixedSTHPF(fs=10, ft=2)
        with torch.no_grad():
            high_freq = sthpf(batch)
        assert high_freq.shape == batch.shape
        assert torch.isfinite(high_freq).all()

    def test_figure_saved(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from models.sthpf import FixedSTHPF
        from scripts.visualize_sthpf import _make_synthetic_batch, _tensor_to_img

        batch = _make_synthetic_batch(seq_len=6, height=64, width=32)
        sthpf = FixedSTHPF(fs=5, ft=1)
        with torch.no_grad():
            high_freq = sthpf(batch)

        frame_idx = 0
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        axes[0].imshow(_tensor_to_img(batch[0, frame_idx]))
        axes[1].imshow(_tensor_to_img(high_freq[0, frame_idx]))
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_vis.png")
            plt.savefig(path, dpi=50)
            plt.close(fig)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
