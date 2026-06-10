"""Structural tests for dataset shape correctness and metadata integrity."""

import torch
import pytest


def _make_dummy_sample(seq_len=6, img_size=(288, 144)):
    """Create a synthetic sample mimicking the BaseVideoDataset contract."""
    c, h, w = 3, img_size[0], img_size[1]
    return {
        "frames": torch.randn(seq_len, c, h, w),
        "pid": 1,
        "camid": 2,
        "modality": "rgb",
        "track_id": 0,
    }


class TestSampleLevelShapes:
    """Verify that individual dataset samples match the expected contract."""

    def test_sample_contract_shape(self):
        sample = _make_dummy_sample(seq_len=6, img_size=(288, 144))
        frames = sample["frames"]
        assert frames.ndim == 4, f"Expected 4D [T, C, H, W], got {frames.ndim}D"
        t, c, h, w = frames.shape
        assert t == 6, f"Expected T=6, got {t}"
        assert c == 3, f"Expected C=3, got {c}"
        assert (h, w) == (288, 144), f"Expected HxW=(288, 144), got {(h, w)}"

    def test_sample_contract_metadata(self):
        sample = _make_dummy_sample()
        assert "pid" in sample, "Missing pid key"
        assert isinstance(sample["pid"], int), "pid must be int"
        assert "camid" in sample, "Missing camid key"
        assert isinstance(sample["camid"], int), "camid must be int"
        assert "modality" in sample, "Missing modality key"
        assert sample["modality"] in ("rgb", "ir"), "modality must be 'rgb' or 'ir'"
        assert "track_id" in sample, "Missing track_id key"
        assert isinstance(sample["track_id"], int), "track_id must be int"

    def test_sample_modality_values(self):
        for mod in ("rgb", "ir"):
            sample = _make_dummy_sample()
            sample["modality"] = mod
            assert sample["modality"] in ("rgb", "ir")

    def test_sample_tensor_type(self):
        sample = _make_dummy_sample()
        assert isinstance(sample["frames"], torch.Tensor), "frames must be torch.Tensor"


class TestBatchLevelShapes:
    """Verify that collated batches match the expected training format."""

    def test_batch_contract_shape(self):
        from data.collate import collate_video_fn

        batch_size = 4
        samples = [_make_dummy_sample(seq_len=6, img_size=(288, 144))
                   for _ in range(batch_size)]
        batch = collate_video_fn(samples)

        frames = batch["frames"]
        assert frames.ndim == 5, f"Expected 5D [B, T, C, H, W], got {frames.ndim}D"
        b, t, c, h, w = frames.shape
        assert b == batch_size, f"Expected B={batch_size}, got {b}"
        assert t == 6, f"Expected T=6, got {t}"
        assert c == 3, f"Expected C=3, got {c}"
        assert (h, w) == (288, 144), f"Expected HxW=(288, 144), got {(h, w)}"

    def test_batch_contract_metadata(self):
        from data.collate import collate_video_fn

        samples = [_make_dummy_sample() for _ in range(4)]
        batch = collate_video_fn(samples)

        assert "pids" in batch, "Missing pids key"
        assert isinstance(batch["pids"], torch.Tensor), "pids must be Tensor"
        assert batch["pids"].shape == (4,), f"pids shape {batch['pids'].shape} != (4,)"

        assert "camids" in batch, "Missing camids key"
        assert isinstance(batch["camids"], torch.Tensor), "camids must be Tensor"
        assert batch["camids"].shape == (4,), f"camids shape {batch['camids'].shape} != (4,)"

        assert "modalities" in batch, "Missing modalities key"
        assert len(batch["modalities"]) == 4, "modalities length != batch_size"

        assert "track_ids" in batch, "Missing track_ids key"
        assert isinstance(batch["track_ids"], torch.Tensor), "track_ids must be Tensor"

    def test_batch_modality_values(self):
        from data.collate import collate_video_fn

        samples = [_make_dummy_sample() for _ in range(4)]
        samples[0]["modality"] = "rgb"
        samples[1]["modality"] = "ir"
        samples[2]["modality"] = "rgb"
        samples[3]["modality"] = "ir"
        batch = collate_video_fn(samples)
        for mod in batch["modalities"]:
            assert mod in ("rgb", "ir"), f"Invalid modality: {mod}"

    def test_batch_shapes_different_sizes(self):
        from data.collate import collate_video_fn

        samples = [_make_dummy_sample(seq_len=6, img_size=(288, 144))
                   for _ in range(2)]
        batch = collate_video_fn(samples)
        assert batch["frames"].shape == (2, 6, 3, 288, 144)


class TestVideoSampler:
    """Verify identity-balanced VideoSampler produces correct batch sizes."""

    def test_sampler_produces_correct_batch_size(self):
        from data.video_sampler import VideoSampler

        class FakeDataset:
            def __init__(self):
                self.samples = [
                    {"pid": 0}, {"pid": 0}, {"pid": 0}, {"pid": 0},
                    {"pid": 1}, {"pid": 1}, {"pid": 1}, {"pid": 1},
                    {"pid": 2}, {"pid": 2}, {"pid": 2}, {"pid": 2},
                    {"pid": 3}, {"pid": 3}, {"pid": 3}, {"pid": 3},
                ]

        dataset = FakeDataset()
        sampler = VideoSampler(dataset, num_ids=4, clips_per_id=2, shuffle=False)
        batches = list(sampler)
        for batch in batches:
            assert len(batch) == 8, f"Expected batch size 8, got {len(batch)}"

    def test_sampler_batch_has_all_unique_pids(self):
        from data.video_sampler import VideoSampler

        class FakeDataset:
            def __init__(self):
                self.samples = [
                    {"pid": 0}, {"pid": 0}, {"pid": 0},
                    {"pid": 1}, {"pid": 1}, {"pid": 1},
                    {"pid": 2}, {"pid": 2}, {"pid": 2},
                    {"pid": 3}, {"pid": 3}, {"pid": 3},
                ]

        dataset = FakeDataset()
        sampler = VideoSampler(dataset, num_ids=4, clips_per_id=1, shuffle=False)
        batches = list(sampler)
        for batch in batches:
            pids = [dataset.samples[i]["pid"] for i in batch]
            assert len(set(pids)) == 4, f"Expected 4 unique PIDs, got {set(pids)}"

    def test_sampler_len_matches_expected_batches(self):
        from data.video_sampler import VideoSampler

        class FakeDataset:
            def __init__(self):
                self.samples = [{"pid": i // 4} for i in range(16)]

        dataset = FakeDataset()
        sampler = VideoSampler(dataset, num_ids=4, clips_per_id=4, shuffle=False)
        expected = 16 // (4 * 4)
        assert len(sampler) == max(1, expected)
