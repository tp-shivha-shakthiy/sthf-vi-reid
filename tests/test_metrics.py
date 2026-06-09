"""Unit tests for ReID metric computation and the Evaluator wrapper."""

import torch
import pytest

from metrics.reid_metrics import compute_cmc, compute_map
from metrics.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_features(n, dim=64, seed=0):
    g = torch.Generator()
    g.manual_seed(seed + n)
    return torch.randn(n, dim, generator=g)


def _make_payload(
    num_query=10,
    num_gallery=50,
    num_ids=5,
    seed=42,
):
    g = torch.Generator()
    g.manual_seed(seed)

    query_feat = torch.randn(num_query, 64, generator=g)
    gallery_feat = torch.randn(num_gallery, 64, generator=g)
    query_pids = torch.randint(0, num_ids, (num_query,), generator=g)
    gallery_pids = torch.randint(0, num_ids, (num_gallery,), generator=g)
    query_camids = torch.randint(0, 3, (num_query,), generator=g)
    gallery_camids = torch.randint(0, 3, (num_gallery,), generator=g)
    return query_feat, query_pids, query_camids, gallery_feat, gallery_pids, gallery_camids


# ---------------------------------------------------------------------------
# Input Parsing
# ---------------------------------------------------------------------------

class TestInputParsing:
    def test_accepts_tensors(self):
        q, qp, qc, g, gp, gc = _make_payload(5, 20, 3)
        distmat = torch.cdist(q, g)
        cmc = compute_cmc(distmat, qp, qc, gp, gc, topk=(1,))
        assert isinstance(cmc, dict)
        m = compute_map(distmat, qp, qc, gp, gc)
        assert isinstance(m, float)

    def test_accepts_different_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        q, qp, qc, g, gp, gc = _make_payload(5, 20, 3)
        q = q.cuda(); g = g.cuda()
        qp = qp.cuda(); gp = gp.cuda()
        qc = qc.cuda(); gc = gc.cuda()
        distmat = torch.cdist(q, g)
        cmc = compute_cmc(distmat, qp, qc, gp, gc, topk=(1, 5))
        assert all(k in cmc for k in ("rank1", "rank5"))


# ---------------------------------------------------------------------------
# Distance / Similarity Matrix Construction
# ---------------------------------------------------------------------------

class TestDistanceMatrix:
    def test_evaluator_cosine(self):
        q, qp, qc, g, gp, gc = _make_payload(5, 20, 3)
        ev = Evaluator(metric="cosine")
        dist = ev._compute_distmat(q, g)
        assert dist.shape == (5, 20)
        assert (dist >= 0).all() and (dist <= 2).all()  # cosine dist in [0, 2]

    def test_evaluator_euclidean(self):
        q, qp, qc, g, gp, gc = _make_payload(5, 20, 3)
        ev = Evaluator(metric="euclidean")
        dist = ev._compute_distmat(q, g)
        assert dist.shape == (5, 20)
        assert (dist >= 0).all()

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            Evaluator(metric="invalid")


# ---------------------------------------------------------------------------
# CMC Computation
# ---------------------------------------------------------------------------

class TestCMC:
    def test_output_format(self):
        q, qp, qc, g, gp, gc = _make_payload(10, 50, 5)
        distmat = torch.cdist(q, g)
        result = compute_cmc(distmat, qp, qc, gp, gc, topk=(1, 5, 10, 20))
        assert all(k in result for k in ("rank1", "rank5", "rank10", "rank20"))
        for v in result.values():
            assert 0.0 <= v <= 100.0

    def test_perfect_match(self):
        # Each query has an identical copy in gallery (same PID, different cam)
        q_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        g_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        q_pids = torch.tensor([0, 1, 2])
        g_pids = torch.tensor([0, 1, 2, 3])
        q_camids = torch.tensor([0, 0, 0])
        g_camids = torch.tensor([1, 1, 1, 1])
        distmat = torch.cdist(q_feat, g_feat)
        result = compute_cmc(distmat, q_pids, q_camids, g_pids, g_camids, topk=(1,))
        # All queries should match at rank 1
        assert result["rank1"] == 100.0

    def test_no_match(self):
        q_feat = torch.randn(3, 16)
        g_feat = torch.randn(5, 16)
        q_pids = torch.tensor([99, 98, 97])
        g_pids = torch.tensor([0, 1, 2, 3, 4])
        q_camids = torch.zeros(3, dtype=torch.long)
        g_camids = torch.ones(5, dtype=torch.long)
        distmat = torch.cdist(q_feat, g_feat)
        result = compute_cmc(distmat, q_pids, q_camids, g_pids, g_camids, topk=(1, 5))
        assert result["rank1"] == 0.0

    def test_same_camera_exclusion(self):
        # Query and gallery have same PID but also same camera; that gallery
        # sample should be excluded (junk), so a different gallery match is used.
        q_feat = torch.tensor([[1.0, 0.0]])
        g_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
        q_pids = torch.tensor([0])
        g_pids = torch.tensor([0, 0, 0])
        q_camids = torch.tensor([0])
        g_camids = torch.tensor([0, 1, 2])
        distmat = torch.cdist(q_feat, g_feat)
        # Gallery[0] is same PID + same cam -> junk -> excluded
        # Closest remaining match should be gallery[2] (dist ~0.14)
        result = compute_cmc(distmat, q_pids, q_camids, g_pids, g_camids, topk=(1,))
        assert result["rank1"] == 100.0


# ---------------------------------------------------------------------------
# mAP Computation
# ---------------------------------------------------------------------------

class TestMAP:
    def test_output_value(self):
        q, qp, qc, g, gp, gc = _make_payload(10, 50, 5)
        distmat = torch.cdist(q, g)
        m = compute_map(distmat, qp, qc, gp, gc)
        assert 0.0 <= m <= 100.0

    def test_perfect_map(self):
        q_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        g_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        q_pids = torch.tensor([0, 1])
        g_pids = torch.tensor([0, 1])
        q_camids = torch.tensor([0, 0])
        g_camids = torch.tensor([1, 1])
        distmat = torch.cdist(q_feat, g_feat)
        m = compute_map(distmat, q_pids, q_camids, g_pids, g_camids)
        assert m == 100.0

    def test_zero_map(self):
        q_feat = torch.randn(3, 16)
        g_feat = torch.randn(5, 16)
        q_pids = torch.tensor([99, 98, 97])
        g_pids = torch.tensor([0, 1, 2, 3, 4])
        q_camids = torch.zeros(3, dtype=torch.long)
        g_camids = torch.ones(5, dtype=torch.long)
        distmat = torch.cdist(q_feat, g_feat)
        m = compute_map(distmat, q_pids, q_camids, g_pids, g_camids)
        assert m == 0.0


# ---------------------------------------------------------------------------
# Evaluator Wrapper
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_evaluate_returns_all_keys(self):
        q, qp, qc, g, gp, gc = _make_payload(10, 50, 5)
        ev = Evaluator()
        results = ev.evaluate(q, qp, qc, g, gp, gc)
        expected_keys = {"rank1", "rank5", "rank10", "rank20", "mAP"}
        assert set(results.keys()) == expected_keys

    def test_evaluate_values_in_range(self):
        q, qp, qc, g, gp, gc = _make_payload(10, 50, 5)
        ev = Evaluator()
        results = ev.evaluate(q, qp, qc, g, gp, gc)
        for v in results.values():
            assert 0.0 <= v <= 100.0

    def test_evaluate_euclidean(self):
        q, qp, qc, g, gp, gc = _make_payload(10, 50, 5)
        ev = Evaluator(metric="euclidean")
        results = ev.evaluate(q, qp, qc, g, gp, gc)
        assert set(results.keys()) == {"rank1", "rank5", "rank10", "rank20", "mAP"}


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_query(self):
        q_feat = torch.randn(1, 64)
        g_feat = torch.randn(20, 64)
        q_pids = torch.tensor([0])
        g_pids = torch.randint(0, 5, (20,))
        q_camids = torch.zeros(1, dtype=torch.long)
        g_camids = torch.randint(0, 3, (20,))
        distmat = torch.cdist(q_feat, g_feat)
        cmc = compute_cmc(distmat, q_pids, q_camids, g_pids, g_camids, topk=(1, 5))
        assert all(k in cmc for k in ("rank1", "rank5"))
        m = compute_map(distmat, q_pids, q_camids, g_pids, g_camids)
        assert 0.0 <= m <= 100.0

    def test_single_gallery(self):
        q_feat = torch.randn(5, 64)
        g_feat = torch.randn(1, 64)
        q_pids = torch.tensor([0, 1, 2, 3, 4])
        g_pids = torch.tensor([0])
        q_camids = torch.ones(5, dtype=torch.long)
        g_camids = torch.zeros(1, dtype=torch.long)
        distmat = torch.cdist(q_feat, g_feat)
        cmc = compute_cmc(distmat, q_pids, q_camids, g_pids, g_camids, topk=(1,))
        assert isinstance(cmc, dict)

    def test_multiple_matches(self):
        q_feat = torch.tensor([[1.0, 0.0]])
        g_feat = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]])
        q_pids = torch.tensor([0])
        g_pids = torch.tensor([0, 0, 0, 1])
        q_camids = torch.tensor([0])
        g_camids = torch.tensor([1, 1, 1, 1])
        distmat = torch.cdist(q_feat, g_feat)
        # All three PID-0 gallery items are relevant → perfect ranking
        cmc = compute_cmc(distmat, q_pids, q_camids, g_pids, g_camids, topk=(1,))
        assert cmc["rank1"] == 100.0
        m = compute_map(distmat, q_pids, q_camids, g_pids, g_camids)
        assert m == 100.0

    def test_repeated_identities(self):
        q_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        g_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        q_pids = torch.tensor([0, 0])
        g_pids = torch.tensor([0, 0, 1])
        q_camids = torch.tensor([0, 1])
        g_camids = torch.tensor([1, 0, 1])
        distmat = torch.cdist(q_feat, g_feat)
        results = compute_cmc(distmat, q_pids, q_camids, g_pids, g_camids, topk=(1, 5))
        assert all(0.0 <= v <= 100.0 for v in results.values())
