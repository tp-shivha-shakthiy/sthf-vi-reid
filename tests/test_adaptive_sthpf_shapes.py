import torch

from models.adaptive_sthpf import AdaptiveSTHPF


class TestAdaptiveSTHPF:
    def test_branch_output_shapes_5d(self):
        model = AdaptiveSTHPF()
        x = torch.randn(2, 6, 3, 288, 144)

        weak_out = model.weak_branch(x)
        paper_out = model.paper_branch(x)
        strong_out = model.strong_branch(x)

        assert weak_out.shape == x.shape
        assert paper_out.shape == x.shape
        assert strong_out.shape == x.shape

    def test_gate_output_shape(self):
        model = AdaptiveSTHPF()
        x = torch.randn(4, 6, 3, 288, 144)
        _ = model(x)
        w = model.latest_gate_weights
        assert w.shape == (4, 3)

    def test_softmax_constraint(self):
        model = AdaptiveSTHPF()
        x = torch.randn(4, 6, 3, 288, 144)
        _ = model(x)
        w = model.latest_gate_weights
        assert torch.allclose(w.sum(dim=1), torch.ones(4), atol=1e-6)

    def test_adaptive_output_shape_5d(self):
        model = AdaptiveSTHPF()
        x = torch.randn(2, 6, 3, 288, 144)
        out = model(x)
        assert out.shape == x.shape

    def test_multi_branch_routing(self):
        model = AdaptiveSTHPF()
        x = torch.randn(2, 6, 3, 288, 144)
        out = model(x)
        assert torch.isfinite(out).all()
        assert model.latest_gate_weights is not None

    def test_gate_is_not_placeholder(self):
        model = AdaptiveSTHPF()
        assert model.gate is not None
        assert hasattr(model.gate, "parameters")

    def test_all_branches_have_correct_cutoffs(self):
        model = AdaptiveSTHPF()
        assert model.weak_branch.fs == 5
        assert model.weak_branch.ft == 1
        assert model.paper_branch.fs == 10
        assert model.paper_branch.ft == 2
        assert model.strong_branch.fs == 15
        assert model.strong_branch.ft == 2

    def test_fusion_differs_from_equal_average(self):
        model = AdaptiveSTHPF()
        model.eval()
        x = torch.randn(2, 6, 3, 288, 144)
        with torch.no_grad():
            weak = model.weak_branch(x)
            paper = model.paper_branch(x)
            strong = model.strong_branch(x)
            avg = (weak + paper + strong) / 3.0
            adaptive = model(x)
        assert not torch.allclose(adaptive, avg, atol=1e-6), (
            "Adaptive fusion should differ from simple average"
        )
