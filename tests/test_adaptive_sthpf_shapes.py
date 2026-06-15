import torch

from models.adaptive_sthpf import AdaptiveSTHPF


class TestAdaptiveSTHPF:
    def test_branch_output_shapes(self):
        model = AdaptiveSTHPF()
        x = torch.randn(2, 6, 3, 288, 144)

        weak_out = model.weak_branch(x)
        paper_out = model.paper_branch(x)
        strong_out = model.strong_branch(x)

        assert weak_out.shape == x.shape
        assert paper_out.shape == x.shape
        assert strong_out.shape == x.shape

    def test_fusion_output_shape(self):
        model = AdaptiveSTHPF()
        x = torch.randn(2, 6, 3, 288, 144)
        output = model(x)
        assert output.shape == x.shape

    def test_multi_branch_routing(self):
        model = AdaptiveSTHPF()
        x = torch.randn(2, 6, 3, 288, 144)
        output = model(x)
        assert torch.isfinite(output).all()

    def test_gate_placeholder(self):
        model = AdaptiveSTHPF()
        assert model.gate is None

    def test_fusion_differs_from_single_branch(self):
        model = AdaptiveSTHPF()
        model.eval()
        x = torch.randn(2, 6, 3, 288, 144)

        with torch.no_grad():
            fused = model(x)
            weak = model.weak_branch(x)

        assert not torch.allclose(fused, weak, atol=1e-6), (
            "Fused output should differ from any single branch output"
        )

    def test_all_branches_have_correct_cutoffs(self):
        model = AdaptiveSTHPF()
        assert model.weak_branch.fs == 5
        assert model.weak_branch.ft == 1
        assert model.paper_branch.fs == 10
        assert model.paper_branch.ft == 2
        assert model.strong_branch.fs == 15
        assert model.strong_branch.ft == 2
