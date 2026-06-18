import torch

from losses.id_loss import IDLoss
from losses.triplet_loss import TripletLoss
from losses.build_loss import ReIDLossBuilder, build_loss


def test_id_loss_returns_finite_scalar():
    loss_fn = IDLoss()
    logits = torch.randn(4, 10)
    pids = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, pids)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_triplet_loss_returns_finite_scalar():
    loss_fn = TripletLoss()
    features = torch.randn(4, 2048)
    pids = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(features, pids)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_combined_loss_keys():
    criterion = ReIDLossBuilder()
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": None,
        "int_logits": None,
        "extra": {"model_type": "baseline"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    expected_keys = {"loss_total", "loss_id", "loss_tri", "loss_id_int", "loss_tri_int"}
    assert set(losses.keys()) == expected_keys


def test_baseline_int_losses_zero():
    criterion = ReIDLossBuilder()
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": None,
        "int_logits": None,
        "extra": {"model_type": "baseline"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    assert losses["loss_id_int"].item() == 0.0
    assert losses["loss_tri_int"].item() == 0.0


def test_total_loss_finite():
    criterion = ReIDLossBuilder()
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": None,
        "int_logits": None,
        "extra": {"model_type": "baseline"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    assert torch.isfinite(losses["loss_total"])
