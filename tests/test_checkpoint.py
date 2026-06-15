import os
import tempfile
import torch
from engine.checkpoint import save_checkpoint, load_checkpoint, CheckpointManager


def test_save_checkpoint_creates_file():
    state = {"epoch": 1, "loss": 0.5, "model_state": {"weight": torch.tensor([1.0])}}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pt")
        save_checkpoint(state, path)
        assert os.path.isfile(path)


def test_save_checkpoint_creates_parent_dirs():
    state = {"epoch": 1}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "nested", "test.pt")
        save_checkpoint(state, path)
        assert os.path.isfile(path)


def test_load_checkpoint_restores_state():
    state = {"epoch": 5, "loss": 0.3, "model_state": {"weight": torch.tensor([2.0])}}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pt")
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == 5
        assert loaded["loss"] == 0.3
        assert torch.equal(loaded["model_state"]["weight"], torch.tensor([2.0]))


def test_checkpoint_manager_save_and_load():
    state = {"epoch": 3, "some_tensor": torch.randn(4)}
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(state, "last.pth")
        loaded = mgr.load("last.pth")
        assert loaded["epoch"] == 3
        assert torch.equal(loaded["some_tensor"], state["some_tensor"])
