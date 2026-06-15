import os
import torch


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def save(self, state, filename):
        path = os.path.join(self.save_dir, filename)
        save_checkpoint(state, path)

    def load(self, filename):
        path = os.path.join(self.save_dir, filename)
        return load_checkpoint(path)
