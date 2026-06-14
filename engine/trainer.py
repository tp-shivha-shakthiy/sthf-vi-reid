import torch
from tqdm import tqdm

from engine.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

    def training_step(self, batch):
        frames = batch["frames"]
        pids = batch["pids"]

        self.optimizer.zero_grad()
        outputs = self.model(frames, modalities=batch.get("modalities"))
        losses = self.criterion(outputs, pids)
        total = losses["total"]

        if not torch.isfinite(total):
            raise RuntimeError(f"Loss is not finite: {total}")

        total.backward()
        self.optimizer.step()

        return losses

    def train_epoch(self, loader):
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        for batch in tqdm(loader, desc="Train", leave=False):
            losses = self.training_step(batch)
            for key, val in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + val.item()
            num_batches += 1

        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        return avg_losses

    def fit(self, train_loader, val_loader, epochs):
        model_name = self.cfg.get("model", {}).get("name", "run")
        save_dir = self.cfg.get("train", {}).get("save_dir", f"experiments/{model_name}")

        for epoch in range(1, epochs + 1):
            train_losses = self.train_epoch(train_loader)

            log_parts = [f"Epoch {epoch}/{epochs}"]
            for key, val in train_losses.items():
                log_parts.append(f"{key}: {val:.4f}")
            print(" | ".join(log_parts))

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": train_losses,
            }
            save_checkpoint(state, f"{save_dir}/checkpoint_last.pth")

            if self.scheduler is not None:
                self.scheduler.step()
