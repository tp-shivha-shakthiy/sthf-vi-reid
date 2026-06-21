import time

import torch
from tqdm import tqdm

from engine.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, cfg, caj=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.caj = caj

    def training_step(self, batch):
        frames = batch["frames"]
        pids = batch["pids"]

        # Apply CAJ augmentation if available
        if self.caj is not None and self.model.training:
            frames = self.caj(frames, pids, batch.get("modalities", []))

        self.optimizer.zero_grad()
        outputs = self.model(frames, modalities=batch.get("modalities"))
        losses = self.criterion(outputs, pids)
        total = losses["loss_total"]

        if not torch.isfinite(total):
            raise RuntimeError(f"Loss is not finite: {total}")

        total.backward()
        self.optimizer.step()

        return losses

    def train_epoch(self, loader, max_batches=None):
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        use_cuda = torch.cuda.is_available()

        iterator = enumerate(tqdm(loader, desc="Train", leave=False))
        for batch_idx, batch in iterator:
            # Timing: data loading (already loaded by iterator, mark t0)
            t_data_end = time.perf_counter()

            # Timing: forward
            t_fwd_start = time.perf_counter()
            frames = batch["frames"]
            pids = batch["pids"]
            if self.caj is not None and self.model.training:
                frames = self.caj(frames, pids, batch.get("modalities", []))
            outputs = self.model(frames, modalities=batch.get("modalities"))
            t_fwd_end = time.perf_counter()

            # Timing: loss
            t_loss_start = time.perf_counter()
            losses = self.criterion(outputs, pids)
            total = losses["loss_total"]
            t_loss_end = time.perf_counter()

            if not torch.isfinite(total):
                raise RuntimeError(f"Loss is not finite: {total}")

            # Timing: backward
            t_bwd_start = time.perf_counter()
            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            t_bwd_end = time.perf_counter()

            # Accumulate losses
            for key, val in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + val.item()
            num_batches += 1

            # Print per-batch timing
            fwd_ms = (t_fwd_end - t_fwd_start) * 1000
            loss_ms = (t_loss_end - t_loss_start) * 1000
            bwd_ms = (t_bwd_end - t_bwd_start) * 1000
            total_ms = (t_bwd_end - t_fwd_start) * 1000

            gpu_mem = ""
            if use_cuda:
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                gpu_mem = f" | GPU {alloc:.0f}MB alloc / {reserved:.0f}MB reserved"

            print(
                f"  Batch {batch_idx} | "
                f"fwd {fwd_ms:.0f}ms | loss {loss_ms:.0f}ms | bwd {bwd_ms:.0f}ms | "
                f"total {total_ms:.0f}ms{gpu_mem}"
            )

            if max_batches is not None and batch_idx + 1 >= max_batches:
                print(f"  Reached max_batches={max_batches}, stopping early.")
                break

        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        return avg_losses

    def fit(self, train_loader, val_loader, epochs, max_batches=None, start_epoch=1):
        model_name = self.cfg.get("model", {}).get("name", "run")
        save_dir = self.cfg.get("train", {}).get("save_dir", f"experiments/{model_name}")

        for epoch in range(start_epoch, epochs + 1):
            train_losses = self.train_epoch(train_loader, max_batches=max_batches)

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
