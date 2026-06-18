import torch
import torch.nn as nn

from .id_loss import IDLoss
from .triplet_loss import TripletLoss


class ReIDLossBuilder(nn.Module):
    def __init__(
        self,
        lambda_id: float = 1.0,
        lambda_tri: float = 1.0,
        lambda_int_id: float = 1.0,
        lambda_int_tri: float = 1.0,
    ):
        super().__init__()

        self.lambda_id = lambda_id
        self.lambda_tri = lambda_tri
        self.lambda_int_id = lambda_int_id
        self.lambda_int_tri = lambda_int_tri

        self.id_loss_fn = IDLoss()
        self.triplet_loss_fn = TripletLoss()

    def forward(self, outputs, pids):
        loss_id = self.id_loss_fn(outputs["logits"], pids)
        loss_tri = self.triplet_loss_fn(outputs["features"], pids)

        loss_id_int = loss_id * 0.0
        loss_tri_int = loss_tri * 0.0

        if outputs.get("int_logits") is not None:
            loss_id_int = self.id_loss_fn(outputs["int_logits"], pids)
        if outputs.get("int_features") is not None:
            loss_tri_int = self.triplet_loss_fn(outputs["int_features"], pids)

        loss_total = (
            self.lambda_id * loss_id
            + self.lambda_tri * loss_tri
            + self.lambda_int_id * loss_id_int
            + self.lambda_int_tri * loss_tri_int
        )

        return {
            "loss_total": loss_total,
            "loss_id": loss_id.detach(),
            "loss_tri": loss_tri.detach(),
            "loss_id_int": loss_id_int.detach(),
            "loss_tri_int": loss_tri_int.detach(),
        }


def build_loss(config):
    loss_cfg = config.get("loss", {})
    return ReIDLossBuilder(
        lambda_id=loss_cfg.get("lambda_id", 1.0),
        lambda_tri=loss_cfg.get("lambda_tri", 1.0),
        lambda_int_id=loss_cfg.get("lambda_int_id", 1.0),
        lambda_int_tri=loss_cfg.get("lambda_int_tri", 1.0),
    )
