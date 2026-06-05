import torch
import torch.nn as nn


class ReIDLossBuilder(nn.Module):
    """
    Loss wrapper for baseline and STHF models.

    Expected model outputs:
        outputs["logits"]
        outputs["features"]
        outputs["int_logits"]
        outputs["int_features"]

    Expected labels:
        pids: [B]
    """

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

        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, pids):
        device = pids.device

        id_loss = self.ce(outputs["logits"], pids)

        # Temporary placeholder for triplet loss.
        # Real triplet loss will be implemented later in triplet_loss.py.
        triplet_loss = torch.zeros((), device=device)

        int_id_loss = torch.zeros((), device=device)
        int_triplet_loss = torch.zeros((), device=device)

        if outputs.get("int_logits") is not None:
            int_id_loss = self.ce(outputs["int_logits"], pids)

        total = (
            self.lambda_id * id_loss
            + self.lambda_tri * triplet_loss
            + self.lambda_int_id * int_id_loss
            + self.lambda_int_tri * int_triplet_loss
        )

        return {
            "total": total,
            "id_loss": id_loss.detach(),
            "triplet_loss": triplet_loss.detach(),
            "int_id_loss": int_id_loss.detach(),
            "int_triplet_loss": int_triplet_loss.detach(),
        }
