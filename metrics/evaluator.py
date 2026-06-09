import torch

from .reid_metrics import compute_cmc, compute_map


class Evaluator:
    """Reusable, model-agnostic ReID evaluator.

    Operates exclusively on feature tensors and metadata — no model
    instances, dataloaders, or training code required.

    Args:
        metric: Distance function. One of ``"cosine"`` or ``"euclidean"``.
    """

    def __init__(self, metric="cosine"):
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric

    def _compute_distmat(self, query_features, gallery_features):
        """Compute pairwise distance matrix.

        Args:
            query_features: Tensor (num_query, feat_dim).
            gallery_features: Tensor (num_gallery, feat_dim).

        Returns:
            FloatTensor (num_query, num_gallery).
        """
        if self.metric == "cosine":
            q = query_features / query_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
            g = gallery_features / gallery_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
            return 1.0 - torch.mm(q, g.t())
        else:
            # Euclidean distance
            q_sq = query_features.pow(2).sum(dim=1, keepdim=True)
            g_sq = gallery_features.pow(2).sum(dim=1, keepdim=True)
            cross = torch.mm(query_features, gallery_features.t())
            return torch.sqrt(
                torch.clamp(q_sq + g_sq.t() - 2.0 * cross, min=1e-12)
            )

    def evaluate(
        self,
        query_features,
        query_pids,
        query_camids,
        gallery_features,
        gallery_pids,
        gallery_camids,
        topk=(1, 5, 10, 20),
    ):
        """Run full ReID evaluation.

        Args:
            query_features: Tensor (num_query, feat_dim).
            query_pids: 1D LongTensor (num_query,).
            query_camids: 1D LongTensor (num_query,).
            gallery_features: Tensor (num_gallery, feat_dim).
            gallery_pids: 1D LongTensor (num_gallery,).
            gallery_camids: 1D LongTensor (num_gallery,).
            topk: Rank cutoffs to report.

        Returns:
            dict with keys ``"rank1"``, ``"rank5"``, ``"rank10"``,
            ``"rank20"``, ``"mAP"``.
        """
        distmat = self._compute_distmat(query_features, gallery_features)
        cmc = compute_cmc(
            distmat, query_pids, query_camids,
            gallery_pids, gallery_camids, topk=topk,
        )
        mAP = compute_map(
            distmat, query_pids, query_camids,
            gallery_pids, gallery_camids,
        )
        return {**cmc, "mAP": mAP}
