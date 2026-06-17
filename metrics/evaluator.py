import os
import json
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
        """Compute pairwise distance matrix."""
        if self.metric == "cosine":
            q = query_features / query_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
            g = gallery_features / gallery_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
            return 1.0 - torch.mm(q, g.t())
        else:
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

    def evaluate_direction(
        self,
        query_features,
        query_pids,
        query_camids,
        gallery_features,
        gallery_pids,
        gallery_camids,
        topk=(1, 5, 10, 20),
    ):
        return self.evaluate(
            query_features, query_pids, query_camids,
            gallery_features, gallery_pids, gallery_camids,
            topk=topk,
        )

    @staticmethod
    def _split_by_modality(features, pids, camids, modalities, query_modality, gallery_modality):
        q_mask = [m == query_modality for m in modalities]
        g_mask = [m == gallery_modality for m in modalities]
        q_idx = [i for i, m in enumerate(q_mask) if m]
        g_idx = [i for i, m in enumerate(g_mask) if m]
        return (
            features[q_idx], pids[q_idx], camids[q_idx],
            features[g_idx], pids[g_idx], camids[g_idx],
        )

    def evaluate_all_directions(
        self,
        features,
        pids,
        camids,
        modalities,
        topk=(1, 5, 10, 20),
    ):
        results = {}
        for direction, q_mod, g_mod in [
            ("ir_to_rgb", "ir", "rgb"),
            ("rgb_to_ir", "rgb", "ir"),
        ]:
            q_feat, q_pid, q_cam, g_feat, g_pid, g_cam = self._split_by_modality(
                features, pids, camids, modalities, q_mod, g_mod,
            )
            results[direction] = self.evaluate(
                q_feat, q_pid, q_cam, g_feat, g_pid, g_cam, topk=topk,
            )
        return results

    def save_metrics_json(self, results, experiment_name, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join("experiments", experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {path}")
        return path
