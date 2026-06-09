import torch


def compute_cmc(
    distmat,
    query_pids,
    query_camids,
    gallery_pids,
    gallery_camids,
    topk=(1, 5, 10, 20),
):
    """Compute Cumulative Matching Characteristic (CMC) curves.

    For each query, gallery samples from the same camera are excluded
    (standard ReID protocol to avoid trivial same-camera matches).

    Args:
        distmat: FloatTensor of shape (num_query, num_gallery) with
            pairwise distances (smaller = more similar).
        query_pids: 1D LongTensor of shape (num_query,) with query IDs.
        query_camids: 1D LongTensor of shape (num_query,) with query
            camera IDs.
        gallery_pids: 1D LongTensor of shape (num_gallery,) with gallery
            IDs.
        gallery_camids: 1D LongTensor of shape (num_gallery,) with
            gallery camera IDs.
        topk: Tuple of rank cutoffs to report (e.g., 1, 5, 10, 20).

    Returns:
        dict {f"rank{k}": float} for each k in topk.
    """
    num_query = distmat.size(0)
    max_k = max(topk)
    device = distmat.device

    query_pids = query_pids.to(device)
    query_camids = query_camids.to(device)
    gallery_pids = gallery_pids.to(device)
    gallery_camids = gallery_camids.to(device)

    all_ranks = torch.zeros(num_query, dtype=torch.long, device=device)

    for q_idx in range(num_query):
        # Sort gallery indices by ascending distance
        sorted_idx = torch.argsort(distmat[q_idx])

        # Remove gallery samples from the same camera (junk)
        junk = (gallery_camids == query_camids[q_idx]) & (
            gallery_pids != query_pids[q_idx]
        )
        sorted_idx = sorted_idx[~junk[sorted_idx]]

        # Find rank of first matching gallery sample
        matches = gallery_pids[sorted_idx] == query_pids[q_idx]
        first_match = torch.where(matches)[0]

        if first_match.numel() > 0:
            all_ranks[q_idx] = first_match[0].item() + 1  # 1-indexed
        else:
            all_ranks[q_idx] = max_k + 1  # No match found

    results = {}
    for k in topk:
        results[f"rank{k}"] = (all_ranks <= k).float().mean().item() * 100.0

    return results


def compute_map(
    distmat,
    query_pids,
    query_camids,
    gallery_pids,
    gallery_camids,
):
    """Compute mean Average Precision (mAP).

    For each query, average precision is the area under the precision-
    recall curve across all ranked gallery samples. The final mAP is the
    mean over all queries.

    Gallery samples from the same camera as the query are excluded
    (standard ReID protocol).

    Args:
        distmat: FloatTensor of shape (num_query, num_gallery).
        query_pids: 1D LongTensor of shape (num_query,).
        query_camids: 1D LongTensor of shape (num_query,).
        gallery_pids: 1D LongTensor of shape (num_gallery,).
        gallery_camids: 1D LongTensor of shape (num_gallery,).

    Returns:
        float: mAP in percent (0-100).
    """
    num_query = distmat.size(0)
    device = distmat.device

    query_pids = query_pids.to(device)
    query_camids = query_camids.to(device)
    gallery_pids = gallery_pids.to(device)
    gallery_camids = gallery_camids.to(device)

    ap_sum = 0.0

    for q_idx in range(num_query):
        sorted_idx = torch.argsort(distmat[q_idx])

        # Remove same-camera junk from the ranking
        junk = (gallery_camids == query_camids[q_idx]) & (
            gallery_pids != query_pids[q_idx]
        )
        keep = ~junk
        sorted_idx = sorted_idx[keep[sorted_idx]]

        # Binary relevance vector
        matches = gallery_pids[sorted_idx] == query_pids[q_idx]
        num_relevant = matches.sum().item()

        if num_relevant == 0:
            continue

        # Precision at each relevant hit
        positions = torch.where(matches)[0] + 1  # 1-indexed
        precisions = torch.arange(1, num_relevant + 1, device=device).float() / positions.float()
        ap_sum += precisions.sum().item() / num_relevant

    return (ap_sum / num_query) * 100.0 if num_query > 0 else 0.0
