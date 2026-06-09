import torch


def collate_video_fn(batch):
    """Collate a list of dataset samples into a batched dictionary.

    Args:
        batch (list[dict]): A list of samples, each produced by a
            :class:`BaseVideoDataset`. Each sample dict contains:
            - "frames": Tensor (T, C, H, W)
            - "pid": int
            - "camid": int
            - "modality": str
            - "track_id": int

    Returns:
        dict: A batched dictionary with the following keys:

        - frames (torch.Tensor): Stacked frames tensor of shape
          (B, T, C, H, W), where B is the batch size.
        - pids (torch.Tensor): Tensor of shape (B,) containing
          identity IDs.
        - camids (torch.Tensor): Tensor of shape (B,) containing
          camera IDs.
        - modalities (list[str]): List of length B with modality
          strings ("rgb" or "ir").
        - track_ids (torch.Tensor): Tensor of shape (B,) containing
          tracklet IDs.
    """
    frames = torch.stack([sample["frames"] for sample in batch], dim=0)
    pids = torch.tensor([sample["pid"] for sample in batch], dtype=torch.long)
    camids = torch.tensor([sample["camid"] for sample in batch], dtype=torch.long)
    modalities = [sample["modality"] for sample in batch]
    track_ids = torch.tensor(
        [sample["track_id"] for sample in batch], dtype=torch.long
    )

    # TODO: Add future preprocessing steps here (e.g. image normalization,
    #       data augmentation, modality-specific transforms, etc.).

    return {
        "frames": frames,
        "pids": pids,
        "camids": camids,
        "modalities": modalities,
        "track_ids": track_ids,
    }
