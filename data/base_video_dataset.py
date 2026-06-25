from torch.utils.data import Dataset
class BaseVideoDataset(Dataset):
    """Base class for video-based person ReID datasets.

    All dataset implementations must follow this sample contract:

    Returns:
        dict: A dictionary with the following keys:

        - frames (torch.Tensor): Video clip tensor of shape (T, C, H, W),
          where T is the sequence length, C is the number of channels,
          H is the height, and W is the width.
        - pid (int): Identity ID (person label).
        - camid (int): Camera ID the clip was captured from.
        - modality (str): Modality of the clip, either "rgb" or "ir".
        - track_id (int): Tracklet ID identifying the specific video
          track this clip was sampled from.
    """

    def __init__(self, root, seq_len=6, transform=None):
        self.root = root
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
