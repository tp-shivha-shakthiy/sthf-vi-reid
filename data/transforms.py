import torchvision.transforms as T

from data.caj import ChannelAugmentedJoint


def get_video_transforms(mode="train", img_size=(288, 144)):
    """Build the image transform pipeline per paper settings.

    Args:
        mode: ``"train"`` or ``"test"``.
        img_size: (height, width) tuple to resize frames to.

    Returns:
        A ``torchvision.transforms.Compose`` instance.

    Paper augmentations (training):
        - Random resized crop
        - Random horizontal flip
        - Channel-Augmented Joint Learning (CAJ)
        - Normalize with ImageNet stats
    """
    if mode == "train":
        return T.Compose([
            T.Resize(img_size),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


def build_caj(caj_cfg=None):
    """Build CAJ augmentation module from config.

    Args:
        caj_cfg: dict with keys 'enabled' (bool), 'alpha' (float), 'prob' (float).
            If None or enabled=False, returns None.

    Returns:
        ChannelAugmentedJoint module or None.
    """
    if caj_cfg is None:
        return None
    if not caj_cfg.get("enabled", False):
        return None
    return ChannelAugmentedJoint(
        alpha=caj_cfg.get("alpha", 1.0),
        prob=caj_cfg.get("prob", 0.5),
    )
