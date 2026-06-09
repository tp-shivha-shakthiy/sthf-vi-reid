import torchvision.transforms as T


def get_video_transforms(mode="train", img_size=(288, 144)):
    """Build the image transform pipeline.

    Args:
        mode: ``"train"`` or ``"test"`` (both use the same pipeline for
            this stage).
        img_size: (height, width) tuple to resize frames to.

    Returns:
        A ``torchvision.transforms.Compose`` instance with:
            Resize(img_size) -> ToTensor() -> Normalize(ImageNet stats)

    Augmentation (horizontal flip, colour jitter, random erasing) is
    intentionally disabled per the paper reproduction settings.
    """
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
