"""Image preprocessing transforms for vision encoders."""

import torch
from torchvision import transforms
from PIL import Image
import io


# Vision encoder normalization stats
VISION_STATS = {
    "siglip_vit_l14": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "clip_vit_l14": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}


def get_vision_transforms(encoder_name="siglip_vit_l14", image_size=336, is_train=False):
    """
    Get preprocessing transforms for vision encoder.

    Args:
        encoder_name: Vision encoder model name
        image_size: Target image resolution (224 or 336)
        is_train: Whether training (adds augmentation) or inference

    Returns:
        torchvision.transforms.Compose
    """
    # Get normalization stats for this encoder
    stats = VISION_STATS.get(encoder_name, VISION_STATS["siglip_vit_l14"])

    if is_train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"]),
        ])
    else:
        # Inference transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"]),
        ])

    return transform


def preprocess_image(image, encoder_name="siglip_vit_l14", image_size=336, is_train=False):
    """
    Preprocess a single image for vision encoder.

    Args:
        image: PIL.Image, bytes, or torch.Tensor
        encoder_name: Vision encoder model name
        image_size: Target image resolution
        is_train: Whether training mode

    Returns:
        tensor: [3, H, W] preprocessed image tensor
    """
    # Convert to PIL if needed
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, torch.Tensor):
        # Already a tensor, assume preprocessed
        return image
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Apply transforms
    transform = get_vision_transforms(encoder_name, image_size, is_train)
    return transform(image)


def batch_preprocess_images(images, encoder_name="siglip_vit_l14", image_size=336, is_train=False):
    """
    Preprocess a batch of images.

    Args:
        images: List of PIL.Image, bytes, or tensors
        encoder_name: Vision encoder model name
        image_size: Target image resolution
        is_train: Whether training mode

    Returns:
        tensor: [B, 3, H, W] batch of preprocessed images
    """
    processed = []
    for img in images:
        processed.append(preprocess_image(img, encoder_name, image_size, is_train))

    return torch.stack(processed)
