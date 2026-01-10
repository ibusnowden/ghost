"""Image preprocessing transforms for vision encoders."""

import torch
from torchvision import transforms
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
from functools import lru_cache


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


@lru_cache(maxsize=8)
def get_vision_transforms(encoder_name="siglip_vit_l14", image_size=336, is_train=False):
    """
    Get preprocessing transforms for vision encoder (cached).

    Args:
        encoder_name: Vision encoder model name
        image_size: Target image resolution (224 or 336)
        is_train: Whether training (adds augmentation) or inference

    Returns:
        torchvision.transforms.Compose

    Note: Results are cached to avoid recreating transform pipelines.
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
    Preprocess a batch of images (sequential).

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


def batch_preprocess_images_parallel(
    images: List[Union[Image.Image, bytes, torch.Tensor]],
    encoder_name: str = "siglip_vit_l14",
    image_size: int = 336,
    is_train: bool = False,
    num_workers: int = 4
) -> torch.Tensor:
    """
    Preprocess a batch of images in parallel using ThreadPoolExecutor.

    This is 2-4x faster than sequential processing for large batches.

    Args:
        images: List of PIL.Image, bytes, or tensors
        encoder_name: Vision encoder model name
        image_size: Target image resolution
        is_train: Whether training mode
        num_workers: Number of parallel workers (default 4)

    Returns:
        tensor: [B, 3, H, W] batch of preprocessed images
    """
    if len(images) <= 2:
        # For small batches, sequential is faster (no thread overhead)
        return batch_preprocess_images(images, encoder_name, image_size, is_train)

    # Process images in parallel
    results = [None] * len(images)

    def process_single(idx_img):
        idx, img = idx_img
        return idx, preprocess_image(img, encoder_name, image_size, is_train)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single, (i, img)): i for i, img in enumerate(images)}
        for future in as_completed(futures):
            idx, tensor = future.result()
            results[idx] = tensor

    return torch.stack(results)


class ImagePreprocessor:
    """
    Reusable image preprocessor with optional parallel processing.

    Example:
        preprocessor = ImagePreprocessor("siglip_vit_l14", 336, num_workers=4)
        batch = preprocessor.process_batch(images)
        stats = preprocessor.stats()
    """

    def __init__(
        self,
        encoder_name: str = "siglip_vit_l14",
        image_size: int = 336,
        is_train: bool = False,
        num_workers: int = 4,
        parallel_threshold: int = 4
    ):
        """
        Args:
            encoder_name: Vision encoder model name
            image_size: Target image resolution
            is_train: Whether training mode
            num_workers: Number of parallel workers
            parallel_threshold: Minimum batch size to use parallel processing
        """
        self.encoder_name = encoder_name
        self.image_size = image_size
        self.is_train = is_train
        self.num_workers = num_workers
        self.parallel_threshold = parallel_threshold

        # Pre-fetch the transform (cached)
        self.transform = get_vision_transforms(encoder_name, image_size, is_train)

        # Stats tracking
        self.images_processed = 0
        self.batches_processed = 0
        self.parallel_batches = 0

    def process_single(self, image: Union[Image.Image, bytes, torch.Tensor]) -> torch.Tensor:
        """Process a single image."""
        self.images_processed += 1
        return preprocess_image(image, self.encoder_name, self.image_size, self.is_train)

    def process_batch(self, images: List[Union[Image.Image, bytes, torch.Tensor]]) -> torch.Tensor:
        """Process a batch of images, using parallel processing if beneficial."""
        self.batches_processed += 1
        self.images_processed += len(images)

        if len(images) >= self.parallel_threshold:
            self.parallel_batches += 1
            return batch_preprocess_images_parallel(
                images, self.encoder_name, self.image_size, self.is_train, self.num_workers
            )
        else:
            return batch_preprocess_images(
                images, self.encoder_name, self.image_size, self.is_train
            )

    def stats(self) -> dict:
        """Return preprocessing statistics."""
        return {
            "images_processed": self.images_processed,
            "batches_processed": self.batches_processed,
            "parallel_batches": self.parallel_batches,
            "parallel_ratio": self.parallel_batches / max(1, self.batches_processed),
            "encoder_name": self.encoder_name,
            "image_size": self.image_size,
        }
