"""Vision-language tasks for multimodal training."""

from .coco_captions import COCOCaptions
from .vqav2 import VQAv2
from .textvqa import TextVQA

__all__ = [
    'COCOCaptions',
    'VQAv2',
    'TextVQA',
]
