"""Vision encoder wrapper for pretrained vision towers."""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    Wrapper around pretrained vision encoders (SigLIP, CLIP).

    Supports:
    - siglip_vit_l14: SigLIP ViT-L/14 (1024 dim)
    - clip_vit_l14: CLIP ViT-L/14 (768 dim)
    """

    def __init__(self, model_name="siglip_vit_l14", trainable=False):
        super().__init__()
        self.model_name = model_name
        self.trainable = trainable

        # Load pretrained vision tower
        if "siglip" in model_name.lower():
            self.encoder, self.output_dim = self._load_siglip(model_name)
        elif "clip" in model_name.lower():
            self.encoder, self.output_dim = self._load_clip(model_name)
        else:
            raise ValueError(f"Unknown vision encoder: {model_name}")

        # Freeze if not trainable
        if not trainable:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def _load_siglip(self, model_name):
        """Load SigLIP vision tower from timm or HF."""
        try:
            import timm
            # SigLIP ViT-L/14 from timm
            encoder = timm.create_model(
                'vit_large_patch14_clip_224.openai',
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            output_dim = 1024  # ViT-L hidden dimension
            return encoder, output_dim
        except ImportError:
            # Fallback to HuggingFace
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained("google/siglip-large-patch14-224")
            output_dim = encoder.config.hidden_size
            return encoder.vision_model, output_dim

    def _load_clip(self, model_name):
        """Load CLIP vision tower from HuggingFace."""
        from transformers import CLIPVisionModel
        encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        output_dim = encoder.config.hidden_size  # 1024 for ViT-L
        return encoder, output_dim

    def forward(self, images):
        """
        Process images through vision encoder.

        Args:
            images: [B, 3, H, W] tensor (normalized, resized to 224x224 or 336x336)

        Returns:
            vision_features: [B, num_patches, output_dim]
        """
        if not self.trainable:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(images)
        else:
            outputs = self.encoder(images)

        # Extract patch embeddings (handle different output formats)
        if hasattr(outputs, 'last_hidden_state'):
            # HuggingFace format
            vision_features = outputs.last_hidden_state
        elif isinstance(outputs, torch.Tensor):
            # timm format
            vision_features = outputs
        else:
            raise ValueError(f"Unexpected encoder output type: {type(outputs)}")

        return vision_features

    def train(self, mode=True):
        """Override train to respect trainable flag."""
        if not self.trainable:
            # Keep in eval mode if frozen
            super().train(False)
        else:
            super().train(mode)
        return self
