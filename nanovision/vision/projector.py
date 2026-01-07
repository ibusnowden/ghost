"""Vision-to-language projector."""

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """
    Project vision tokens to LLM hidden dimension.

    Architecture: 2-layer MLP with GELU activation
    vision_dim -> hidden_dim -> llm_dim
    """

    def __init__(self, vision_dim, llm_dim, hidden_dim=2048, dropout=0.0):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with small values for stable training."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, vision_tokens):
        """
        Project vision tokens to LLM dimension.

        Args:
            vision_tokens: [B, N, vision_dim]

        Returns:
            projected: [B, N, llm_dim]
        """
        return self.mlp(vision_tokens)
