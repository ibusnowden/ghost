"""Vision resampler to reduce patch tokens to fixed count."""

import torch
import torch.nn as nn
import math


class VisionResampler(nn.Module):
    """
    Reduce variable number of vision patches to fixed token count.

    Modes:
    - avgpool: Simple spatial average pooling
    - perceiver: Learned query tokens with cross-attention (better quality)
    """

    def __init__(self, vision_dim, num_tokens=64, mode="perceiver", depth=2, heads=8):
        super().__init__()
        self.vision_dim = vision_dim
        self.num_tokens = num_tokens
        self.mode = mode

        if mode == "avgpool":
            # Simple adaptive average pooling
            self.pool = nn.AdaptiveAvgPool1d(num_tokens)

        elif mode == "perceiver":
            # Learned query tokens
            self.queries = nn.Parameter(torch.randn(1, num_tokens, vision_dim))

            # Perceiver cross-attention layers
            self.layers = nn.ModuleList([
                PerceiverBlock(vision_dim, heads=heads)
                for _ in range(depth)
            ])

            # Initialize queries
            nn.init.trunc_normal_(self.queries, std=0.02)

        else:
            raise ValueError(f"Unknown resampler mode: {mode}")

    def forward(self, vision_features):
        """
        Resample vision features to fixed token count.

        Args:
            vision_features: [B, P, D_v] where P is num_patches

        Returns:
            resampled: [B, num_tokens, D_v]
        """
        B, P, D = vision_features.shape

        if self.mode == "avgpool":
            # Simple pooling: [B, P, D] -> [B, D, P] -> pool -> [B, D, num_tokens] -> [B, num_tokens, D]
            x = vision_features.transpose(1, 2)  # [B, D, P]
            x = self.pool(x)  # [B, D, num_tokens]
            x = x.transpose(1, 2)  # [B, num_tokens, D]
            return x

        elif self.mode == "perceiver":
            # Expand learned queries for batch
            queries = self.queries.expand(B, -1, -1)  # [B, num_tokens, D]

            # Cross-attend to vision features
            x = queries
            for layer in self.layers:
                x = layer(x, vision_features)

            return x


class PerceiverBlock(nn.Module):
    """Single Perceiver cross-attention block."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, queries, context):
        """
        Args:
            queries: [B, N, D] learned query tokens
            context: [B, P, D] vision features to attend to

        Returns:
            output: [B, N, D] updated queries
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(
            self.norm1(queries),
            context,
            context
        )
        queries = queries + attn_out

        # MLP
        queries = queries + self.mlp(self.norm2(queries))

        return queries
