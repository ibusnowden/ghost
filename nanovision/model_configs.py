"""
GhostVis model configuration.
Single scalable architecture with SwiGLU + GQA + fused kernels.
"""

from nanochat.gpt import GPTConfig


def get_config(
    depth: int = 32,
    vocab_size: int = 65536,
    sequence_len: int = 2048,
) -> GPTConfig:
    """
    Scalable model configuration with modern architecture.

    Architecture features:
    - SwiGLU activation (better than ReLU²)
    - Grouped-Query Attention (GQA) with 2:1 ratio
    - RoPE positional encoding
    - Attention bias (Qwen-style)
    - Fused kernels support (FlashAttention, FusedRMSNorm, FusedSwiGLU)

    Args:
        depth: Number of transformer layers (scales model size)
        vocab_size: Vocabulary size
        sequence_len: Maximum sequence length

    Returns:
        GPTConfig with SwiGLU + GQA architecture

    Examples:
        depth=20 -> ~500M params
        depth=32 -> ~1.5B params
        depth=48 -> ~3B params
    """
    model_dim = depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim ~128
    num_kv_heads = max(1, num_heads // 2)  # 2:1 GQA ratio
    intermediate_size = int(model_dim * 2.7)  # SwiGLU intermediate

    return GPTConfig(
        vocab_size=vocab_size,
        sequence_len=sequence_len,
        n_layer=depth,
        n_embd=model_dim,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        intermediate_size=intermediate_size,
        mlp_type="swiglu",
        rope_theta=10000.0,
        attention_bias=True,
        use_flash_attn=True,
        use_fused_kernels=True,
        use_fused_loss=True,
    )


def get_vlm_config(
    depth: int = 32,
    vocab_size: int = 65536,
    sequence_len: int = 2048,
    vision_encoder: str = "siglip_vit_l14",
) -> GPTConfig:
    """
    Vision-Language Model configuration.

    Extends base config with vision modules:
    - Vision encoder: SigLIP ViT-L/14 (frozen)
    - Perceiver resampler: 256 patches -> 64 tokens
    - MLP projector: vision_dim -> llm_dim

    Args:
        depth: Number of transformer layers
        vocab_size: Vocabulary size
        sequence_len: Maximum sequence length
        vision_encoder: Vision encoder name ("siglip_vit_l14" or "clip_vit_l14")

    Returns:
        GPTConfig with vision modules
    """
    config = get_config(depth=depth, vocab_size=vocab_size, sequence_len=sequence_len)

    # Vision configuration
    config.vision_encoder_name = vision_encoder
    config.vision_encoder_trainable = False  # Frozen
    config.vision_image_size = 336
    config.vision_num_tokens = 64
    config.vision_resampler_mode = "perceiver"
    config.vision_resampler_depth = 2
    config.vision_resampler_heads = 8
    config.vision_proj_hidden = 2048
    config.vision_proj_dropout = 0.0

    return config
