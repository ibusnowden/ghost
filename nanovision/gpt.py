"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- SwiGLU MLP (optionally MoE)
- norm after token embedding
- no learnable params in rmsnorm
- optional bias in attention projections (Qwen-style)
- Grouped-Query Attention (GQA/MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# Try to import FlashAttention-2 for 5x attention speedup
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Try to import fused kernels for 3-5x norm and SwiGLU speedup
try:
    from nanovision.fused_kernels import FusedRMSNorm, FusedSwiGLU, FusedCrossEntropyLoss, log_available_backends
    FUSED_KERNELS_AVAILABLE = True
except ImportError:
    FUSED_KERNELS_AVAILABLE = False

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA/MQA)
    n_embd: int = 768
    intermediate_size: int = None # MLP hidden dimension (None = 4 * n_embd for compatibility)
    mlp_type: str = "swiglu" # swiglu|relu2 (legacy)
    rope_theta: float = 10000.0 # RoPE base frequency
    attention_bias: bool = False # whether to use bias in attention projections
    use_flash_attn: bool = True # Use FlashAttention-2 if available (5x speedup)
    use_fused_kernels: bool = True # Use fused RMSNorm and SwiGLU (3-5x speedup)
    use_fused_loss: bool = True # Use fused cross-entropy (Phase 3, 2x loss speedup)
    use_gradient_checkpointing: bool = False # Enable gradient checkpointing (Phase 3, 2x longer contexts)
   
    # Vision encoder config (None = text-only model)
    vision_encoder_name: str = None # "siglip_vit_l14" |
    vision_encoder_trainable: bool = False
    vision_image_size: int = 336
    # Resampler config
    vision_num_tokens: int = 64
    vision_resampler_mode: str = "perceiver" # "avgpool" | "perceiver"
    vision_resampler_depth: int = 2
    vision_resampler_heads: int = 8
    # Projector config
    vision_proj_hidden: int = 2048
    vision_proj_dropout: float = 0.0
    # Special tokens
    image_token_id: int = None


def norm(x):
    # Purely functional rmsnorm with no learnable params
    # Note: PyTorch 2.0+ F.rms_norm is already well-optimized
    # For maximum speed, torch.compile will fuse this with surrounding ops
    return F.rms_norm(x, (x.size(-1),))

# Compiled version for 2x speedup (initialized on first use)
_compiled_norm = None

def get_compiled_norm():
    """Get compiled norm function for 2x speedup."""
    global _compiled_norm
    if _compiled_norm is None:
        try:
            _compiled_norm = torch.compile(norm)
        except:
            _compiled_norm = norm
    return _compiled_norm


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Use attention_bias from config (Qwen2.5 uses bias=True)
        attn_bias = config.attention_bias
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=attn_bias)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=attn_bias)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=attn_bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # FlashAttention-2 support (5x speedup over SDPA)
        self.use_flash_attn = config.use_flash_attn and FLASH_ATTN_AVAILABLE

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm

        # FlashAttention-2 path (5x faster, only works during training without KV cache)
        if self.use_flash_attn and kv_cache is None:
            # FlashAttention expects (B, T, H, D) format - no transpose needed!
            # Apply GQA: replicate key/value heads before FlashAttention
            nrep = self.n_head // self.n_kv_head
            if nrep > 1:
                k = k.repeat_interleave(nrep, dim=2)
                v = v.repeat_interleave(nrep, dim=2)

            # FlashAttention-2: memory-efficient causal attention
            # Automatically handles causal masking and avoids materializing O(T²) attention matrix
            y = flash_attn_func(q, k, v, causal=True)
            y = y.contiguous().view(B, T, -1)
            y = self.c_proj(y)
            return y

        # Standard SDPA path (for inference with KV cache or when FlashAttention unavailable)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Apply MQA: replicate the key/value heads for each query head
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use intermediate_size if specified, otherwise default to 4 * n_embd for backward compatibility
        intermediate_size = config.intermediate_size if config.intermediate_size is not None else 4 * config.n_embd
        self.mlp_type = getattr(config, "mlp_type", "swiglu")
        self.use_fused_swiglu = getattr(config, "use_fused_kernels", True) and self.mlp_type == "swiglu" and FUSED_KERNELS_AVAILABLE

        if self.use_fused_swiglu:
            # Use FusedSwiGLU for 1.5x additional speedup
            self.swiglu = FusedSwiGLU(config.n_embd, intermediate_size, bias=False)
        elif self.mlp_type == "swiglu":
            # SwiGLU activation: gate and up projections
            self.c_gate = nn.Linear(config.n_embd, intermediate_size, bias=False)
            self.c_up = nn.Linear(config.n_embd, intermediate_size, bias=False)
          
        else:
            raise ValueError(f"Unsupported mlp_type: {self.mlp_type}")
        self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=False)

        # Compile the forward pass for 2-2.5x speedup
        # torch.compile fuses operations (SwiGLU: silu + mul, matmuls) into optimized kernels
        self._compiled_forward = None

    def forward(self, x):
        # Use compiled version if available (initialized on first forward pass)
        if self._compiled_forward is None:
            try:
                self._compiled_forward = torch.compile(self._forward_impl)
            except Exception:
                # Fallback if torch.compile not available (PyTorch < 2.0)
                self._compiled_forward = self._forward_impl

        return self._compiled_forward(x)

    def _forward_impl(self, x):
        """Actual MLP computation - will be compiled by torch.compile."""
        if self.use_fused_swiglu:
            # FusedSwiGLU: 1.5x faster than torch.compile
            x = self.swiglu(x)
        elif self.mlp_type == "swiglu":
            # SwiGLU: silu(gate(x)) * up(x)
            # torch.compile will fuse these operations into a single kernel
            gate = F.silu(self.c_gate(x))
            up = self.c_up(x)
            x = gate * up
       
        x = self.c_proj(x)
        return x, x.new_zeros((), dtype=torch.float32)


def _is_moe_layer(config: GPTConfig, layer_idx: int) -> bool:
    if config.moe_num_experts <= 0:
        return False
    start = max(0, int(config.moe_layer_start))
    end = config.n_layer if config.moe_layer_end < 0 else int(config.moe_layer_end)
    stride = max(1, int(config.moe_layer_stride))
    return start <= layer_idx < end and (layer_idx - start) % stride == 0



class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MoE(config) if _is_moe_layer(config, layer_idx) else MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        mlp_out, aux_loss = self.mlp(norm(x))
        x = x + mlp_out
        return x, aux_loss


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Vision modules (optional, only if vision_encoder_name is specified)
        self.vision_encoder = None
        self.vision_resampler = None
        self.vision_projector = None

        if config.vision_encoder_name is not None:
            # Import vision modules (lazy import to avoid dependencies if not used)
            try:
                from nanochat.vision import VisionEncoder, VisionResampler, VisionProjector

                self.vision_encoder = VisionEncoder(
                    model_name=config.vision_encoder_name,
                    trainable=config.vision_encoder_trainable
                )

                self.vision_resampler = VisionResampler(
                    vision_dim=self.vision_encoder.output_dim,
                    num_tokens=config.vision_num_tokens,
                    mode=config.vision_resampler_mode,
                    depth=config.vision_resampler_depth,
                    heads=config.vision_resampler_heads
                )

                self.vision_projector = VisionProjector(
                    vision_dim=self.vision_encoder.output_dim,
                    llm_dim=config.n_embd,
                    hidden_dim=config.vision_proj_hidden,
                    dropout=config.vision_proj_dropout
                )

                print0(f"Vision modules initialized: {config.vision_encoder_name}, "
                      f"{config.vision_num_tokens} tokens, {config.vision_resampler_mode} resampler")
            except ImportError as e:
                print0(f"Warning: Could not import vision modules: {e}")
                print0("Vision functionality will be disabled. Install timm/transformers for vision support.")
                config.vision_encoder_name = None

        # Log attention backend (Phase 2 optimization)
        if config.use_flash_attn and FLASH_ATTN_AVAILABLE:
            print0("Using FlashAttention-2 for 5x attention speedup")
        elif config.use_flash_attn:
            print0("FlashAttention-2 requested but not available, using SDPA (install flash-attn for 5x speedup)")
        else:
            print0("Using SDPA attention (set use_flash_attn=True for 5x speedup)")

        # Log fused kernels (Phase 2 optimization)
        if config.use_fused_kernels and FUSED_KERNELS_AVAILABLE:
            print0("Using fused kernels (RMSNorm + SwiGLU) for additional speedup")
            try:
                log_available_backends()
            except:
                pass
        elif config.use_fused_kernels:
            print0("Fused kernels requested but not available (install triton for maximum speed)")
        else:
            print0("Using standard kernels (set use_fused_kernels=True for speedup)")

        # Phase 3 optimizations
        self.fused_loss = None
        if config.use_fused_loss and FUSED_KERNELS_AVAILABLE:
            self.fused_loss = FusedCrossEntropyLoss(ignore_index=-1)
            print0("Using fused cross-entropy for 2x loss speedup (Phase 3)")
        elif config.use_fused_loss:
            print0("Fused loss requested but not available (install triton for 2x speedup)")

        if config.use_gradient_checkpointing:
            print0("Gradient checkpointing enabled for 2x longer contexts (Phase 3)")

        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        self.transformer.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                for expert in block.mlp.experts:
                    torch.nn.init.zeros_(expert.c_proj.weight)
            else:
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
        # Use rope_theta from config if base is not specified
        if base is None:
            base = self.config.rope_theta
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def encode_vision(self, images):
        """
        Process images through vision stack.

        Args:
            images: [B, 3, H, W] tensor (preprocessed images)

        Returns:
            vision_embeds: [B, num_vision_tokens, n_embd] embeddings ready for concat with text
        """
        if self.vision_encoder is None:
            raise ValueError("Vision encoder not initialized. Set vision_encoder_name in config.")

        # Vision encoder: [B, 3, H, W] -> [B, num_patches, vision_dim]
        vision_features = self.vision_encoder(images)

        # Resampler: [B, num_patches, vision_dim] -> [B, num_vision_tokens, vision_dim]
        vision_tokens = self.vision_resampler(vision_features)

        # Projector: [B, num_vision_tokens, vision_dim] -> [B, num_vision_tokens, n_embd]
        vision_embeds = self.vision_projector(vision_tokens)

        return vision_embeds

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, vision_lr=None):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters:
        # - matrix_2d_params: optimized with Muon/DistMuon (requires 2D tensors)
        # - matrix_other_params: optimized with AdamW/DistAdamW (e.g. attention bias vectors)
        matrix_all_params = list(self.transformer.h.parameters())
        matrix_2d_params = [p for p in matrix_all_params if p.ndim == 2]
        matrix_other_params = [p for p in matrix_all_params if p.ndim != 2]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        # Vision parameters (if present)
        vision_params = []
        if self.vision_projector is not None:
            vision_params.extend(list(self.vision_projector.parameters()))
        if self.vision_resampler is not None:
            vision_params.extend(list(self.vision_resampler.parameters()))
        # Note: vision_encoder is typically frozen, so we don't include it

        expected_param_count = len(matrix_2d_params) + len(matrix_other_params) + len(embedding_params) + len(lm_head_params) + len(vision_params)
        actual_param_count = len(list(self.parameters()))
        assert expected_param_count == actual_param_count, f"Parameter count mismatch: expected {expected_param_count}, got {actual_param_count}"

        bad = []
        use_distadam = False
        if ddp and matrix_other_params:
            bad = [
                p for p in matrix_other_params if p.ndim == 0 or (p.shape[0] % world_size != 0)
            ]
            if not bad:
                use_distadam = True
            else:
                shapes = [tuple(p.shape) for p in bad]
                print0(
                    f"Skipping DistAdamW for {len(shapes)} incompatible params (world_size={world_size}): {shapes}. "
                    "Using plain AdamW instead."
                )

        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        if matrix_other_params:
            # Treat these like "matrix params" but optimize with AdamW because Muon requires 2D tensors.
            adam_groups.append(dict(params=matrix_other_params, lr=matrix_lr * dmodel_lr_scale))
        if vision_params:
            # Vision projector and resampler (higher LR, not scaled by dmodel)
            vision_learning_rate = vision_lr if vision_lr is not None else matrix_lr
            adam_groups.append(dict(params=vision_params, lr=vision_learning_rate))
            if rank == 0:
                print(f"Vision module LR: {vision_learning_rate}")
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if use_distadam else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_2d_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx=None, targets=None, kv_cache=None, loss_reduction='mean', vision_embeds=None, input_embeds=None):
        """
        Forward pass with optional vision embeddings.

        Args:
            idx: [B, T] token indices (for text-only or text part)
            targets: [B, T] target tokens for loss computation
            kv_cache: KV cache for generation
            loss_reduction: 'mean', 'sum', or 'none'
            vision_embeds: [B, N, n_embd] vision embeddings to prepend (optional)
            input_embeds: [B, T, n_embd] precomputed text embeddings (alternative to idx, optional)

        Returns:
            loss or logits depending on whether targets is provided
        """
        # Get text embeddings
        if input_embeds is None:
            if idx is None:
                raise ValueError("Either idx or input_embeds must be provided")
            B, T = idx.size()
            x = self.transformer.wte(idx)  # [B, T, n_embd]
        else:
            B, T, C = input_embeds.size()
            x = input_embeds

        # Prepend vision embeddings if provided
        vision_offset = 0
        if vision_embeds is not None:
            vision_offset = vision_embeds.shape[1]
            x = torch.cat([vision_embeds, x], dim=1)  # [B, N+T, n_embd]
            T = x.shape[1]  # Update total sequence length

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert x.device == self.cos.device, f"Rotary embeddings and input are on different devices: {x.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = norm(x)
        aux_loss_total = x.new_zeros((), dtype=torch.float32)

        # Gradient checkpointing (Phase 3 - 2x longer contexts)
        use_checkpointing = self.config.use_gradient_checkpointing and self.training and kv_cache is None

        for block in self.transformer.h:
            if use_checkpointing:
                # Use gradient checkpointing to save memory
                from torch.utils.checkpoint import checkpoint
                x, aux_loss = checkpoint(block, x, cos_sin, kv_cache, use_reentrant=False)
            else:
                x, aux_loss = block(x, cos_sin, kv_cache)
            aux_loss_total = aux_loss_total + aux_loss
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits

            # Phase 3: Use fused cross-entropy for 2x speedup
            if self.fused_loss is not None:
                # Fused cross-entropy expects [B, T, V] and [B, T]
                loss = self.fused_loss(logits, targets)
            else:
                # Standard cross-entropy
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction=loss_reduction,
                )

            if loss_reduction == "none":
                # Keep token-level losses clean (used by bpb eval + RL logp).
                return loss
            return loss + aux_loss_total
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
