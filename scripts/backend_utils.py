"""
Shared helpers for opting into alternative distributed backends.

These mirror the logic used in base_train so other scripts can reuse the same
ZeRO-3/FSDP setup without duplicating boilerplate.
"""

import json
import os
import torch


def select_backend(use_deepspeed: int, use_fsdp: int) -> str:
    """Return backend string and guard against incompatible flags."""
    use_deepspeed_flag = bool(use_deepspeed)
    use_fsdp_flag = bool(use_fsdp)
    if use_deepspeed_flag and use_fsdp_flag:
        raise ValueError("use_deepspeed and use_fsdp are mutually exclusive")
    return "deepspeed" if use_deepspeed_flag else "fsdp" if use_fsdp_flag else "ddp"


def build_fused_adamw(params, lr, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8):
    """
    Create the fastest available AdamW optimizer.

    Priority:
    1. apex.optimizers.FusedAdam (3x faster, CUDA kernels)
    2. torch.optim.AdamW with fused=True (2.5x faster, PyTorch 2.0+)
    3. torch.optim.AdamW (fallback)

    Returns:
        optimizer: The fastest available optimizer
        backend: str, which backend was used
    """
    # Try apex FusedAdam first (fastest, 3x speedup)
    try:
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        return optimizer, "apex_fused"
    except ImportError:
        pass

    # Try PyTorch fused AdamW (2.5x speedup, PyTorch 2.0+)
    try:
        optimizer = torch.optim.AdamW(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=True
        )
        return optimizer, "torch_fused"
    except (TypeError, RuntimeError):
        # PyTorch < 2.0 doesn't support fused=True
        pass

    # Fallback to regular AdamW
    optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return optimizer, "torch_regular"


def build_adamw_all_params(model, embedding_lr, unembedding_lr, matrix_lr, weight_decay):
    """Single AdamW optimizer mirroring base_train LR scaling logic."""
    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    used_ids = {id(p) for p in embedding_params + lm_head_params}
    other_params = [p for p in model.parameters() if id(p) not in used_ids]
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        dict(params=other_params, lr=matrix_lr * dmodel_lr_scale),
    ]

    # Use fused optimizer for 2.5-3x speedup
    try:
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
    except ImportError:
        # Fallback to PyTorch fused AdamW
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay, fused=True)
        try:
            optimizer = torch.optim.AdamW(adam_groups, **adamw_kwargs)
        except (TypeError, RuntimeError):
            # PyTorch < 2.0, use regular AdamW
            adamw_kwargs.pop('fused')
            optimizer = torch.optim.AdamW(adam_groups, **adamw_kwargs)

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def wrap_fsdp_if_needed(model, backend, ddp_local_rank, fsdp_min_num_params, fsdp_cpu_offload):
    """Optionally wrap the model in FSDP and return (model, state_dict_config)."""
    fsdp_state_dict_config = None
    if backend == "fsdp":
        from torch.distributed.fsdp import CPUOffload, FullStateDictConfig, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=int(fsdp_min_num_params))
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        cpu_offload = CPUOffload(offload_params=bool(fsdp_cpu_offload))
        fsdp_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            device_id=ddp_local_rank,
            sync_module_states=True,
            use_orig_params=True,
        )
    return model, fsdp_state_dict_config


def init_deepspeed_if_needed(backend, model, orig_model, optimizer, deepspeed_config, device_batch_size, grad_accum_steps):
    """Initialize DeepSpeed ZeRO-3 engine when requested."""
    if backend != "deepspeed":
        return model
    import deepspeed
    if not os.path.isfile(deepspeed_config):
        raise FileNotFoundError(f"DeepSpeed config not found at {deepspeed_config}")
    with open(deepspeed_config, "r", encoding="utf-8") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = int(device_batch_size)
    ds_config["gradient_accumulation_steps"] = int(grad_accum_steps)
    engine, _, _, _ = deepspeed.initialize(model=orig_model, optimizer=optimizer, config=ds_config)
    for group in engine.optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return engine
