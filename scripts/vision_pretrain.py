"""
Stage 2: Vision-Language Alignment Pretraining

This script trains the vision projector and resampler to align the vision encoder
with the LLM. The LLM and vision encoder are kept frozen.

Run as:
python -m scripts.vision_pretrain

Or torchrun for distributed training:
torchrun --standalone --nproc_per_node=8 -m scripts.vision_pretrain -- --device_batch_size=16
"""

import os
import time
try:
    import wandb
except ImportError:
    wandb = None
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.backend_utils import (
    build_adamw_all_params,
    build_fused_adamw,
    init_deepspeed_if_needed,
    select_backend,
    wrap_fsdp_if_needed,
)
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.data_recipes import build_sft_recipe
from nanochat.vision.transforms import get_vision_transforms
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Configuration
run = "dummy"  # wandb run name ("dummy" = no wandb logging)
source = "base"  # Load from base_checkpoints (text-only model)
step = None  # Step to load from
depth = 32  # Transformer depth (scales model size)
data_recipe = "vision_pretrain"  # Use vision_pretrain recipe (COCO captions)
dtype = "bfloat16"
device_batch_size = 16  # Lower than text training (images are memory-heavy)
total_batch_size = 256  # Much smaller than text pretraining
vision_lr = 5e-5  # Learning rate for vision projector
projector_lr = 5e-5  # Projector learning rate
resampler_lr = 3e-5  # Resampler learning rate (slightly lower)
weight_decay = 0.01
num_epochs = 1  # Usually just 1 epoch for vision alignment
max_steps = None  # If set, overrides num_epochs
eval_every = 500  # Evaluate every N steps
save_every = 2000  # Save checkpoint every N steps
freeze_llm = True  # Keep LLM frozen (only train vision modules)
freeze_vision_encoder = True  # Keep vision encoder frozen
use_deepspeed = 0
deepspeed_config = "slurm/deepspeed_zero3.json"
use_fsdp = 0
fsdp_min_num_params = 1_000_000
fsdp_cpu_offload = 0
dry_run = 0

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
backend = select_backend(use_deepspeed, use_fsdp)
if backend != "ddp":
    print0(f"Using backend={backend}")
dtype_torch = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype_torch)

# Wandb logging
use_dummy_wandb = run == "dummy" or not master_process
if not use_dummy_wandb and wandb is None:
    print0("wandb not installed; proceeding without wandb logging")
    use_dummy_wandb = True
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="ghostvis-vision", name=run, config=user_config)

# Load base model and convert to VLM
print0(f"Loading {source} model and adding vision modules...")
from nanochat.model_configs import get_vlm_config
vlm_config = get_vlm_config(depth=depth)
print0(f"Using GhostVis VLM config (depth={depth}, SwiGLU + GQA + vision)")

# Create VLM model (includes vision modules)
from nanochat.gpt import GPT
model = GPT(vlm_config)

# Load text-only checkpoint if source is base
if source == "base":
    from nanochat.checkpoint_manager import load_checkpoint, find_largest_model, find_last_step

    # Find checkpoint
    if model_tag is None:
        model_tag = find_largest_model(source)
    if step is None:
        step = find_last_step(source, model_tag)

    checkpoint_dir = get_base_dir() / f"{source}_checkpoints" / model_tag
    checkpoint_path = checkpoint_dir / f"model_{step:06d}.pt"

    print0(f"Loading text-only checkpoint: {checkpoint_path}")

    # Load checkpoint (vision modules will be randomly initialized)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load only LLM weights (ignore missing vision keys)
    model_state = checkpoint['model']
    model.load_state_dict(model_state, strict=False)
    print0("✓ Loaded text-only LLM, vision modules randomly initialized")

elif source == "mid":
    # Load existing vision checkpoint
    model, tokenizer, meta = load_model("mid", device, phase="train", model_tag=model_tag, step=step)
    print0("✓ Loaded existing vision checkpoint")
else:
    raise ValueError(f"Unknown source: {source}")

# Get tokenizer
from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer()

# Freeze modules as specified
if freeze_llm:
    print0("Freezing LLM parameters...")
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False

if freeze_vision_encoder and model.vision_encoder is not None:
    print0("Freezing vision encoder...")
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print0(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# Wrap model for distributed training
orig_model = model
model, fsdp_state_dict_config = wrap_fsdp_if_needed(
    model,
    backend=backend,
    ddp_local_rank=ddp_local_rank,
    fsdp_min_num_params=fsdp_min_num_params,
    fsdp_cpu_offload=fsdp_cpu_offload,
)

# Initialize DeepSpeed if needed
ds_engine = None
if backend == "deepspeed":
    ds_engine, _, _, _ = init_deepspeed_if_needed(
        model=model,
        config=deepspeed_config,
    )

# Create optimizer (only for trainable params) - use fused for 2.5-3x speedup
trainable_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer, optimizer_backend = build_fused_adamw(
    trainable_params_list,
    lr=vision_lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.95),
)
print0(f"Using optimizer backend: {optimizer_backend}")

# Load vision dataset
print0(f"Loading vision dataset: {data_recipe}")
train_ds, val_ds = build_sft_recipe(data_recipe)
print0(f"Train dataset: {len(train_ds)} examples")
print0(f"Val dataset: {len(val_ds)} examples")

# Get vision transforms
vision_transforms = get_vision_transforms(
    encoder_name=vlm_config.vision_encoder_name,
    image_size=vlm_config.vision_image_size,
    is_train=True,
)

# Calculate training steps
examples_per_step = total_batch_size
grad_accum_steps = examples_per_step // (device_batch_size * ddp_world_size)
assert examples_per_step % (device_batch_size * ddp_world_size) == 0, "total_batch_size must be divisible by device_batch_size * world_size"

if max_steps is not None:
    total_steps = max_steps
else:
    total_steps = (len(train_ds) * num_epochs) // examples_per_step

print0(f"Training config:")
print0(f"  Examples per step: {examples_per_step}")
print0(f"  Device batch size: {device_batch_size}")
print0(f"  Gradient accumulation: {grad_accum_steps}")
print0(f"  Total steps: {total_steps}")
print0(f"  Num epochs: {num_epochs}")

# Training state
step_idx = 0
epoch_idx = 0
dataset_idx = 0

# Training loop
print0("Starting vision pretraining...")
model.train()
if freeze_llm:
    model.transformer.eval()  # Keep LLM in eval mode
if freeze_vision_encoder and model.vision_encoder is not None:
    model.vision_encoder.eval()

while step_idx < total_steps:
    # Prepare batch
    batch_images = []
    batch_tokens = []
    batch_targets = []

    for micro_step in range(grad_accum_steps):
        micro_batch_images = []
        micro_batch_tokens = []
        micro_batch_targets = []

        for _ in range(device_batch_size):
            # Get example
            if dataset_idx >= len(train_ds):
                dataset_idx = 0
                epoch_idx += 1
                print0(f"Completed epoch {epoch_idx}")
                if epoch_idx >= num_epochs and max_steps is None:
                    break

            doc = train_ds[dataset_idx]
            dataset_idx += 1

            # Extract image and conversation
            image = doc.get("image")
            conversation = {"messages": doc["messages"]}

            # Preprocess image
            if image is not None:
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                image_tensor = vision_transforms(image)
                micro_batch_images.append(image_tensor)

            # Tokenize conversation
            tokens, targets = tokenizer.render_conversation(
                conversation,
                max_tokens=2048,
                num_vision_tokens=vlm_config.vision_num_tokens,
            )

            micro_batch_tokens.append(torch.tensor(tokens, dtype=torch.long))
            micro_batch_targets.append(torch.tensor(targets, dtype=torch.long))

        # Collate micro-batch
        if micro_batch_images:
            batch_images.extend(micro_batch_images)
        batch_tokens.extend(micro_batch_tokens)
        batch_targets.extend(micro_batch_targets)

    if not batch_tokens:
        break  # End of training

    # Pad sequences
    max_len = max(len(t) for t in batch_tokens)
    tokens_padded = torch.full((len(batch_tokens), max_len), 0, dtype=torch.long)
    targets_padded = torch.full((len(batch_targets), max_len), -1, dtype=torch.long)

    for i, (toks, tgts) in enumerate(zip(batch_tokens, batch_targets)):
        tokens_padded[i, :len(toks)] = toks
        targets_padded[i, :len(tgts)] = tgts

    # Move to device
    tokens_padded = tokens_padded.to(device)
    targets_padded = targets_padded.to(device)

    # Process images
    vision_embeds = None
    if batch_images:
        images_tensor = torch.stack(batch_images).to(device)
        with autocast_ctx:
            vision_embeds = orig_model.encode_vision(images_tensor)

    # Forward pass
    t0 = time.time()
    with autocast_ctx:
        loss = model(
            idx=tokens_padded,
            targets=targets_padded,
            vision_embeds=vision_embeds,
        )
        loss = loss / grad_accum_steps

    # Backward pass
    loss.backward()

    # Optimizer step
    if (step_idx + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Sync and log
    if ddp:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    loss_item = loss.item() * grad_accum_steps
    dt = time.time() - t0

    if step_idx % 10 == 0:
        print0(f"step {step_idx}/{total_steps} | loss: {loss_item:.4f} | dt: {dt*1000:.1f}ms")

    wandb_run.log({
        "train/loss": loss_item,
        "train/step": step_idx,
        "train/epoch": epoch_idx,
    }, step=step_idx)

    # Save checkpoint
    if step_idx > 0 and step_idx % save_every == 0 and master_process and not dry_run:
        print0(f"Saving checkpoint at step {step_idx}...")
        checkpoint_dir = get_base_dir() / "mid_checkpoints" / f"d{depth}"
        save_checkpoint(
            ddp_rank,
            model,
            optimizer,
            checkpoint_dir,
            step_idx,
            vlm_config,
            fsdp_state_dict_config=fsdp_state_dict_config,
        )

    step_idx += 1

# Final checkpoint
if master_process and not dry_run:
    print0(f"Saving final checkpoint at step {step_idx}...")
    checkpoint_dir = get_base_dir() / "mid_checkpoints" / f"d{depth}"
    save_checkpoint(
        ddp_rank,
        model,
        optimizer,
        checkpoint_dir,
        step_idx,
        vlm_config,
        fsdp_state_dict_config=fsdp_state_dict_config,
    )

print0("Vision pretraining complete!")
wandb_run.finish()
compute_cleanup()
