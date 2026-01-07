# GhostVis: Vision-Language Model Implementation Roadmap

**Goal:** Transform nanochat (text-only LLM) into a full vision-language model
**Architecture:** LLaVA-style (frozen vision encoder → projector → LLM)
**Scale:** 1-2B parameters, 8xH100, DeepSpeed ZeRO-3
**Philosophy:** Follow nanochat principles - small/clear/hackable

---

## Quick Reference: Training Pipeline

```
Stage 1: BASE (text-only pretraining)        [EXISTING] ✅
Stage 2: MID (vision alignment)              [TO BUILD] ❌
Stage 3: SFT (multimodal instruction)        [TO BUILD] ❌
Stage 4: RL (task-specific refinement)       [TO BUILD] ❌
Stage 5: INFERENCE (chat interface)          [TO BUILD] ❌
```

---

## Architecture Overview

### Vision Stack (New Components)
```
Image (PIL)
  ↓
VisionEncoder (SigLIP ViT-L/14, frozen)
  ↓ [B, 256, 1024]
Resampler (Perceiver or AvgPool)
  ↓ [B, 64, 1024]
Projector (2-layer MLP)
  ↓ [B, 64, 2048]
Vision Tokens
```

### Fusion Strategy
```
Input: <image>\nWhat is in this image?
  ↓
Tokens: [image_tok_1, ..., image_tok_64, text_tok_1, ..., text_tok_N]
  ↓
Embeddings: concat(vision_embeds, text_embeds)
  ↓
GPT Transformer (unchanged)
  ↓
Logits → Loss (mask out image tokens + prompt)
```

---

## Phase 1: Vision Module Foundation

### 1.1 Create Vision Package Structure

**New directory:** `nanovision/vision/`

```
nanovision/vision/
├── __init__.py
├── encoder.py      # Vision encoder wrapper (SigLIP/CLIP)
├── resampler.py    # Perceiver or pooling resampler
├── projector.py    # MLP projector to LLM dimension
└── transforms.py   # Image preprocessing
```

### 1.2 Implement Vision Encoder (`encoder.py`)

**Purpose:** Wrap pretrained vision tower (SigLIP or CLIP)

**Key Requirements:**
- Load pretrained weights from HuggingFace/timm
- Support frozen mode (default) and trainable mode
- Output patch embeddings: `[batch, num_patches, vision_dim]`
- Handle variable image sizes (resize to 336×336)

**Interface:**
```python
class VisionEncoder(nn.Module):
    def __init__(self, model_name="siglip_vit_l14", trainable=False):
        # Load pretrained vision tower
        # Freeze if not trainable

    def forward(self, images):
        # images: [B, 3, 336, 336]
        # returns: [B, num_patches, vision_dim]
```

**Config knobs to add:**
- `vision_encoder_name`: "siglip_vit_l14" | "clip_vit_l14"
- `vision_encoder_trainable`: bool (default False)
- `vision_image_size`: int (default 336)

### 1.3 Implement Resampler (`resampler.py`)

**Purpose:** Reduce variable patch count to fixed token count

**Options:**
1. **AvgPool** (simple): Spatial average pooling
2. **Perceiver** (better): Learned query tokens with cross-attention

**Interface:**
```python
class VisionResampler(nn.Module):
    def __init__(self, vision_dim, num_tokens=64, mode="perceiver"):
        # mode: "avgpool" or "perceiver"
        # num_tokens: output token count

    def forward(self, vision_features):
        # vision_features: [B, P, D_v]
        # returns: [B, num_tokens, D_v]
```

**Config knobs:**
- `vision_num_tokens`: int (default 64)
- `vision_resampler_mode`: "avgpool" | "perceiver"
- `vision_resampler_depth`: int (default 2, for perceiver)
- `vision_resampler_heads`: int (default 8, for perceiver)

### 1.4 Implement Projector (`projector.py`)

**Purpose:** Project vision tokens to LLM hidden dimension

**Architecture:** 2-layer MLP with GELU and LayerNorm
```
vision_dim → hidden_dim → llm_dim
```

**Interface:**
```python
class VisionProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim, hidden_dim=2048):
        # Linear(vision_dim, hidden_dim)
        # GELU
        # Linear(hidden_dim, llm_dim)
        # LayerNorm(llm_dim)

    def forward(self, vision_tokens):
        # vision_tokens: [B, N, vision_dim]
        # returns: [B, N, llm_dim]
```

**Config knobs:**
- `vision_proj_hidden`: int (default 2048)
- `vision_proj_dropout`: float (default 0.0)

### 1.5 Implement Image Transforms (`transforms.py`)

**Purpose:** Image preprocessing pipeline

**Requirements:**
- Resize to `vision_image_size` (default 336×336)
- Normalize with vision encoder stats (CLIP/SigLIP specific)
- Support PIL images, torch tensors, and bytes

**Interface:**
```python
def get_vision_transforms(encoder_name, image_size=336):
    # Returns torchvision.transforms.Compose
    # Resize, ToTensor, Normalize

def preprocess_image(image, encoder_name="siglip_vit_l14"):
    # image: PIL.Image, bytes, or tensor
    # returns: [3, H, W] tensor
```

---

## Phase 2: Model Integration

### 2.1 Extend GPTConfig (`model_configs.py`)

**Add vision configuration fields:**

```python
@dataclass
class GPTConfig:
    # ... existing fields ...

    # Vision encoder config
    vision_encoder_name: str = None          # "siglip_vit_l14" | "clip_vit_l14" | None
    vision_encoder_trainable: bool = False
    vision_image_size: int = 336

    # Resampler config
    vision_num_tokens: int = 64
    vision_resampler_mode: str = "perceiver"
    vision_resampler_depth: int = 2
    vision_resampler_heads: int = 8

    # Projector config
    vision_proj_hidden: int = 2048
    vision_proj_dropout: float = 0.0

    # Special tokens
    image_token_id: int = None
```

**Add vision model constructor:**

```python
def get_vlm_1_5b_config():
    """1.5B vision-language model based on Qwen2.5-1.5B"""
    config = get_qwen25_1_5b_config()
    config.vision_encoder_name = "siglip_vit_l14"
    config.vision_encoder_trainable = False
    config.vision_num_tokens = 64
    config.vision_resampler_mode = "perceiver"
    return config
```

### 2.2 Modify GPT Class (`gpt.py`)

**Add vision modules to GPT.__init__():**

```python
class GPT(nn.Module):
    def __init__(self, config):
        # ... existing LLM modules ...

        # Vision modules (optional)
        self.vision_encoder = None
        self.vision_resampler = None
        self.vision_projector = None

        if config.vision_encoder_name is not None:
            from nanovision.vision import VisionEncoder, VisionResampler, VisionProjector

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
```

**Add vision processing method:**

```python
def encode_vision(self, images):
    """
    Process images through vision stack.

    Args:
        images: [B, 3, H, W] tensor

    Returns:
        vision_embeds: [B, num_vision_tokens, n_embd]
    """
    if self.vision_encoder is None:
        raise ValueError("Vision encoder not initialized")

    # Vision encoder: [B, 3, H, W] -> [B, P, D_v]
    vision_features = self.vision_encoder(images)

    # Resampler: [B, P, D_v] -> [B, N, D_v]
    vision_tokens = self.vision_resampler(vision_features)

    # Projector: [B, N, D_v] -> [B, N, D_llm]
    vision_embeds = self.vision_projector(vision_tokens)

    return vision_embeds
```

**Modify forward() to accept vision embeddings:**

```python
def forward(self, idx=None, targets=None, input_embeds=None, vision_embeds=None):
    """
    Args:
        idx: [B, T] token indices (for text-only)
        targets: [B, T] target tokens
        input_embeds: [B, T, n_embd] precomputed embeddings (alternative to idx)
        vision_embeds: [B, N, n_embd] vision embeddings to prepend
    """

    # Get text embeddings
    if input_embeds is None:
        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
    else:
        tok_emb = input_embeds

    # Prepend vision embeddings if provided
    if vision_embeds is not None:
        x = torch.cat([vision_embeds, tok_emb], dim=1)  # [B, N+T, n_embd]
    else:
        x = tok_emb  # [B, T, n_embd]

    # Apply RoPE (adjust for vision token offset)
    if vision_embeds is not None:
        # RoPE positions start after vision tokens
        vision_len = vision_embeds.shape[1]
        # ... adjust RoPE application ...

    # Rest of forward pass unchanged
    # ...
```

**Update setup_optimizers() for vision modules:**

```python
def setup_optimizers(self, learning_rate, weight_decay, vision_lr=None):
    """
    Creates optimizers with separate LR for vision components.

    Args:
        vision_lr: LR for vision projector (if None, use learning_rate)
    """
    # Group 1: Vision projector (higher LR)
    # Group 2: LLM matrices (Muon)
    # Group 3: LLM embeddings (AdamW)
    # Vision encoder stays frozen (no optimizer)
```

### 2.3 Update Checkpoint Manager (`checkpoint_manager.py`)

**Modify save_checkpoint() to save vision modules:**

```python
def save_checkpoint(rank, model, optimizer, checkpoint_dir, step, config):
    # ... existing code ...

    # Save vision modules separately
    if hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
        vision_state = {
            'vision_encoder': model.vision_encoder.state_dict(),
            'vision_resampler': model.vision_resampler.state_dict(),
            'vision_projector': model.vision_projector.state_dict(),
        }
        vision_path = checkpoint_dir / f"vision_{step:06d}.pt"
        torch.save(vision_state, vision_path)
```

**Modify load_checkpoint() to load vision modules:**

```python
def load_checkpoint(checkpoint_path, model, optimizer=None):
    # ... load main checkpoint ...

    # Load vision checkpoint if exists
    vision_path = checkpoint_path.parent / checkpoint_path.name.replace("model_", "vision_")
    if vision_path.exists():
        vision_state = torch.load(vision_path)
        model.vision_encoder.load_state_dict(vision_state['vision_encoder'])
        model.vision_resampler.load_state_dict(vision_state['vision_resampler'])
        model.vision_projector.load_state_dict(vision_state['vision_projector'])
```

---

## Phase 3: Tokenizer & Conversation Rendering

### 3.1 Add Image Special Tokens (`tokenizer.py`)

**Modify HuggingFaceTokenizer and RustBPETokenizer:**

**Add special tokens:**
```python
special_tokens = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
    "<|image|>",  # NEW: Image placeholder token
]
```

**Add token ID accessor:**
```python
@property
def image_token_id(self):
    return self.special_tokens["<|image|>"]
```

### 3.2 Extend Conversation Rendering

**Modify render_conversation() to handle image fields:**

```python
def render_conversation(self, conversation, image_token_count=64):
    """
    Render conversation with image support.

    Args:
        conversation: List[Dict] with {"role": ..., "content": ..., "image": ...}
        image_token_count: Number of tokens to reserve for <|image|>

    Returns:
        tokens: List[int] with image placeholders
        targets: List[int] with supervision mask
        has_image: bool indicating if conversation contains images
    """

    # Detect if any message has "image" field
    has_image = any("image" in msg for msg in conversation)

    # Render text as usual
    # If <|image|> in content, replace with image_token_count repetitions of image_token_id
    # This creates "slots" for vision embeddings

    # Targets: mask out image tokens (set to -100)
```

**Example conversation format:**
```python
conversation = [
    {
        "role": "user",
        "content": "<|image|>\nWhat is in this image?",
        "image": PIL.Image.open("cat.jpg")  # NEW: Image field
    },
    {
        "role": "assistant",
        "content": "A cat sitting on a table."
    }
]
```

**Token layout after rendering:**
```
[bos, user_start, img_tok_1, ..., img_tok_64, What, is, in, this, image, ?, user_end,
 assistant_start, A, cat, sitting, on, a, table, ., assistant_end]

Targets:
[-100, -100,      -100,       ..., -100,       -100, -100, ..., -100,
 -100,            A,   cat, sitting, on, a, table, ., assistant_end]
```

---

## Phase 4: Data Pipeline

### 4.1 Create Vision Dataset Loaders (`tasks/vision/`)

**New directory:** `tasks/vision/`

```
tasks/vision/
├── __init__.py
├── vqav2.py           # VQAv2 dataset
├── textvqa.py         # TextVQA dataset
├── chartqa.py         # ChartQA dataset
├── mmmu.py            # MMMU benchmark
├── coco_captions.py   # COCO captioning (for pretraining)
└── laion.py           # LAION subset (for pretraining)
```

**Base class pattern (extend tasks/common.py):**

```python
from tasks.common import Task
from PIL import Image

class VisionTask(Task):
    """Base class for vision-language tasks"""

    def __getitem__(self, idx):
        """
        Returns:
            {
                "image": PIL.Image,
                "messages": List[Dict[str, str]]
            }
        """
        raise NotImplementedError
```

**Example: COCO Captions for pretraining:**

```python
class COCOCaptions(VisionTask):
    def __init__(self, split="train"):
        self.dataset = load_dataset("HuggingFaceM4/COCO", split=split)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": item["image"],
            "messages": [
                {"role": "user", "content": "<|image|>\nDescribe this image."},
                {"role": "assistant", "content": item["caption"]}
            ]
        }
```

### 4.2 Add Vision Recipes (`data_recipes.py`)

**Add vision training recipes:**

```python
def get_vision_pretrain_recipe():
    """Vision-language pretraining mix"""
    return {
        'coco_captions': (0.3, 'train'),
        'laion_filtered': (0.7, None),  # Streaming dataset
    }

def get_vision_sft_recipe():
    """Multimodal instruction following"""
    return {
        'vqav2': (0.3, 'train'),
        'textvqa': (0.2, 'train'),
        'chartqa': (0.1, 'train'),
        'coco_captions': (0.2, 'train'),
        'smoltalk': (0.2, 'train'),  # Mix in text-only to prevent forgetting
    }
```

### 4.3 Create Vision Dataloader (`dataloader.py`)

**Add multimodal data loader:**

```python
def vision_data_loader(task_mixture, tokenizer, batch_size, vision_transforms):
    """
    Yields batches with both images and text.

    Yields:
        {
            'images': [B, 3, H, W],
            'tokens': [B, T],
            'targets': [B, T],
            'has_vision': [B] bool tensor
        }
    """

    for batch in task_mixture:
        # Batch is list of dicts with 'image' and 'messages' fields

        # Process images
        images = []
        for item in batch:
            if 'image' in item:
                img_tensor = vision_transforms(item['image'])
                images.append(img_tensor)

        # Tokenize conversations (handles <|image|> placeholders)
        tokens, targets = [], []
        for item in batch:
            t, tgt = tokenizer.render_conversation(item['messages'])
            tokens.append(t)
            targets.append(tgt)

        # Collate (pad to max length in batch)
        # ...

        yield {
            'images': torch.stack(images) if images else None,
            'tokens': tokens_tensor,
            'targets': targets_tensor,
            'has_vision': torch.tensor([('image' in item) for item in batch])
        }
```

---

## Phase 5: Training Script Modifications

### 5.1 Create Vision Pretraining Script (`scripts/vision_pretrain.py`)

**Purpose:** Stage 2 - Align vision encoder to LLM

**Copy from:** `base_train.py` (similar structure)

**Key changes:**
- Load vision-enabled model (use VLM config)
- Use vision dataloader (COCO + LAION mix)
- Freeze LLM completely, only train projector + resampler
- Lower learning rate (5e-5 for projector)
- Shorter training (1-2 epochs over 1M image-text pairs)

**Critical optimizer setup:**
```python
# Only train vision projector + resampler
optimizer = torch.optim.AdamW([
    {'params': model.vision_projector.parameters(), 'lr': 5e-5},
    {'params': model.vision_resampler.parameters(), 'lr': 3e-5},
], weight_decay=0.01)

# Freeze everything else
for param in model.transformer.parameters():
    param.requires_grad = False
for param in model.vision_encoder.parameters():
    param.requires_grad = False
```

### 5.2 Modify SFT Script (`scripts/chat_sft.py`)

**Line 179-184: Extend data loading:**

```python
# ORIGINAL:
doc = dataset[i]
tokens, targets = tokenizer.render_conversation(doc)

# MODIFIED:
doc = dataset[i]  # Now may contain 'image' field

# Render conversation (handles <|image|> placeholders)
tokens, targets = tokenizer.render_conversation(doc['messages'])

# Process image if present
vision_embeds = None
if 'image' in doc:
    image_tensor = vision_transforms(doc['image'])
    vision_embeds = model.encode_vision(image_tensor.unsqueeze(0))
    # vision_embeds: [1, 64, n_embd]

# Store for batch collation
batch_items.append({
    'tokens': tokens,
    'targets': targets,
    'vision_embeds': vision_embeds
})
```

**Modify forward pass to use vision_embeds:**

```python
# Collate batch
tokens_batch = pad_sequence([item['tokens'] for item in batch_items])
targets_batch = pad_sequence([item['targets'] for item in batch_items])

# Collect vision embeddings (may be None for text-only samples)
vision_embeds_batch = [item['vision_embeds'] for item in batch_items]

# Forward pass
loss = model(
    idx=tokens_batch,
    targets=targets_batch,
    vision_embeds=vision_embeds_batch  # NEW
)
```

**Add vision recipe selection:**
```python
# In config section
data_recipe_name = "vision_sft"  # or "default" for text-only

if data_recipe_name == "vision_sft":
    recipe = get_vision_sft_recipe()
else:
    recipe = get_default_sft_recipe()
```

### 5.3 Modify GRPO Script (`scripts/chat_grpo.py`)

**Line 493-520: Generation with vision:**

```python
# ORIGINAL:
tokens = tokenizer.render_for_completion(conversation)
completions = engine.generate_batch(tokens, ...)

# MODIFIED:
tokens = tokenizer.render_for_completion(conversation['messages'])

# Process image if present
vision_embeds = None
if 'image' in conversation:
    image_tensor = vision_transforms(conversation['image'])
    vision_embeds = model.encode_vision(image_tensor.unsqueeze(0))

completions = engine.generate_batch(
    tokens,
    vision_embeds=vision_embeds,  # NEW
    ...
)
```

### 5.4 Modify Engine (`nanovision/engine.py`)

**Line 288-309: Extend generate_batch() signature:**

```python
def generate_batch(
    self,
    prompt_tokens,
    vision_embeds=None,  # NEW: [B, N, D] or None
    max_new_tokens=128,
    temperature=1.0,
    top_k=0,
    ...
):
    """
    Generate text completions, optionally conditioned on images.

    Args:
        prompt_tokens: [B, T] text token indices
        vision_embeds: [B, N, D] vision embeddings (optional)
    """

    # Prefill KV cache with vision tokens if provided
    if vision_embeds is not None:
        # Vision tokens go first in sequence
        with torch.no_grad():
            _ = self.model(input_embeds=vision_embeds)
        # KV cache now contains vision context

    # Continue with text tokens
    # ...
```

**Update KVCache to handle vision prefix:**

```python
class KVCache:
    def __init__(self, ...):
        self.vision_token_count = 0  # Track vision prefix length

    def prefill_vision(self, vision_embeds):
        """Prefill cache with vision embeddings"""
        # Run through model to populate cache
        # Track vision_token_count for position offsets
```

### 5.5 Modify Evaluation (`scripts/chat_eval.py`)

**Line 52-60: Add vision support:**

```python
# ORIGINAL:
tokens = tokenizer.render_for_completion(conversation)
response = engine.generate(tokens, ...)

# MODIFIED:
tokens = tokenizer.render_for_completion(conversation['messages'])

# Check for image
vision_embeds = None
if isinstance(conversation, dict) and 'image' in conversation:
    image_tensor = vision_transforms(conversation['image'])
    vision_embeds = model.encode_vision(image_tensor.unsqueeze(0))

response = engine.generate(
    tokens,
    vision_embeds=vision_embeds,  # NEW
    ...
)
```

**Add vision benchmarks:**
```python
from tasks.vision import VQAv2, TextVQA, ChartQA, MMMU

vision_tasks = {
    'vqav2': VQAv2(split='val'),
    'textvqa': TextVQA(split='val'),
    'chartqa': ChartQA(split='val'),
    'mmmu': MMMU(split='val'),
}
```

---

## Phase 6: Inference & Serving

### 6.1 Modify CLI (`scripts/chat_cli.py`)

**Add image input support:**

```python
# Check if user input contains image path
if message.startswith("/image "):
    image_path = message[7:].strip()
    image = Image.open(image_path)

    conversation.append({
        "role": "user",
        "content": "<|image|>\n" + input("Caption or question: "),
        "image": image
    })
else:
    conversation.append({
        "role": "user",
        "content": message
    })
```

### 6.2 Modify Web UI (`scripts/chat_web.py`)

**Add image upload endpoint:**

```python
from fastapi import FastAPI, UploadFile

@app.post("/upload_image")
async def upload_image(file: UploadFile):
    # Save uploaded image
    # Return image ID for use in chat
    pass

@app.post("/chat")
async def chat(message: str, image_id: str = None):
    # If image_id provided, load image
    # Include in conversation
    # Generate response
    pass
```

**Update HTML UI (`ui.html`):**
- Add image upload button
- Display uploaded images in chat
- Show vision token visualization

---

## Training Schedule & Config Knobs

### Stage 1: BASE (Language Pretraining)
**Script:** `scripts/base_train.py` (unchanged)
**Data:** FineWeb-Edu
**Duration:** ~4 hours (d20), ~12 hours (d26)
**Output:** `base_checkpoints/`

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train -- \
  --depth=20 \
  --device_batch_size=32 \
  --total_batch_size=524288
```

### Stage 2: MID (Vision Alignment)
**Script:** `scripts/vision_pretrain.py` (new)
**Data:** COCO Captions (118K) + LAION subset (1M)
**Train:** Projector + resampler only (frozen LLM + vision encoder)
**Duration:** ~2-3 hours (1 epoch)
**Output:** `mid_checkpoints/`

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.vision_pretrain -- \
  --architecture_style=vlm_1.5b \
  --vision_encoder_name=siglip_vit_l14 \
  --vision_encoder_trainable=false \
  --projector_lr=5e-5 \
  --resampler_lr=3e-5 \
  --device_batch_size=16 \
  --total_batch_size=256 \
  --num_epochs=1
```

### Stage 3: SFT (Multimodal Instruction)
**Script:** `scripts/chat_sft.py` (modified)
**Data:** VQA + TextVQA + ChartQA + SmolTalk (20% text-only)
**Train:** Last 4-6 LLM layers + projector (frozen vision encoder)
**Duration:** ~3-4 hours
**Output:** `chatsft_checkpoints/`

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_sft -- \
  --architecture_style=vlm_1.5b \
  --data_recipe_name=vision_sft \
  --unfreeze_llm_layers=4 \
  --llm_lr=1e-6 \
  --projector_lr=3e-5 \
  --device_batch_size=8 \
  --num_epochs=3
```

### Stage 4: RL (Task-Specific)
**Script:** `scripts/chat_grpo.py` (modified)
**Data:** Visual math, diagram QA, chart reasoning
**Train:** Full model (or last N layers)
**Duration:** ~4-6 hours
**Output:** `chatrl_checkpoints/`

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_grpo -- \
  --source=sft \
  --data_recipe_name=vision_rl \
  --device_batch_size=4 \
  --num_samples=8
```

---

## Config Knob Reference

### Vision Architecture
```python
--vision_encoder_name=siglip_vit_l14    # or clip_vit_l14
--vision_encoder_trainable=false
--vision_image_size=336
--vision_num_tokens=64
--vision_resampler_mode=perceiver       # or avgpool
--vision_proj_hidden=2048
```

### Vision Training
```python
--projector_lr=5e-5                     # Projector learning rate
--resampler_lr=3e-5                     # Resampler learning rate
--llm_lr=1e-6                           # LLM learning rate (when unfrozen)
--unfreeze_llm_layers=4                 # Number of top LLM layers to train
--freeze_vision_encoder=true            # Keep vision encoder frozen
```

### Data & Optimization
```python
--data_recipe_name=vision_sft           # Use vision recipe
--device_batch_size=8                   # Lower for vision (higher memory)
--grad_accum_steps=auto                 # Auto-computed
--max_sequence_length=2048              # Including vision tokens
```

---

## Performance Optimization

### Memory Management
1. **Reduce batch size:** Vision adds ~30-40% memory overhead
   - BASE: `device_batch_size=32` → MID/SFT: `device_batch_size=8-16`
2. **Use bf16:** Enabled by default, halves memory
3. **Gradient checkpointing:** Add for vision encoder if needed
4. **Freeze vision encoder:** Saves activation memory

### Speed Optimization
1. **FlashAttention:** Already used in `gpt.py` (F.scaled_dot_product_attention)
2. **Fused kernels:** SwiGLU already fused in MLP
3. **DeepSpeed ZeRO-3:** Use `slurm/deepspeed_zero3.json`
4. **Image preprocessing:** Precompute and cache vision embeddings for static datasets

### RL-Specific
1. **Lower NUM_SAMPLES:** 16 → 8 for vision tasks
2. **Reduce MAX_NEW_TOKENS:** Vision reasoning needs shorter outputs
3. **Cache vision embeddings:** Don't recompute for multiple samples

---

## Evaluation Metrics

### Vision Benchmarks (Add to chat_eval.py)
- **VQAv2:** Visual question answering accuracy
- **TextVQA:** OCR + reasoning accuracy
- **ChartQA:** Chart understanding accuracy
- **MMMU:** Multimodal multi-discipline understanding
- **Visual CORE:** Vision-augmented CORE score

### Text Benchmarks (Regression Guard)
- **GSM8K, MMLU, MBPP, HumanEval:** Should not degrade significantly
- **ChatCORE:** May improve with visual grounding

### Compute Metrics
- **Vision FLOPs:** ~2.5x more than text-only (336×336 images)
- **Memory:** +30-40% over text-only baseline
- **Throughput:** Expect ~60-70% of text-only speed

---

## Critical Implementation Notes

### 1. RoPE Position Handling
**Problem:** Vision tokens need position indices
**Solution:** Start text positions after vision token count

```python
# In GPT.forward()
if vision_embeds is not None:
    vision_len = vision_embeds.shape[1]
    text_positions = torch.arange(vision_len, vision_len + T)
else:
    text_positions = torch.arange(0, T)

# Apply RoPE with offset positions
```

### 2. Loss Masking
**Problem:** Don't train on vision tokens or prompt
**Solution:** Extend target masking in tokenizer

```python
# In render_conversation()
targets = tokens.copy()

# Mask image tokens
for i, tok in enumerate(tokens):
    if tok == image_token_id:
        targets[i] = -100

# Mask user prompts (existing logic)
# ...
```

### 3. KV Cache with Vision
**Problem:** Vision tokens must be in cache for all samples
**Solution:** Prefill cache with vision embeddings

```python
# In Engine.generate_batch()
if vision_embeds is not None:
    # Prefill cache
    with torch.no_grad():
        self.model(input_embeds=vision_embeds, targets=None)

    # Mark cache as prefilled
    self.kv_cache.mark_prefilled(vision_embeds.shape[1])
```

### 4. Distributed Training
**Problem:** Vision encoder may not fit in DDP broadcast
**Solution:** Use FSDP or ZeRO-3 for large vision encoders

```python
# In base_train.py / vision_pretrain.py
if config.vision_encoder_name:
    # Wrap vision modules separately
    model.vision_encoder = FSDP(model.vision_encoder, ...)
    model.vision_projector = FSDP(model.vision_projector, ...)
```

### 5. Image Augmentation
**Problem:** Need diverse visual inputs during training
**Solution:** Add augmentation to transforms

```python
# In vision/transforms.py
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(336, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=..., std=...)
])
```

---

## Testing Strategy

### Unit Tests
1. Vision encoder output shapes
2. Resampler token reduction
3. Projector dimension mapping
4. Tokenizer with <|image|> placeholders
5. Loss masking correctness

### Integration Tests
1. End-to-end forward pass with vision
2. Generation with vision embeddings
3. Checkpoint save/load with vision modules
4. Multi-GPU training (DDP smoke test)

### Validation Tests
1. Overfit single image-text pair (sanity check)
2. Vision token visualization (check alignment)
3. Text-only regression (ensure no degradation)

---

## Migration Checklist

### Phase 1: Foundation ✅
- [ ] Create `nanovision/vision/` package
- [ ] Implement `VisionEncoder`
- [ ] Implement `VisionResampler`
- [ ] Implement `VisionProjector`
- [ ] Implement image transforms
- [ ] Add vision configs to `GPTConfig`
- [ ] Unit test all vision modules

### Phase 2: Model Integration ✅
- [ ] Add vision modules to `GPT.__init__()`
- [ ] Implement `GPT.encode_vision()`
- [ ] Modify `GPT.forward()` for vision_embeds
- [ ] Update `setup_optimizers()` for vision
- [ ] Update checkpoint save/load
- [ ] Integration test: forward pass with vision

### Phase 3: Tokenization ✅
- [ ] Add `<|image|>` special token
- [ ] Modify `render_conversation()` for images
- [ ] Update loss masking for vision tokens
- [ ] Test tokenizer with multimodal conversations

### Phase 4: Data Pipeline ✅
- [ ] Create vision task base class
- [ ] Implement COCO Captions loader
- [ ] Implement VQAv2 loader
- [ ] Implement TextVQA loader
- [ ] Add vision recipes to `data_recipes.py`
- [ ] Create vision dataloader
- [ ] Test data loading end-to-end

### Phase 5: Training ✅
- [ ] Create `vision_pretrain.py` script
- [ ] Modify `chat_sft.py` for vision
- [ ] Modify `chat_grpo.py` for vision
- [ ] Test training loop on single GPU
- [ ] Test multi-GPU training
- [ ] Validate checkpointing

### Phase 6: Inference ✅
- [ ] Modify `Engine` for vision_embeds
- [ ] Update KVCache for vision prefix
- [ ] Modify `chat_eval.py` for vision
- [ ] Add vision benchmarks
- [ ] Update CLI for image input
- [ ] Update web UI for image upload
- [ ] End-to-end inference test

### Phase 7: Optimization ✅
- [ ] Profile memory usage
- [ ] Tune batch sizes
- [ ] Enable gradient checkpointing if needed
- [ ] Optimize image preprocessing
- [ ] Add vision embedding caching
- [ ] Benchmark throughput

---

## Quick Start Commands

### 1. Setup Environment
```bash
cd ghostvis
pip install timm pillow datasets
```

### 2. Test Vision Modules
```bash
python -c "from nanovision.vision import VisionEncoder; print('Vision modules OK')"
```

### 3. Run Vision Pretraining (2-3 hours)
```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.vision_pretrain -- \
  --architecture_style=vlm_1.5b
```

### 4. Run Multimodal SFT (3-4 hours)
```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_sft -- \
  --source=mid \
  --data_recipe_name=vision_sft
```

### 5. Evaluate on Vision Benchmarks
```bash
python -m scripts.chat_eval --source=sft --tasks=vqav2,textvqa
```

### 6. Interactive Vision Chat
```bash
python -m scripts.chat_cli
# Then: /image path/to/image.jpg
```

---

## End-to-End Example

**Goal:** Train a 1.5B vision-language model from scratch

```bash
# Step 1: Base pretraining (4 hours, already done if you have base checkpoint)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Step 2: Vision alignment (2 hours)
torchrun --standalone --nproc_per_node=8 -m scripts.vision_pretrain

# Step 3: Multimodal SFT (3 hours)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft --data_recipe_name=vision_sft

# Step 4: Visual reasoning RL (4 hours, optional)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo --data_recipe_name=vision_rl

# Step 5: Evaluate
python -m scripts.chat_eval --tasks=vqav2,textvqa,chartqa,gsm8k,mmlu

# Step 6: Deploy
python -m scripts.chat_web
```

**Total time:** ~13 hours
**Total cost:** ~$312 on 8xH100 @ $24/hr

---

## Expected Results

### After Vision Pretraining (Stage 2):
- Model can describe images (basic captioning)
- Vision-text alignment score: ~0.6-0.7
- Text benchmarks unchanged (frozen LLM)

### After Multimodal SFT (Stage 3):
- VQAv2: ~45-50% accuracy (1.5B model)
- TextVQA: ~30-35% accuracy
- ChartQA: ~20-25% accuracy
- Text benchmarks: Minor regression (<2%)

### After Visual RL (Stage 4):
- Visual math: +5-10% over SFT
- Diagram reasoning: +10-15% over SFT
- General VQA: Slight improvement

### Comparison to Text-Only Baseline:
- **Parameters:** +0B (vision encoder frozen, not counted)
- **Latency:** +30-40% (vision encoding overhead)
- **Memory:** +30-40% (vision activations)
- **FLOPs:** +2.5x (vision forward pass)

---

## Troubleshooting

### OOM During Training
1. Reduce `device_batch_size` (32 → 16 → 8)
2. Enable gradient checkpointing for vision encoder
3. Use ZeRO-3 instead of DDP
4. Reduce image size (336 → 224)

### Poor Vision-Text Alignment
1. Increase vision pretraining duration (1 epoch → 2-3 epochs)
2. Higher projector LR (5e-5 → 1e-4)
3. Use Perceiver resampler instead of avgpool
4. Check image preprocessing (normalization stats)

### Text Capabilities Degraded
1. Increase text-only mixing ratio (20% → 40%)
2. Lower LLM learning rate during SFT
3. Freeze more LLM layers (unfreeze only last 2-3)
4. Add text-only RL phase after vision RL

### Slow Inference
1. Cache vision embeddings for repeated images
2. Reduce vision_num_tokens (64 → 32)
3. Use avgpool resampler instead of Perceiver
4. Compile model with torch.compile()

---

## Next Steps After Implementation

1. **Scale Up:** Train d26 (1.5B → 3B params) for better performance
2. **Better Data:** Curate higher-quality image-text pairs
3. **Video Support:** Extend to video frames (temporal modeling)
4. **Interleaved Inputs:** Support multiple images per conversation
5. **Image Generation:** Add diffusion decoder for text-to-image
6. **Compression:** Quantize to int8/int4 for deployment
7. **Serving:** Integrate with vLLM for production

---

**Last Updated:** 2026-01-06
**Status:** Ready for implementation
**Estimated Total Implementation Time:** 2-3 weeks (with testing)
