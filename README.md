# GhostVis

> Vision-Language Model: The best multimodal ChatGPT you can train yourself.

**GhostVis** is a vision-language model built on top of [nanochat](https://github.com/karpathy/nanochat), transforming a text-only LLM into a full multimodal system capable of understanding both images and text. Like nanochat, it maintains a clean, minimal, hackable codebase designed to run on a single 8xH100 node.

## Model Configuration

GhostVis uses a single, scalable architecture with modern optimizations:

```python
from nanovision.model_configs import get_config, get_vlm_config

# Text-only model (scales with depth)
config = get_config(depth=32)  # ~1.5B params

# Vision-language model
vlm_config = get_vlm_config(depth=32)  # ~1.5B + vision modules
```

### Architecture Features

| Feature | Description |
|---------|-------------|
| **SwiGLU** | Better activation than ReLU/GELU |
| **GQA** | Grouped-Query Attention (2:1 ratio) |
| **FlashAttention-2** | Memory-efficient attention |
| **Fused Kernels** | Triton-optimized RMSNorm, SwiGLU |
| **RoPE** | Rotary positional encoding |
| **Fused Loss** | Memory-efficient cross-entropy |

### Model Scaling

| Depth | Params | Dim | Heads | KV Heads |
|-------|--------|-----|-------|----------|
| 20 | ~500M | 1280 | 10 | 5 |
| 32 | ~1.5B | 2048 | 16 | 8 |
| 48 | ~3B | 3072 | 24 | 12 |

## Architecture Overview

```
Input Image (336x336)
    |
Vision Encoder (SigLIP ViT-L/14) [Frozen]
    | [B, 256, 1024]
Perceiver Resampler (2 layers)
    | [B, 64, 1024]
MLP Projector (2 layers)
    | [B, 64, model_dim]
    |
    + <-- Text Embeddings [B, T, model_dim]
    |
GPT Transformer (SwiGLU + GQA + FlashAttn)
    |
Output Logits
```

---

## End-to-End Training Pipeline

GhostVis follows a 4-phase training pipeline. Each phase can be run with SLURM on HPC clusters or directly with torchrun.

### Phase 1: Base Pretraining (Text-Only)

Train the base language model on text data.

**Direct (torchrun):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 \
    --run=base_d32 \
    --device_batch_size=32
```

**SLURM:**
```bash
# slurm/phase1_base_pretrain.sh
#!/bin/bash
#SBATCH --job-name=ghostvis-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

source ~/.bashrc
conda activate ghostvis

cd /path/to/ghost
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 \
    --run=base_d32 \
    --device_batch_size=32

# Submit: sbatch slurm/phase1_base_pretrain.sh
```

**Config:**
- **Time**: ~4-6 hours on 8xH100
- **Cost**: ~$100-150
- **Output**: `~/.cache/ghostvis/base_checkpoints/d32/`

---

### Phase 2: Vision Alignment (Mid-Training)

Train only the vision projector and resampler to align vision with text.

**Direct (torchrun):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.vision_pretrain -- \
    --depth=32 \
    --source=base \
    --run=vision_align_d32 \
    --device_batch_size=16 \
    --data_recipe=vision_pretrain
```

**SLURM:**
```bash
# slurm/phase2_vision_align.sh
#!/bin/bash
#SBATCH --job-name=ghostvis-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --partition=gpu

source ~/.bashrc
conda activate ghostvis

cd /path/to/ghost
torchrun --standalone --nproc_per_node=8 -m scripts.vision_pretrain -- \
    --depth=32 \
    --source=base \
    --run=vision_align_d32 \
    --device_batch_size=16

# Submit: sbatch slurm/phase2_vision_align.sh
```

**Config:**
- **Time**: ~2-3 hours on 8xH100
- **Cost**: ~$50-75
- **Frozen**: LLM + Vision Encoder
- **Trainable**: Projector + Resampler only
- **Dataset**: COCO Captions
- **Output**: `~/.cache/ghostvis/mid_checkpoints/d32/`

---

### Phase 3: Multimodal SFT (Alignment)

Fine-tune with multimodal instruction data. Optionally unfreeze last N LLM layers.

**Direct (torchrun):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --depth=32 \
    --source=mid \
    --run=sft_d32 \
    --device_batch_size=8 \
    --data_recipe=vision_sft \
    --unfreeze_llm_layers=4
```

**SLURM:**
```bash
# slurm/phase3_multimodal_sft.sh
#!/bin/bash
#SBATCH --job-name=ghostvis-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --partition=gpu

source ~/.bashrc
conda activate ghostvis

cd /path/to/ghost
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --depth=32 \
    --source=mid \
    --run=sft_d32 \
    --device_batch_size=8 \
    --data_recipe=vision_sft \
    --unfreeze_llm_layers=4

# Submit: sbatch slurm/phase3_multimodal_sft.sh
```

**Config:**
- **Time**: ~3-4 hours on 8xH100
- **Cost**: ~$75-100
- **Frozen**: Vision Encoder + first N-4 LLM layers
- **Trainable**: Last 4 LLM layers + Projector + Resampler
- **Dataset**: LLaVA-Instruct, VQAv2, TextVQA, ChartQA
- **Output**: `~/.cache/ghostvis/sft_checkpoints/d32/`

---

### Phase 4 (Optional): GRPO Reinforcement Learning

Further improve with reinforcement learning on reasoning tasks.

**Direct (torchrun):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- \
    --depth=32 \
    --source=sft \
    --run=rl_d32 \
    --device_batch_size=4 \
    --reward_mode=dapo
```

**SLURM:**
```bash
# slurm/phase4_grpo_rl.sh
#!/bin/bash
#SBATCH --job-name=ghostvis-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

source ~/.bashrc
conda activate ghostvis

cd /path/to/ghost
torchrun --standalone --nproc_per_node=8 -m scripts.chat_grpo -- \
    --depth=32 \
    --source=sft \
    --run=rl_d32 \
    --device_batch_size=4 \
    --reward_mode=dapo

# Submit: sbatch slurm/phase4_grpo_rl.sh
```

**Config:**
- **Time**: ~4-6 hours on 8xH100
- **Cost**: ~$100-150
- **Output**: `~/.cache/ghostvis/rl_checkpoints/d32/`

---

## Inference

### Web Chat Interface

```bash
python -m scripts.chat_web --source=sft --depth=32
# Open http://localhost:7860
# Use /image path/to/image.jpg to chat with images
```

### SGLang (67-129% Faster)

For production inference, use SGLang with RadixAttention:

```bash
# Install
pip install 'sglang[all]'

# Interactive chat
python -m scripts.sglang_inference --chat --source=sft --depth=32

# Benchmark
python -m scripts.sglang_inference --benchmark --num-prompts=100 --with-vision
```

**Python API:**
```python
from nanovision.sglang_backend import create_sglang_engine
from PIL import Image

engine = create_sglang_engine(source="sft", depth=32)

image = Image.open("cat.jpg")
outputs = engine.generate(
    prompts=["What is in this image?"],
    images=[image],
    max_tokens=100
)
print(outputs[0])
```

### SLURM Inference Job

```bash
# slurm/inference.sh
#!/bin/bash
#SBATCH --job-name=ghostvis-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=gpu

source ~/.bashrc
conda activate ghostvis

cd /path/to/ghost
python -m scripts.sglang_inference \
    --source=sft \
    --depth=32 \
    --benchmark \
    --num-prompts=100

# Submit: sbatch slurm/inference.sh
```

---

## Training Data Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GhostVis Training Data Flow                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Base Pretraining (Text-Only)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FineWeb-Edu (HuggingFaceFW/fineweb-edu)                            │    │
│  │  • ~40B tokens across 800 parquet shards                            │    │
│  │  • High-quality educational web text                                │    │
│  │  • Tokenized with rust BPE (65K vocab)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  Phase 2: Vision Alignment                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  COCO Captions 2017 (HuggingFaceM4/COCO)                            │    │
│  │  • ~100K image-text pairs                                           │    │
│  │  • Natural images with human captions                               │    │
│  │  • SigLIP ViT-L/14 vision encoding (336x336)                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  Phase 3: Multimodal SFT                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Vision-focused (vision_sft recipe):                                │    │
│  │  • VQAv2: ~440K visual QA pairs                                     │    │
│  │  • TextVQA: ~34K text-in-image QA pairs                             │    │
│  │  • COCO Captions: ~100K image-caption pairs                         │    │
│  │                                                                     │    │
│  │  Reasoning-focused (r1_ot_mixed recipe):                            │    │
│  │  • OpenThoughts: ~1.2M reasoning chains                             │    │
│  │  • SmolTalk: ~1M chat conversations                                 │    │
│  │  • GSM8K: ~7.5K math word problems                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  Phase 4: GRPO RL (Optional)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Vision Tasks (verifiable rewards):                                 │    │
│  │  • VQAv2-RL: Visual QA with exact match (~40K)                      │    │
│  │  • ChartQA: Chart understanding (~18K)                              │    │
│  │  • DocVQA: Document understanding (~10K)                            │    │
│  │                                                                     │    │
│  │  Reasoning Tasks:                                                   │    │
│  │  • GSM8K: Grade school math (~7.5K)                                 │    │
│  │  • MATH: Competition math (~12K)                                    │    │
│  │  • MBPP: Python programming (~1K)                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Summary

| Phase | Script | Time | Cost | Trainable | Data |
|-------|--------|------|------|-----------|------|
| **1. Base** | `scripts.base_train` | 4-6h | $100-150 | Full model | FineWeb-Edu (40B tokens) |
| **2. Mid** | `scripts.vision_pretrain` | 2-3h | $50-75 | Projector + Resampler | COCO (100K pairs) |
| **3. SFT** | `scripts.chat_sft` | 3-4h | $75-100 | Last 4 layers + Projector | VQAv2 + TextVQA + SmolTalk |
| **4. RL** | `scripts.chat_grpo` | 4-6h | $100-150 | Full model | VQAv2 + ChartQA + GSM8K |

**Total**: ~13-19 hours, ~$325-475 for full pipeline
**Skip Phase 1**: ~9-13 hours, ~$225-325 if reusing text checkpoint

---

## Evaluation Benchmarks

Run evaluation after any training phase:

```bash
# Primary vision benchmarks (recommended)
python -m scripts.eval --source=sft --depth=32 --tasks=vqav2,textvqa,chartqa,docvqa,mmmu

# Optional reasoning benchmarks
python -m scripts.eval --source=sft --depth=32 --tasks=gsm8k,math
```

### Vision Benchmarks (Primary)

| Benchmark | Description | Metric | Split |
|-----------|-------------|--------|-------|
| **VQAv2** | Visual question answering | Accuracy | val |
| **TextVQA** | Text-in-image reading | Accuracy | val |
| **ChartQA** | Chart/graph understanding | Relaxed Acc | test |
| **DocVQA** | Document understanding | ANLS | val |
| **MMMU** | Multimodal multitask understanding | Accuracy | val |

### Reasoning Benchmarks (Secondary)

| Benchmark | Description | Metric | Split |
|-----------|-------------|--------|-------|
| **GSM8K** | Grade school math | Accuracy | test |
| **MATH** | Competition math | Accuracy | test |

---

## Checkpoints

Pre-trained checkpoints available on HuggingFace:

```python
# Download from HuggingFace
from huggingface_hub import hf_hub_download

# SFT checkpoint
hf_hub_download(
    repo_id="ibrahima2222/nanochat-d32",
    filename="sft/model_035900.pt",
    local_dir="./checkpoints"
)

# RL checkpoint
hf_hub_download(
    repo_id="ibrahima2222/nanochat-d32",
    filename="rl/model_000499.pt",
    local_dir="./checkpoints"
)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `nanovision/gpt.py` | Core GPT model with vision support |
| `nanovision/model_configs.py` | `get_config()` and `get_vlm_config()` |
| `nanovision/vision/` | Vision modules (encoder, resampler, projector) |
| `nanovision/sglang_backend.py` | SGLang inference (67-129% faster) |
| `nanovision/tokenizer.py` | Rust BPE tokenizer with `<\|image\|>` support |
| `scripts/base_train.py` | Phase 1: Base pretraining |
| `scripts/vision_pretrain.py` | Phase 2: Vision alignment |
| `scripts/chat_sft.py` | Phase 3: Multimodal SFT |
| `scripts/chat_grpo.py` | Phase 4: GRPO RL |
| `scripts/sglang_inference.py` | Fast inference |
| `tasks/` | Vision datasets (VQAv2, COCO, TextVQA, ChartQA) |

---

## Requirements

```bash
# Core
pip install torch torchvision
pip install flash-attn --no-build-isolation
pip install tiktoken rustbpe transformers datasets

# Vision
pip install pillow timm open_clip_torch

# Inference (optional)
pip install 'sglang[all]'

# Development
pip install pytest wandb
```

---

## Acknowledgements

- Built on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- Vision architecture inspired by [LLaVA](https://llava-vl.github.io/) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- Base architecture from [Qwen2.5](https://github.com/QwenLM/Qwen2.5)

## License

MIT
