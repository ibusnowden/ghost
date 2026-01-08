# GhostVis

![nanochat logo](dev/nanochat.png)

> Vision-Language Model: The best multimodal ChatGPT you can train yourself.

**GhostVis** is a vision-language model built on top of [nanochat](https://github.com/karpathy/nanochat), transforming a text-only LLM into a full multimodal system capable of understanding both images and text. Like nanochat, it maintains a clean, minimal, hackable codebase designed to run on a single 8XH100 node.

## What is GhostVis?

GhostVis extends nanochat's capabilities by adding:
- **Vision Understanding**: Process images alongside text using a frozen vision encoder (SigLIP/CLIP)
- **Modern Architecture**: Built on Qwen2.5-1.5B with SwiGLU activation and Grouped-Query Attention (GQA)
- **LLaVA-Style Fusion**: Vision encoder → Perceiver resampler → MLP projector → LLM
- **Full Pipeline**: Pretraining, vision alignment, multimodal SFT, RL, and inference
- **Hackable Design**: Same nanochat philosophy - minimal, readable, forkable code

## Current Status

🚧 **In Development** - Vision capabilities are being implemented

**Completed:**
- ✅ Vision module foundation (encoder, resampler, projector)
- ✅ Model integration (GPT class extensions)
- ✅ Tokenizer & conversation rendering with image support
- ✅ Vision dataset loaders (COCO, VQA, TextVQA, ChartQA)

**In Progress:**
- 🔄 Training script modifications
- 🔄 Inference & serving updates
- 🔄 Vision benchmarks

See [skills.md](skills.md) for the complete implementation roadmap.

## Architecture Overview

```
Input Image (PIL)
    ↓
Vision Encoder (SigLIP ViT-L/14) [Frozen]
    ↓ [B, 256, 1024]
Vision Resampler (Perceiver)
    ↓ [B, 64, 1024]
Vision Projector (2-layer MLP)
    ↓ [B, 64, 1536]
Vision Embeddings
    ↓
    ⊕ (concat) ← Text Embeddings [B, T, 1536]
    ↓
GPT Transformer (Qwen2.5-1.5B: 28L, SwiGLU, GQA 6:1)
    ↓
Output Logits
```

**Key Features:**
- **Base Model**: Qwen2.5-1.5B (1.5B parameters, 28 layers, 1536 hidden dim)
- **Activation**: SwiGLU (better than ReLU/GELU for modern LLMs)
- **Attention**: Grouped-Query Attention (6:1 ratio for efficient inference)
- **Context**: 32k tokens (more than enough for vision + long conversations)
- **Vision**: 64 tokens per image at 336×336 resolution

## Quick start

### Text-Only Training (Original nanochat)

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier model. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

### Vision-Language Training (GhostVis)

🚧 **Coming Soon** - Full vision training pipeline

Once vision training is complete, you'll be able to train a multimodal model:

```bash
# Stage 1: Base pretraining (text-only, ~4 hours)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Stage 2: Vision alignment (~2-3 hours)
torchrun --standalone --nproc_per_node=8 -m scripts.vision_pretrain -- --architecture_style=vlm_1.5b

# Stage 3: Multimodal SFT (~3-4 hours)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --data_recipe_name=vision_sft

# Stage 4: Serve the vision model
python -m scripts.chat_web
# Then use /image path/to/image.jpg to chat with images
```

**Total training time**: ~13 hours | **Total cost**: ~$312 on 8XH100 @ $24/hr

See [skills.md](skills.md) for the complete vision training roadmap.

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensates by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run GhostVis/nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

## Documentation

GhostVis includes comprehensive documentation for all aspects of the vision-language implementation:

### Core Documentation
- **[skills.md](skills.md)** - Complete implementation roadmap for vision capabilities (7 phases, detailed specs)
- **[walkthrough.md](walkthrough.md)** - Qwen2.5 integration walkthrough (SwiGLU, GQA, GPU metrics)
- **[MODEL_CONFIG_ANALYSIS.md](MODEL_CONFIG_ANALYSIS.md)** - Deep dive into model architecture choices (SwiGLU vs ReLU, GQA ratios, context windows)

### Implementation Details
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - High-level project status and architecture overview
- **[docs/AGENTS.md](docs/AGENTS.md)** - R1-style post-training pipeline (GRPO, rejection sampling)

### Training & Optimization Guides
- **[COMPLETE_OPTIMIZATION_GUIDE.md](COMPLETE_OPTIMIZATION_GUIDE.md)** - Comprehensive optimization reference
- **[PHASE1_OPTIMIZATIONS_COMPLETE.md](PHASE1_OPTIMIZATIONS_COMPLETE.md)** - Phase 1 optimizations
- **[PHASE2_OPTIMIZATIONS_COMPLETE.md](PHASE2_OPTIMIZATIONS_COMPLETE.md)** - Phase 2 optimizations
- **[PHASE3_OPTIMIZATIONS_COMPLETE.md](PHASE3_OPTIMIZATIONS_COMPLETE.md)** - Phase 3 optimizations
- **[OPTIMIZATIONS_SUMMARY.md](OPTIMIZATIONS_SUMMARY.md)** - Quick optimization summary

### Verification & Testing
- **[PIPELINE_VERIFICATION.md](PIPELINE_VERIFICATION.md)** - Training pipeline verification
- **[TRAINING_TIMELINE.md](TRAINING_TIMELINE.md)** - Expected training timelines

## Questions & Exploring the Code

GhostVis (like nanochat) is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

This includes all py, rs, html, toml, sh files, excludes the `rustbpe/target` folder, and chooses the cxml output format. Everything is written to the `packaged.txt` file.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

**Key files to start exploring:**
- `nanovision/gpt.py` - Core model architecture (Qwen2.5 with vision support)
- `nanovision/model_configs.py` - Model configurations (Qwen2.5-1.5B, 7B, small variants)
- `nanovision/vision/` - Vision modules (encoder, resampler, projector)
- `tasks/vision/` - Vision dataset loaders (VQA, COCO, TextVQA, ChartQA)
- `scripts/base_train.py` - Pretraining script
- `scripts/chat_sft.py` - SFT script (being extended for vision)
- **[skills.md](skills.md)** - Complete vision implementation roadmap

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## Contributing

GhostVis extends nanochat's mission to make state-of-the-art models accessible. The goal is to improve the state of the art in micro vision-language models that are accessible to work with end to end on budgets of < $1000 dollars. Like nanochat, GhostVis maintains the philosophy: no giant configuration objects, model factories, or if-then-else monsters. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase for vision-language models.

**Vision Roadmap**: See [skills.md](skills.md) for the detailed implementation plan.

## Acknowledgements

- **GhostVis** is built on top of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- The name (nanochat) derives from [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining
- nanochat is inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified nanoGPT with clear metrics and a leaderboard
- Vision architecture inspired by [LLaVA](https://llava-vl.github.io/) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- Base LLM architecture from [Qwen2.5](https://github.com/QwenLM/Qwen2.5) (SwiGLU, GQA)
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb, smoltalk, and vision datasets
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance

## Cite

If you find GhostVis helpful in your research, please cite both GhostVis and the original nanochat:

```bibtex
@misc{ghostvis,
  title = {GhostVis: Vision-Language Model Extension of nanochat},
  year = {2025},
  note = {Built on nanochat by Andrej Karpathy}
}

@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
