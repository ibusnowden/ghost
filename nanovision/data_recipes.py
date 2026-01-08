"""
Dataset recipes for SFT/R1-style pipelines.

Recipes:
- default: Basic SFT with ARC, GSM8K, SmolTalk
- r1_ot: OpenThoughts cold-start SFT
- r1_ot_mixed: OpenThoughts reasoning + SmolTalk chat mix
- r1_rs_sft: Rejection-sampled CoT + Chat mix
- dolci_mid: Dolci-Think mid-training for reasoning
- dolci_rs_sft: Dolci RS-SFT (RS-CoT + Dolci mix)

Vision recipes:
- vision_pretrain: Vision-language alignment (COCO captions)
- vision_sft: Multimodal instruction following (VQA + TextVQA + SmolTalk mix)
- vision_sft_full: Full multimodal SFT (includes text-only tasks for stability)
"""

import os

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.openthoughts import OpenThoughts
from tasks.jsonl_chat import JsonlChat
from tasks.dolci_think import DolciThink

# Vision tasks (lazy import to avoid dependencies if not used)
try:
    from tasks import COCOCaptions, VQAv2, TextVQA
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    COCOCaptions = VQAv2 = TextVQA = None


def build_sft_recipe(
    recipe,
    *,
    ot_dataset="open-thoughts/OpenThoughts3-1.2M",
    ot_split="train",
    ot_stop=None,
    rs_cot_path=None,
    rs_cot_stop=100_000,
    chat_ratio=0.30,
    chat_ot_answer_ratio=0.10,
    chat_ot_trace_ratio=0.05,
    gsm8k_stop=None,
    val_smoltalk_stop=2000,
    val_arc_stop=400,
    # Dolci-Think config
    dolci_path=None,
    dolci_stop=500_000,
    dolci_streaming=False,
    dolci_stream_cache=None,
):
    """
    Return (train_ds, val_ds) for a named recipe.
    """
    if recipe == "default":
        train_ds = TaskMixture([
            ARC(subset="ARC-Easy", split="train"),
            ARC(subset="ARC-Challenge", split="train"),
            GSM8K(subset="main", split="train"),
            SmolTalk(split="train", stop=10_000),
        ])
        val_ds = SmolTalk(split="test")
        return train_ds, val_ds

    if recipe == "r1_ot":
        train_ds = OpenThoughts(
            dataset_id=ot_dataset,
            split=ot_split,
            mode="cot",
            stop=ot_stop,
        )
        val_ds = SmolTalk(split="test", stop=val_smoltalk_stop)
        return train_ds, val_ds

    if recipe == "r1_ot_mixed":
        # Mixed recipe: OpenThoughts reasoning + SmolTalk chat + GSM8K math.
        # This teaches the model WHEN to think (complex) vs direct answer (simple).
        ot_count = ot_stop if ot_stop else 1_000_000
        chat_count = int(ot_count * chat_ratio / (1.0 - chat_ratio))

        train_ds = TaskMixture([
            OpenThoughts(
                dataset_id=ot_dataset,
                split=ot_split,
                mode="cot",  # With <think> blocks
                stop=ot_stop,
            ),
            SmolTalk(split="train", stop=chat_count),  # Direct answers, no thinking
            GSM8K(subset="main", split="train", stop=gsm8k_stop),
        ])
        val_ds = TaskMixture([
            SmolTalk(split="test", stop=val_smoltalk_stop),
            ARC(subset="ARC-Challenge", split="test", stop=val_arc_stop),
        ])
        return train_ds, val_ds

    if recipe == "r1_rs_sft":
        if not rs_cot_path:
            raise ValueError("r1_rs_sft requires rs_cot_path for rejection-sampled traces.")
        rs_cot_ds = JsonlChat(rs_cot_path, stop=rs_cot_stop)
        rs_cot_target = len(rs_cot_ds)
        chat_total = int(rs_cot_target * chat_ratio / (1.0 - chat_ratio))
        chat_ot_answer = max(1, int(chat_total * chat_ot_answer_ratio))
        chat_ot_trace = max(1, int(chat_total * chat_ot_trace_ratio))
        chat_smoltalk = max(1, chat_total - chat_ot_answer - chat_ot_trace)

        train_ds = TaskMixture([
            rs_cot_ds,
            SmolTalk(split="train", stop=chat_smoltalk),
            OpenThoughts(
                dataset_id=ot_dataset,
                split=ot_split,
                mode="answer_only",
                stop=chat_ot_answer,
            ),
            OpenThoughts(
                dataset_id=ot_dataset,
                split=ot_split,
                mode="cot",
                stop=chat_ot_trace,
            ),
        ])
        val_ds = TaskMixture([
            SmolTalk(split="test", stop=val_smoltalk_stop),
            ARC(subset="ARC-Challenge", split="test", stop=val_arc_stop),
        ])
        return train_ds, val_ds

    # -------------------------------------------------------------------------
    # Dolci-Think recipes (for 1.8B reasoning model)
    # -------------------------------------------------------------------------

    if recipe == "dolci_hf":
        # Load directly from HuggingFace cache (no preprocessing needed)
        train_ds = DolciThink(
            dataset_id="allenai/Dolci-Think-SFT-7B",
            mode="cot",  # Keep reasoning traces
            stop=dolci_stop,
            streaming=dolci_streaming,
            stream_cache_path=dolci_stream_cache,
        )
        val_ds = TaskMixture([
            SmolTalk(split="test", stop=val_smoltalk_stop),
            ARC(subset="ARC-Challenge", split="test", stop=val_arc_stop),
        ])
        return train_ds, val_ds

    if recipe == "dolci_mid":
        # Mid-training on Dolci-Think (preprocessed 800k subset)
        # Use prepare_dolci_think.py to create the JSONL first
        resolved_dolci_path = dolci_path or os.path.expanduser(
            "~/.cache/nanochat/dolci_think_800k.jsonl"
        )

        train_ds = DolciThink(
            local_path=resolved_dolci_path,
            mode="cot",  # Keep reasoning traces
            stop=dolci_stop,
        )
        val_ds = TaskMixture([
            SmolTalk(split="test", stop=val_smoltalk_stop),
            ARC(subset="ARC-Challenge", split="test", stop=val_arc_stop),
        ])
        return train_ds, val_ds

    if recipe == "dolci_rs_sft":
        # Dolci RS-SFT: RS-CoT + Dolci-Think mix (final polishing)
        if not rs_cot_path:
            raise ValueError("dolci_rs_sft requires rs_cot_path for rejection-sampled traces.")

        resolved_dolci_path = dolci_path or os.path.expanduser(
            "~/.cache/nanochat/dolci_think_800k.jsonl"
        )

        rs_cot_ds = JsonlChat(rs_cot_path, stop=rs_cot_stop)
        rs_cot_target = len(rs_cot_ds)

        # 70% RS-CoT, 30% chat mix (Dolci short + SmolTalk)
        chat_total = int(rs_cot_target * chat_ratio / (1.0 - chat_ratio))
        chat_dolci_short = max(1, int(chat_total * 0.50))  # 50% Dolci short-reasoning
        chat_smoltalk = max(1, int(chat_total * 0.40))     # 40% SmolTalk
        chat_dolci_cot = max(1, int(chat_total * 0.10))    # 10% Dolci full CoT (format)

        train_ds = TaskMixture([
            rs_cot_ds,
            DolciThink(
                local_path=resolved_dolci_path,
                mode="answer_only",  # Strip thinking for variety
                category_filter=["short_reasoning", "chat"],
                stop=chat_dolci_short,
            ),
            SmolTalk(split="train", stop=chat_smoltalk),
            DolciThink(
                local_path=resolved_dolci_path,
                mode="cot",
                category_filter=["reasoning_math", "coding"],
                stop=chat_dolci_cot,
            ),
        ])
        val_ds = TaskMixture([
            SmolTalk(split="test", stop=val_smoltalk_stop),
            ARC(subset="ARC-Challenge", split="test", stop=val_arc_stop),
        ])
        return train_ds, val_ds

    # -------------------------------------------------------------------------
    # Vision-Language Model recipes
    # -------------------------------------------------------------------------

    if recipe == "vision_pretrain":
        # Stage 2: Vision-language alignment pretraining
        # Train only projector + resampler, freeze LLM + vision encoder
        if not VISION_AVAILABLE:
            raise ImportError("Vision tasks not available. Install required packages (datasets, PIL).")

        train_ds = COCOCaptions(split="train", stop=100_000)  # 100k samples
        val_ds = COCOCaptions(split="val", stop=1000)
        return train_ds, val_ds

    if recipe == "vision_sft":
        # Stage 3: Multimodal instruction fine-tuning
        # Mix vision and text tasks (20% text-only to prevent forgetting)
        if not VISION_AVAILABLE:
            raise ImportError("Vision tasks not available. Install required packages (datasets, PIL).")

        train_ds = TaskMixture([
            VQAv2(split="train", stop=50_000),      # 50k VQA samples
            TextVQA(split="train", stop=20_000),    # 20k TextVQA samples
            COCOCaptions(split="train", stop=10_000),  # 10k caption samples
            SmolTalk(split="train", stop=20_000),   # 20k text-only (prevent forgetting)
        ])
        val_ds = TaskMixture([
            VQAv2(split="val", stop=1000),
            TextVQA(split="val", stop=500),
            SmolTalk(split="test", stop=500),
        ])
        return train_ds, val_ds

    if recipe == "vision_sft_full":
        # Full multimodal SFT with more text-only tasks for stability
        # 60% vision tasks, 40% text-only tasks
        if not VISION_AVAILABLE:
            raise ImportError("Vision tasks not available. Install required packages (datasets, PIL).")

        train_ds = TaskMixture([
            # Vision tasks (60%)
            VQAv2(split="train", stop=40_000),
            TextVQA(split="train", stop=20_000),
            COCOCaptions(split="train", stop=15_000),
            # Text-only tasks (40%) - prevent catastrophic forgetting
            SmolTalk(split="train", stop=30_000),
            ARC(subset="ARC-Easy", split="train"),
            ARC(subset="ARC-Challenge", split="train"),
            GSM8K(subset="main", split="train", stop=5_000),
        ])
        val_ds = TaskMixture([
            VQAv2(split="val", stop=1000),
            TextVQA(split="val", stop=500),
            SmolTalk(split="test", stop=1000),
            ARC(subset="ARC-Challenge", split="test", stop=200),
        ])
        return train_ds, val_ds

    raise ValueError(f"Unknown SFT recipe: {recipe}")
