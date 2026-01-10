#!/usr/bin/env python3
"""
Vision pipeline benchmark script.

Measures throughput and performance of:
1. Image preprocessing (sequential vs parallel)
2. Vision encoding (with and without cache)
3. End-to-end VLM inference

Usage:
    python -m scripts.vision_benchmark
    python -m scripts.vision_benchmark --num-images 100 --batch-size 8
"""

import argparse
import time
import torch
import numpy as np
from PIL import Image
from typing import List

# Benchmark utilities
def create_dummy_images(num_images: int, size: int = 336) -> List[Image.Image]:
    """Create random PIL images for benchmarking."""
    images = []
    for _ in range(num_images):
        # Random RGB image
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))
    return images


def benchmark_preprocessing(num_images: int = 50, image_size: int = 336, num_runs: int = 3):
    """Benchmark image preprocessing: sequential vs parallel."""
    from nanovision.vision.transforms import (
        batch_preprocess_images,
        batch_preprocess_images_parallel,
        ImagePreprocessor,
    )

    print("\n" + "=" * 60)
    print("IMAGE PREPROCESSING BENCHMARK")
    print("=" * 60)
    print(f"Images: {num_images}, Size: {image_size}x{image_size}, Runs: {num_runs}")

    # Create test images
    images = create_dummy_images(num_images, image_size)

    # Benchmark sequential
    sequential_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = batch_preprocess_images(images, image_size=image_size)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        sequential_times.append(time.perf_counter() - t0)

    seq_avg = np.mean(sequential_times)
    seq_std = np.std(sequential_times)

    # Benchmark parallel (4 workers)
    parallel_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = batch_preprocess_images_parallel(images, image_size=image_size, num_workers=4)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        parallel_times.append(time.perf_counter() - t0)

    par_avg = np.mean(parallel_times)
    par_std = np.std(parallel_times)

    speedup = seq_avg / par_avg if par_avg > 0 else 0

    print(f"\nSequential: {seq_avg*1000:.1f}ms ± {seq_std*1000:.1f}ms ({num_images/seq_avg:.1f} img/s)")
    print(f"Parallel:   {par_avg*1000:.1f}ms ± {par_std*1000:.1f}ms ({num_images/par_avg:.1f} img/s)")
    print(f"Speedup:    {speedup:.2f}x")

    return {
        "sequential_ms": seq_avg * 1000,
        "parallel_ms": par_avg * 1000,
        "speedup": speedup,
        "images_per_sec_sequential": num_images / seq_avg,
        "images_per_sec_parallel": num_images / par_avg,
    }


def benchmark_vision_cache(device: str = "cuda", num_unique: int = 10, num_repeats: int = 5):
    """Benchmark vision embedding cache hit rates and speedup."""
    print("\n" + "=" * 60)
    print("VISION EMBEDDING CACHE BENCHMARK")
    print("=" * 60)

    try:
        from nanovision.engine import VisionEmbeddingCache
        from nanovision.vision.transforms import get_vision_transforms
    except ImportError as e:
        print(f"Could not import vision modules: {e}")
        return None

    # Create cache
    cache = VisionEmbeddingCache(max_size=100)

    # Create test images and their tensors
    transform = get_vision_transforms("siglip_vit_l14", 336, is_train=False)
    unique_images = create_dummy_images(num_unique, 336)
    unique_tensors = [transform(img).unsqueeze(0) for img in unique_images]

    # Simulate workload: each unique image accessed multiple times
    access_pattern = []
    for _ in range(num_repeats):
        access_pattern.extend(range(num_unique))
    np.random.shuffle(access_pattern)

    # Create dummy embeddings (simulate model output)
    dummy_embeddings = [torch.randn(1, 64, 2048) for _ in range(num_unique)]

    # Benchmark with cache
    cache.clear()
    cache_times = []
    for idx in access_pattern:
        tensor = unique_tensors[idx]
        key = cache.compute_key(tensor)

        t0 = time.perf_counter()
        cached = cache.get(key)
        if cached is None:
            # Simulate encoding time (sleep for consistency)
            time.sleep(0.001)  # 1ms simulated encode
            cache.put(key, dummy_embeddings[idx])
        cache_times.append(time.perf_counter() - t0)

    stats = cache.stats()
    avg_time_cached = np.mean(cache_times) * 1000

    # Benchmark without cache (always "miss")
    no_cache_times = []
    for idx in access_pattern:
        t0 = time.perf_counter()
        time.sleep(0.001)  # 1ms simulated encode
        no_cache_times.append(time.perf_counter() - t0)

    avg_time_no_cache = np.mean(no_cache_times) * 1000

    print(f"\nUnique images: {num_unique}, Repeats: {num_repeats}, Total accesses: {len(access_pattern)}")
    print(f"Cache hits:    {stats['hits']}/{stats['hits'] + stats['misses']} ({stats['hit_rate']*100:.1f}%)")
    print(f"With cache:    {avg_time_cached:.3f}ms avg per access")
    print(f"Without cache: {avg_time_no_cache:.3f}ms avg per access")
    print(f"Speedup:       {avg_time_no_cache/avg_time_cached:.2f}x (when cached)")

    return {
        "hit_rate": stats["hit_rate"],
        "avg_time_cached_ms": avg_time_cached,
        "avg_time_no_cache_ms": avg_time_no_cache,
        "speedup": avg_time_no_cache / avg_time_cached if avg_time_cached > 0 else 0,
    }


def benchmark_vlm_inference(source: str = "sft", num_prompts: int = 10, max_tokens: int = 32):
    """Benchmark end-to-end VLM inference with and without vision."""
    print("\n" + "=" * 60)
    print("VLM INFERENCE BENCHMARK")
    print("=" * 60)

    try:
        from nanochat.common import compute_init
        from nanochat.checkpoint_manager import load_model
        from nanovision.engine import Engine
        from nanovision.vision.transforms import get_vision_transforms
    except ImportError as e:
        print(f"Could not import modules: {e}")
        return None

    # Initialize
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()

    try:
        model, tokenizer, meta = load_model(source, device, phase="eval")
    except Exception as e:
        print(f"Could not load model '{source}': {e}")
        return None

    engine = Engine(model, tokenizer)

    # Check for vision support
    has_vision = hasattr(model, 'encode_vision') and model.vision_encoder is not None
    print(f"Model: {source}, Vision: {has_vision}")

    # Benchmark text-only inference
    prompt = "What is the capital of France?"
    tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())

    text_times = []
    for _ in range(num_prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=max_tokens, temperature=0.0)
        torch.cuda.synchronize()
        text_times.append(time.perf_counter() - t0)

    text_avg = np.mean(text_times)
    text_throughput = max_tokens / text_avg

    print(f"\nText-only inference:")
    print(f"  Latency:    {text_avg*1000:.1f}ms avg")
    print(f"  Throughput: {text_throughput:.1f} tokens/s")

    # Benchmark vision inference (if supported)
    if has_vision:
        transform = get_vision_transforms(
            getattr(model.config, 'vision_encoder_name', 'siglip_vit_l14'),
            getattr(model.config, 'vision_image_size', 336),
            is_train=False
        )

        # Create dummy image
        dummy_image = create_dummy_images(1, 336)[0]
        image_tensor = transform(dummy_image).unsqueeze(0).to(device)

        with torch.no_grad():
            vision_embeds = model.encode_vision(image_tensor)

        vision_times = []
        for _ in range(num_prompts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=max_tokens,
                temperature=0.0, vision_embeds=vision_embeds
            )
            torch.cuda.synchronize()
            vision_times.append(time.perf_counter() - t0)

        vision_avg = np.mean(vision_times)
        vision_throughput = max_tokens / vision_avg
        overhead = (vision_avg - text_avg) / text_avg * 100

        print(f"\nVision inference:")
        print(f"  Latency:    {vision_avg*1000:.1f}ms avg")
        print(f"  Throughput: {vision_throughput:.1f} tokens/s")
        print(f"  Overhead:   {overhead:.1f}% vs text-only")

        # Test cache effectiveness
        print(f"\nVision cache stats: {engine.get_vision_cache_stats()}")

        return {
            "text_latency_ms": text_avg * 1000,
            "text_throughput": text_throughput,
            "vision_latency_ms": vision_avg * 1000,
            "vision_throughput": vision_throughput,
            "vision_overhead_pct": overhead,
        }

    return {
        "text_latency_ms": text_avg * 1000,
        "text_throughput": text_throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="Vision pipeline benchmarks")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images for preprocessing benchmark")
    parser.add_argument("--image-size", type=int, default=336, help="Image size")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts for inference benchmark")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--source", type=str, default="sft", help="Model source (sft, mid, rl)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference benchmark (requires model)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GHOSTVIS VISION PIPELINE BENCHMARKS")
    print("=" * 60)

    # Run benchmarks
    results = {}

    # 1. Preprocessing benchmark
    results["preprocessing"] = benchmark_preprocessing(
        num_images=args.num_images,
        image_size=args.image_size
    )

    # 2. Cache benchmark
    results["cache"] = benchmark_vision_cache()

    # 3. Inference benchmark (optional)
    if not args.skip_inference:
        results["inference"] = benchmark_vlm_inference(
            source=args.source,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results.get("preprocessing"):
        print(f"Parallel preprocessing speedup: {results['preprocessing']['speedup']:.2f}x")

    if results.get("cache"):
        print(f"Cache hit rate: {results['cache']['hit_rate']*100:.1f}%")

    if results.get("inference"):
        print(f"Text throughput: {results['inference']['text_throughput']:.1f} tok/s")
        if "vision_throughput" in results["inference"]:
            print(f"Vision throughput: {results['inference']['vision_throughput']:.1f} tok/s")


if __name__ == "__main__":
    main()
