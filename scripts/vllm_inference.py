"""
Fast inference with vLLM backend.

Provides 5x inference throughput through continuous batching and paged attention.

Usage:
    # Single prompt generation
    python -m scripts.vllm_inference --prompt "Hello, how are you?"

    # Batch generation
    python -m scripts.vllm_inference --batch_file prompts.txt --output results.jsonl

    # Interactive chat
    python -m scripts.vllm_inference --chat

    # Benchmark throughput
    python -m scripts.vllm_inference --benchmark --num_prompts 1000
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from nanovision.vllm_backend import create_vllm_engine, benchmark_throughput, VLLM_AVAILABLE
except ImportError:
    print("Error: Could not import vLLM backend.")
    print("Make sure nanovision package is in your Python path.")
    sys.exit(1)


def single_generate(engine, prompt, max_tokens=100, temperature=0.7):
    """Generate completion for a single prompt."""
    outputs = engine.generate(
        prompts=[prompt],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return outputs[0]


def batch_generate(engine, prompts, max_tokens=100, temperature=0.7):
    """Generate completions for batch of prompts."""
    return engine.generate(
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def interactive_chat(engine):
    """Interactive chat loop."""
    print("=== GhostVis vLLM Chat ===")
    print("Type 'quit' or 'exit' to end the session.\n")

    messages = []

    while True:
        # Get user input
        user_input = input("User: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Generate response
        response = engine.chat(
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )

        # Add assistant message
        messages.append({"role": "assistant", "content": response})

        # Print response
        print(f"Assistant: {response}\n")


def benchmark(engine, num_prompts=1000, prompt_length=128, max_tokens=100):
    """Benchmark inference throughput."""
    print(f"Benchmarking with {num_prompts} prompts...")
    print(f"Prompt length: {prompt_length} tokens")
    print(f"Generation length: {max_tokens} tokens\n")

    metrics = benchmark_throughput(
        engine,
        num_prompts=num_prompts,
        prompt_length=prompt_length,
        max_tokens=max_tokens,
    )

    print("=== Benchmark Results ===")
    print(f"Total time: {metrics['total_time']:.2f} seconds")
    print(f"Total tokens generated: {metrics['total_tokens']:,}")
    print(f"Throughput: {metrics['throughput_tokens_per_sec']:.1f} tokens/sec")
    print(f"Throughput: {metrics['throughput_prompts_per_sec']:.1f} prompts/sec")
    print(f"Latency: {metrics['latency_per_prompt']*1000:.1f} ms/prompt")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GhostVis vLLM Inference")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="mid_checkpoints/vlm_small",
                        help="Path to model checkpoint")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype")

    # Generation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--prompt", type=str, help="Single prompt to generate from")
    mode_group.add_argument("--batch_file", type=str, help="File with prompts (one per line)")
    mode_group.add_argument("--chat", action="store_true", help="Interactive chat mode")
    mode_group.add_argument("--benchmark", action="store_true", help="Benchmark throughput")

    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")

    # Benchmark parameters
    parser.add_argument("--num_prompts", type=int, default=1000,
                        help="Number of prompts for benchmark")
    parser.add_argument("--prompt_length", type=int, default=128,
                        help="Average prompt length for benchmark")

    # Output
    parser.add_argument("--output", type=str, help="Output file for batch results (JSONL)")

    args = parser.parse_args()

    # Check if vLLM is available
    if not VLLM_AVAILABLE:
        print("Error: vLLM is not installed.")
        print("Install with: pip install vllm")
        return 1

    # Create engine
    print(f"Loading model from {args.model_path}...")
    print(f"Using {args.tensor_parallel_size} GPU(s), dtype={args.dtype}")

    try:
        engine = create_vllm_engine(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    print("Model loaded successfully!\n")

    # Execute based on mode
    if args.prompt:
        # Single prompt generation
        output = single_generate(
            engine,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Prompt: {output['prompt']}")
        print(f"Generated: {output['generated_text']}")
        print(f"Tokens: {output['num_tokens']}")

    elif args.batch_file:
        # Batch generation from file
        batch_file = Path(args.batch_file)
        if not batch_file.exists():
            print(f"Error: File not found: {batch_file}")
            return 1

        # Read prompts
        with open(batch_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(prompts)} prompts from {batch_file}")

        # Generate
        print("Generating completions...")
        outputs = batch_generate(
            engine,
            prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Save results
        if args.output:
            output_file = Path(args.output)
            with open(output_file, "w") as f:
                for output in outputs:
                    f.write(json.dumps(output) + "\n")
            print(f"Results saved to {output_file}")
        else:
            # Print to stdout
            for i, output in enumerate(outputs):
                print(f"\n--- Prompt {i+1} ---")
                print(f"Input: {output['prompt'][:100]}...")
                print(f"Output: {output['generated_text']}")

    elif args.chat:
        # Interactive chat
        interactive_chat(engine)

    elif args.benchmark:
        # Benchmark throughput
        benchmark(
            engine,
            num_prompts=args.num_prompts,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
        )

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
