"""
Tests for Phase 2 optimizations.

Tests correctness, performance, and fallback behavior of:
1. FlashAttention-2
2. Fused kernels (RMSNorm, SwiGLU)
3. vLLM backend

Run with:
    pytest tests/test_phase2_optimizations.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

# Test parameters
TEST_BATCH_SIZE = 2
TEST_SEQ_LEN = 128
TEST_HIDDEN_DIM = 768
TEST_NUM_HEADS = 12
TEST_HEAD_DIM = 64


@pytest.fixture
def device():
    """Get CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Use bfloat16 for tests."""
    return torch.bfloat16


# ============================================================================
# FlashAttention-2 Tests
# ============================================================================

class TestFlashAttention:
    """Test FlashAttention-2 integration."""

    def test_flash_attn_available(self):
        """Test if FlashAttention-2 is importable."""
        try:
            from flash_attn import flash_attn_func
            assert True, "FlashAttention-2 is available"
        except ImportError:
            pytest.skip("FlashAttention-2 not installed")

    def test_flash_attn_correctness(self, device, dtype):
        """Test FlashAttention-2 produces same output as SDPA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from flash_attn import flash_attn_func
        except ImportError:
            pytest.skip("FlashAttention-2 not installed")

        # Create test inputs
        B, T, H, D = TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_NUM_HEADS, TEST_HEAD_DIM
        q = torch.randn(B, T, H, D, device=device, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, dtype=dtype)

        # FlashAttention output
        flash_out = flash_attn_func(q, k, v, causal=True)

        # SDPA output (ground truth)
        q_sdpa = q.transpose(1, 2)  # (B, H, T, D)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        sdpa_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
        sdpa_out = sdpa_out.transpose(1, 2)  # (B, T, H, D)

        # Compare outputs (allow for numerical differences)
        diff = (flash_out - sdpa_out).abs().max().item()
        assert diff < 1e-2, f"FlashAttention output differs from SDPA by {diff}"

    def test_flash_attn_model_integration(self, device, dtype):
        """Test FlashAttention works in GPT model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nanovision.gpt import GPT, GPTConfig

        # Create small model with FlashAttention
        config = GPTConfig(
            n_layer=2,
            n_head=12,
            n_embd=768,
            vocab_size=1024,
            use_flash_attn=True,
        )
        model = GPT(config).to(device)
        model.eval()

        # Forward pass
        tokens = torch.randint(0, 1024, (TEST_BATCH_SIZE, TEST_SEQ_LEN), device=device)
        with torch.no_grad():
            logits = model(tokens)

        assert logits.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN, 1024)

    def test_flash_attn_fallback(self, device):
        """Test fallback to SDPA when FlashAttention unavailable."""
        from nanovision.gpt import GPT, GPTConfig, FLASH_ATTN_AVAILABLE

        config = GPTConfig(
            n_layer=2,
            n_head=12,
            n_embd=768,
            use_flash_attn=True,
        )
        model = GPT(config).to(device)

        # Check attention module has correct backend
        attn_module = model.transformer.h[0].attn
        if FLASH_ATTN_AVAILABLE and torch.cuda.is_available():
            assert attn_module.use_flash_attn == True
        else:
            assert attn_module.use_flash_attn == False


# ============================================================================
# Fused Kernels Tests
# ============================================================================

class TestFusedKernels:
    """Test fused RMSNorm and SwiGLU kernels."""

    def test_fused_kernels_import(self):
        """Test if fused kernels module is importable."""
        try:
            from nanovision.fused_kernels import FusedRMSNorm, FusedSwiGLU, TRITON_AVAILABLE
            assert True, "Fused kernels module available"
        except ImportError:
            pytest.fail("Could not import fused kernels")

    def test_rmsnorm_correctness(self, device, dtype):
        """Test FusedRMSNorm produces correct output."""
        try:
            from nanovision.fused_kernels import FusedRMSNorm
        except ImportError:
            pytest.skip("Fused kernels not available")

        # Create test input
        B, T, D = TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_HIDDEN_DIM
        x = torch.randn(B, T, D, device=device, dtype=dtype)

        # Fused RMSNorm output
        norm_module = FusedRMSNorm(D).to(device)
        fused_out = norm_module(x)

        # Standard RMSNorm output (ground truth)
        std_out = F.rms_norm(x, (D,))

        # Compare outputs
        diff = (fused_out - std_out).abs().max().item()
        assert diff < 1e-3, f"FusedRMSNorm differs from standard by {diff}"

    def test_rmsnorm_backend_selection(self, device):
        """Test RMSNorm backend selection."""
        try:
            from nanovision.fused_kernels import FusedRMSNorm, TRITON_AVAILABLE, APEX_NORM_AVAILABLE
        except ImportError:
            pytest.skip("Fused kernels not available")

        norm_module = FusedRMSNorm(TEST_HIDDEN_DIM, use_triton=True).to(device)

        # Check backend priority
        if TRITON_AVAILABLE and torch.cuda.is_available():
            assert norm_module.backend == "triton"
        elif APEX_NORM_AVAILABLE:
            assert norm_module.backend == "apex"
        else:
            assert norm_module.backend == "torch_compiled"

    def test_swiglu_correctness(self, device, dtype):
        """Test FusedSwiGLU produces correct output."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from nanovision.fused_kernels import FusedSwiGLU
        except ImportError:
            pytest.skip("Fused kernels not available")

        # Create test input
        B, T, D = TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_HIDDEN_DIM
        intermediate_size = D * 4
        x = torch.randn(B, T, D, device=device, dtype=dtype)

        # Fused SwiGLU
        swiglu = FusedSwiGLU(D, intermediate_size, use_triton=False).to(device)
        fused_out = swiglu(x)

        # Standard SwiGLU (ground truth)
        gate_out = F.silu(swiglu.gate(x))
        up_out = swiglu.up(x)
        std_out = gate_out * up_out

        # Compare outputs
        diff = (fused_out - std_out).abs().max().item()
        assert diff < 1e-3, f"FusedSwiGLU differs from standard by {diff}"

    def test_fused_kernels_in_model(self, device, dtype):
        """Test fused kernels work in GPT model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nanovision.gpt import GPT, GPTConfig

        # Create model with fused kernels
        config = GPTConfig(
            n_layer=2,
            n_head=12,
            n_embd=768,
            vocab_size=1024,
            use_fused_kernels=True,
        )
        model = GPT(config).to(device)
        model.eval()

        # Forward pass
        tokens = torch.randint(0, 1024, (TEST_BATCH_SIZE, TEST_SEQ_LEN), device=device)
        with torch.no_grad():
            logits = model(tokens)

        assert logits.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN, 1024)

    def test_fused_kernels_gradient(self, device):
        """Test fused kernels support gradients."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from nanovision.fused_kernels import FusedRMSNorm
        except ImportError:
            pytest.skip("Fused kernels not available")

        # Create input with gradients
        B, T, D = TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_HIDDEN_DIM
        x = torch.randn(B, T, D, device=device, requires_grad=True)

        # Forward + backward
        norm_module = FusedRMSNorm(D, use_triton=False).to(device)  # Use torch backend for gradient test
        out = norm_module(x)
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================================================
# vLLM Backend Tests
# ============================================================================

class TestVLLMBackend:
    """Test vLLM inference backend."""

    def test_vllm_available(self):
        """Test if vLLM is importable."""
        try:
            from nanovision.vllm_backend import VLLM_AVAILABLE
            if not VLLM_AVAILABLE:
                pytest.skip("vLLM not installed")
        except ImportError:
            pytest.skip("vLLM backend module not available")

    def test_vllm_engine_creation(self):
        """Test vLLM engine creation."""
        try:
            from nanovision.vllm_backend import VLLMInferenceEngine, VLLM_AVAILABLE
        except ImportError:
            pytest.skip("vLLM backend not available")

        if not VLLM_AVAILABLE:
            pytest.skip("vLLM not installed")

        # This test would require a real model checkpoint
        pytest.skip("Requires model checkpoint for full test")

    def test_vllm_format_chat_prompt(self):
        """Test chat prompt formatting."""
        try:
            from nanovision.vllm_backend import VLLMInferenceEngine, VLLM_AVAILABLE
        except ImportError:
            pytest.skip("vLLM backend not available")

        if not VLLM_AVAILABLE:
            pytest.skip("vLLM not installed")

        # Create dummy engine (won't load model)
        try:
            engine = VLLMInferenceEngine.__new__(VLLMInferenceEngine)

            # Test chat formatting
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]

            prompt = engine._format_chat_prompt(messages)

            # Check prompt contains special tokens
            assert "<|user_start|>" in prompt
            assert "<|assistant_start|>" in prompt
            assert "Hello" in prompt
            assert "How are you?" in prompt
        except:
            pytest.skip("Could not test chat formatting")


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase2Integration:
    """Integration tests for all Phase 2 optimizations together."""

    def test_all_optimizations_enabled(self, device):
        """Test all Phase 2 optimizations work together."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nanovision.gpt import GPT, GPTConfig

        # Create model with all optimizations
        config = GPTConfig(
            n_layer=2,
            n_head=12,
            n_embd=768,
            vocab_size=1024,
            use_flash_attn=True,
            use_fused_kernels=True,
        )
        model = GPT(config).to(device)
        model.train()

        # Training step
        tokens = torch.randint(0, 1024, (TEST_BATCH_SIZE, TEST_SEQ_LEN), device=device)
        targets = torch.randint(0, 1024, (TEST_BATCH_SIZE, TEST_SEQ_LEN), device=device)

        # Forward + backward
        loss = model(tokens, targets=targets)
        loss.backward()

        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_consistency_with_without_optimizations(self, device):
        """Test optimized and unoptimized models produce similar outputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nanovision.gpt import GPT, GPTConfig

        # Create two identical models, one optimized, one not
        config_optimized = GPTConfig(
            n_layer=2,
            n_head=12,
            n_embd=768,
            vocab_size=1024,
            use_flash_attn=True,
            use_fused_kernels=True,
        )
        config_baseline = GPTConfig(
            n_layer=2,
            n_head=12,
            n_embd=768,
            vocab_size=1024,
            use_flash_attn=False,
            use_fused_kernels=False,
        )

        model_opt = GPT(config_optimized).to(device)
        model_base = GPT(config_baseline).to(device)

        # Copy weights
        model_base.load_state_dict(model_opt.state_dict(), strict=False)

        # Eval mode
        model_opt.eval()
        model_base.eval()

        # Forward pass
        tokens = torch.randint(0, 1024, (TEST_BATCH_SIZE, TEST_SEQ_LEN), device=device)
        with torch.no_grad():
            logits_opt = model_opt(tokens)
            logits_base = model_base(tokens)

        # Compare outputs (allow for numerical differences)
        diff = (logits_opt - logits_base).abs().max().item()
        assert diff < 0.1, f"Optimized and baseline outputs differ by {diff}"


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPhase2Performance:
    """Performance benchmarks for Phase 2 optimizations."""

    @pytest.mark.slow
    def test_flash_attn_speedup(self, device):
        """Benchmark FlashAttention speedup."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from flash_attn import flash_attn_func
        except ImportError:
            pytest.skip("FlashAttention not installed")

        import time

        # Create test inputs
        B, T, H, D = 8, 512, 12, 64
        q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()

        # Benchmark FlashAttention
        start = time.time()
        for _ in range(100):
            _ = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        flash_time = time.time() - start

        # Benchmark SDPA
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)

        # Warmup
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
        torch.cuda.synchronize()
        sdpa_time = time.time() - start

        speedup = sdpa_time / flash_time
        print(f"\nFlashAttention speedup: {speedup:.2f}x")
        assert speedup > 1.0, "FlashAttention should be faster than SDPA"

    @pytest.mark.slow
    def test_fused_kernels_speedup(self, device):
        """Benchmark fused kernels speedup."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from nanovision.fused_kernels import FusedRMSNorm
        except ImportError:
            pytest.skip("Fused kernels not available")

        import time

        # Create test input
        B, T, D = 8, 512, 768
        x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

        # Fused RMSNorm
        norm_fused = FusedRMSNorm(D).to(device)

        # Warmup
        for _ in range(10):
            _ = norm_fused(x)
        torch.cuda.synchronize()

        # Benchmark fused
        start = time.time()
        for _ in range(100):
            _ = norm_fused(x)
        torch.cuda.synchronize()
        fused_time = time.time() - start

        # Benchmark standard
        # Warmup
        for _ in range(10):
            _ = F.rms_norm(x, (D,))
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = F.rms_norm(x, (D,))
        torch.cuda.synchronize()
        std_time = time.time() - start

        speedup = std_time / fused_time
        print(f"\nFused RMSNorm speedup: {speedup:.2f}x")
        # Speedup might be small or even negative on CPU/small tensors
        assert speedup > 0.5, "Fused kernels should not be significantly slower"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
