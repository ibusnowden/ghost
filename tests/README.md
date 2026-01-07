# GhostVis Tests

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nanovision --cov-report=html

# Run specific test file
pytest tests/test_phase2_optimizations.py -v
```

### Run Specific Test Categories

```bash
# Run only Phase 2 tests
pytest tests/test_phase2_optimizations.py -v

# Skip slow tests
pytest tests/ -m "not slow"

# Run only GPU tests (requires CUDA)
pytest tests/ -m gpu

# Run integration tests
pytest tests/ -m integration
```

### Test Requirements

**Phase 2 tests require:**
- PyTorch 2.0+
- CUDA-capable GPU (for FlashAttention tests)
- FlashAttention-2 (optional, tests will skip if not available)
- Triton (optional, tests will skip if not available)
- vLLM (optional, tests will skip if not available)

## Test Coverage

### Phase 2 Optimizations

**FlashAttention-2:**
- ✅ Import test
- ✅ Correctness vs SDPA
- ✅ Model integration
- ✅ Fallback behavior
- ✅ Performance benchmark

**Fused Kernels:**
- ✅ Import test
- ✅ RMSNorm correctness
- ✅ Backend selection
- ✅ SwiGLU correctness
- ✅ Model integration
- ✅ Gradient support
- ✅ Performance benchmark

**vLLM Backend:**
- ✅ Import test
- ✅ Chat formatting
- ⚠️  Full inference (requires checkpoint)

**Integration:**
- ✅ All optimizations together
- ✅ Consistency with baseline

## Expected Results

All tests should pass on systems with:
- PyTorch 2.0+
- CUDA 11.6+
- GPU with compute capability 7.0+

Tests will skip gracefully if optional dependencies are missing.

## Troubleshooting

### "FlashAttention not installed"
```bash
pip install flash-attn --no-build-isolation
```

### "CUDA not available"
Tests requiring GPU will be skipped on CPU-only systems.

### "Import errors"
Ensure GhostVis is in your Python path:
```bash
export PYTHONPATH=/path/to/ghostvis:$PYTHONPATH
```

## Continuous Integration

These tests are designed to run in CI environments with optional dependencies:
- Core tests run on CPU
- GPU tests run on CUDA-enabled runners
- Optional dependency tests skip gracefully
