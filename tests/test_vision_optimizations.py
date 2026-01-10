"""
Tests for Phase 7 vision optimizations.

Tests:
1. VisionEmbeddingCache - LRU caching behavior
2. Parallel image preprocessing - correctness and speedup
3. ImagePreprocessor class - batch processing
4. Transform caching via lru_cache

Run with:
    pytest tests/test_vision_optimizations.py -v
"""

import pytest
import torch
import numpy as np
from PIL import Image
import time


# ============================================================================
# Vision Embedding Cache Tests
# ============================================================================

class TestVisionEmbeddingCache:
    """Tests for VisionEmbeddingCache class."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        from nanovision.engine import VisionEmbeddingCache

        cache = VisionEmbeddingCache(max_size=50, enabled=True)
        assert cache.max_size == 50
        assert cache.enabled is True
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_put_get(self):
        """Test basic put/get operations."""
        from nanovision.engine import VisionEmbeddingCache

        cache = VisionEmbeddingCache(max_size=10)

        # Create test tensor and embedding
        tensor = torch.randn(1, 3, 336, 336)
        embedding = torch.randn(1, 64, 2048)

        # Compute key and store
        key = cache.compute_key(tensor)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex digest length

        cache.put(key, embedding)
        assert len(cache.cache) == 1

        # Retrieve
        cached = cache.get(key)
        assert cached is not None
        assert torch.allclose(cached, embedding)
        assert cache.hits == 1

    def test_cache_miss(self):
        """Test cache miss behavior."""
        from nanovision.engine import VisionEmbeddingCache

        cache = VisionEmbeddingCache(max_size=10)

        result = cache.get("nonexistent_key")
        assert result is None
        assert cache.misses == 1

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from nanovision.engine import VisionEmbeddingCache

        cache = VisionEmbeddingCache(max_size=3)

        # Fill cache
        for i in range(3):
            tensor = torch.randn(1, 3, 336, 336) + i  # Different tensors
            key = cache.compute_key(tensor)
            cache.put(key, torch.randn(1, 64, 2048))

        assert len(cache.cache) == 3

        # Add one more - should evict oldest
        tensor = torch.randn(1, 3, 336, 336) + 100
        key = cache.compute_key(tensor)
        cache.put(key, torch.randn(1, 64, 2048))

        assert len(cache.cache) == 3  # Still 3, oldest evicted

    def test_cache_disabled(self):
        """Test cache when disabled."""
        from nanovision.engine import VisionEmbeddingCache

        cache = VisionEmbeddingCache(max_size=10, enabled=False)

        tensor = torch.randn(1, 3, 336, 336)
        key = cache.compute_key(tensor)
        embedding = torch.randn(1, 64, 2048)

        cache.put(key, embedding)
        assert len(cache.cache) == 0  # Not stored

        result = cache.get(key)
        assert result is None  # Not retrieved

    def test_cache_stats(self):
        """Test cache statistics."""
        from nanovision.engine import VisionEmbeddingCache

        cache = VisionEmbeddingCache(max_size=10)

        # Generate some activity
        tensor1 = torch.randn(1, 3, 336, 336)
        key1 = cache.compute_key(tensor1)
        cache.put(key1, torch.randn(1, 64, 2048))

        cache.get(key1)  # Hit
        cache.get(key1)  # Hit
        cache.get("miss")  # Miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2/3, rel=0.01)

    def test_cache_clear(self):
        """Test cache clearing."""
        from nanovision.engine import VisionEmbeddingCache

        cache = VisionEmbeddingCache(max_size=10)

        # Add some entries
        for i in range(5):
            tensor = torch.randn(1, 3, 336, 336) + i
            cache.put(cache.compute_key(tensor), torch.randn(1, 64, 2048))

        assert len(cache.cache) == 5

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0


# ============================================================================
# Parallel Image Preprocessing Tests
# ============================================================================

def create_test_images(num_images: int, size: int = 336):
    """Helper to create test images."""
    images = []
    for i in range(num_images):
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))
    return images


class TestParallelPreprocessing:
    """Tests for parallel image preprocessing."""

    def test_sequential_batch_preprocess(self):
        """Test sequential batch preprocessing."""
        from nanovision.vision.transforms import batch_preprocess_images

        images = create_test_images(4)
        result = batch_preprocess_images(images, image_size=336)

        assert result.shape == (4, 3, 336, 336)
        assert result.dtype == torch.float32

    def test_parallel_batch_preprocess(self):
        """Test parallel batch preprocessing."""
        from nanovision.vision.transforms import batch_preprocess_images_parallel

        images = create_test_images(8)
        result = batch_preprocess_images_parallel(images, image_size=336, num_workers=4)

        assert result.shape == (8, 3, 336, 336)
        assert result.dtype == torch.float32

    def test_parallel_equals_sequential(self):
        """Test that parallel and sequential produce same results."""
        from nanovision.vision.transforms import (
            batch_preprocess_images,
            batch_preprocess_images_parallel,
        )

        # Use same random seed for reproducibility
        np.random.seed(42)
        images1 = create_test_images(4)

        np.random.seed(42)
        images2 = create_test_images(4)

        seq_result = batch_preprocess_images(images1, image_size=336)
        par_result = batch_preprocess_images_parallel(images2, image_size=336, num_workers=2)

        assert torch.allclose(seq_result, par_result)

    def test_parallel_small_batch_fallback(self):
        """Test that small batches fall back to sequential."""
        from nanovision.vision.transforms import batch_preprocess_images_parallel

        # Small batch should use sequential internally
        images = create_test_images(2)
        result = batch_preprocess_images_parallel(images, image_size=336, num_workers=4)

        assert result.shape == (2, 3, 336, 336)

    def test_parallel_speedup(self):
        """Test that parallel is faster for large batches."""
        from nanovision.vision.transforms import (
            batch_preprocess_images,
            batch_preprocess_images_parallel,
        )

        images = create_test_images(16)

        # Time sequential
        t0 = time.perf_counter()
        _ = batch_preprocess_images(images, image_size=336)
        seq_time = time.perf_counter() - t0

        # Time parallel
        t0 = time.perf_counter()
        _ = batch_preprocess_images_parallel(images, image_size=336, num_workers=4)
        par_time = time.perf_counter() - t0

        # Parallel should be faster (or at least not significantly slower)
        # Allow some tolerance for thread overhead on small batches
        assert par_time <= seq_time * 1.5, f"Parallel {par_time:.3f}s should be <= {seq_time*1.5:.3f}s"


# ============================================================================
# ImagePreprocessor Class Tests
# ============================================================================

class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        from nanovision.vision.transforms import ImagePreprocessor

        preprocessor = ImagePreprocessor(
            encoder_name="siglip_vit_l14",
            image_size=336,
            num_workers=4
        )

        assert preprocessor.encoder_name == "siglip_vit_l14"
        assert preprocessor.image_size == 336
        assert preprocessor.num_workers == 4
        assert preprocessor.transform is not None

    def test_preprocessor_single_image(self):
        """Test processing single image."""
        from nanovision.vision.transforms import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        image = create_test_images(1)[0]

        result = preprocessor.process_single(image)

        assert result.shape == (3, 336, 336)
        assert preprocessor.images_processed == 1

    def test_preprocessor_batch(self):
        """Test batch processing."""
        from nanovision.vision.transforms import ImagePreprocessor

        preprocessor = ImagePreprocessor(parallel_threshold=4)
        images = create_test_images(8)

        result = preprocessor.process_batch(images)

        assert result.shape == (8, 3, 336, 336)
        assert preprocessor.batches_processed == 1
        assert preprocessor.images_processed == 8
        assert preprocessor.parallel_batches == 1  # Used parallel

    def test_preprocessor_small_batch_no_parallel(self):
        """Test that small batches don't use parallel."""
        from nanovision.vision.transforms import ImagePreprocessor

        preprocessor = ImagePreprocessor(parallel_threshold=4)
        images = create_test_images(2)

        _ = preprocessor.process_batch(images)

        assert preprocessor.parallel_batches == 0  # Did not use parallel

    def test_preprocessor_stats(self):
        """Test preprocessor statistics."""
        from nanovision.vision.transforms import ImagePreprocessor

        preprocessor = ImagePreprocessor(parallel_threshold=4)

        # Process some batches
        preprocessor.process_batch(create_test_images(2))  # Sequential
        preprocessor.process_batch(create_test_images(8))  # Parallel

        stats = preprocessor.stats()

        assert stats["images_processed"] == 10
        assert stats["batches_processed"] == 2
        assert stats["parallel_batches"] == 1
        assert stats["parallel_ratio"] == 0.5


# ============================================================================
# Transform Caching Tests
# ============================================================================

class TestTransformCaching:
    """Tests for transform caching via lru_cache."""

    def test_transform_caching(self):
        """Test that transforms are cached."""
        from nanovision.vision.transforms import get_vision_transforms

        # Get same transform twice
        t1 = get_vision_transforms("siglip_vit_l14", 336, False)
        t2 = get_vision_transforms("siglip_vit_l14", 336, False)

        # Should be the exact same object (cached)
        assert t1 is t2

    def test_transform_different_params(self):
        """Test that different params get different transforms."""
        from nanovision.vision.transforms import get_vision_transforms

        t1 = get_vision_transforms("siglip_vit_l14", 336, False)
        t2 = get_vision_transforms("siglip_vit_l14", 224, False)
        t3 = get_vision_transforms("clip_vit_l14", 336, False)

        # Different params should give different transforms
        assert t1 is not t2
        assert t1 is not t3


# ============================================================================
# Engine Integration Tests
# ============================================================================

class TestEngineVisionCache:
    """Tests for Engine vision cache integration."""

    def test_engine_has_vision_cache(self):
        """Test that Engine has vision cache."""
        from nanovision.engine import Engine

        # Create mock model and tokenizer
        class MockModel:
            vision_encoder = None
            def get_device(self):
                return torch.device("cpu")

        class MockTokenizer:
            pass

        engine = Engine(MockModel(), MockTokenizer())

        assert hasattr(engine, "vision_cache")
        assert engine.vision_cache is not None

    def test_engine_cache_stats(self):
        """Test Engine cache stats method."""
        from nanovision.engine import Engine

        class MockModel:
            vision_encoder = None
            def get_device(self):
                return torch.device("cpu")

        class MockTokenizer:
            pass

        engine = Engine(MockModel(), MockTokenizer())
        stats = engine.get_vision_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_engine_clear_cache(self):
        """Test Engine clear cache method."""
        from nanovision.engine import Engine

        class MockModel:
            vision_encoder = None
            def get_device(self):
                return torch.device("cpu")

        class MockTokenizer:
            pass

        engine = Engine(MockModel(), MockTokenizer())

        # Add something to cache directly
        engine.vision_cache.put("test_key", torch.randn(1, 64, 2048))
        assert len(engine.vision_cache.cache) == 1

        engine.clear_vision_cache()
        assert len(engine.vision_cache.cache) == 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
