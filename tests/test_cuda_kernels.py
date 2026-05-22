"""Tests for CudaEinsumKernel — compile, correctness, fallback, tiling math."""

import numpy as np
import pytest

from sunset.cuda_kernels import CudaEinsumKernel


# ── Helpers ─────────────────────────────────────────────────

def _ref_einsum(room_vectors, signal_matrix, room_mask):
    """Reference numpy implementation."""
    selected = room_vectors[room_mask]
    if selected.size == 0:
        return np.empty((0, signal_matrix.shape[0]), dtype=np.float32)
    return np.einsum("ij,kj->ik", selected, signal_matrix, optimize=True)


# ── Tiling math tests ─────────────────────────────────────

class TestTilingMath:
    """Static calculations independent of CUDA availability."""

    def test_rooms_per_tile_128(self):
        """d_latent=128, float32: 512B per room, 16KB / 512 = 32 rooms."""
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        assert kernel.rooms_per_tile(d_latent=128, dtype_size=4) == 32

    def test_rooms_per_tile_64(self):
        """d_latent=64, float32: 256B per room, 16KB / 256 = 64 rooms."""
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        assert kernel.rooms_per_tile(d_latent=64, dtype_size=4) == 64

    def test_tiles_needed_500_128(self):
        """500 rooms, d=128: ceil(500/32) = 16 tiles."""
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        assert kernel.tiles_needed(n_rooms=500, d_latent=128) == 16

    def test_fits_in_sms_500_128(self):
        """16 tiles <= 20 SMs → fits."""
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        assert kernel.fits_in_sms(n_rooms=500, d_latent=128) is True

    def test_fits_in_sms_1000_128(self):
        """ceil(1000/32)=32 tiles > 20 SMs → does NOT fit."""
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        assert kernel.fits_in_sms(n_rooms=1000, d_latent=128) is False

    def test_rooms_per_tile_dtype_8(self):
        """float64 (8 bytes) halves rooms per tile."""
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        assert kernel.rooms_per_tile(d_latent=128, dtype_size=8) == 16


# ── Compile test ────────────────────────────────────────────

class TestCompile:
    """compile_kernel should return a callable for any valid dimensions."""

    def test_compile_returns_callable(self):
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        fn = kernel.compile_kernel(d_latent=128, n_rooms=500)
        assert callable(fn)

    def test_compile_caches(self):
        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        fn1 = kernel.compile_kernel(d_latent=128, n_rooms=500)
        fn2 = kernel.compile_kernel(d_latent=128, n_rooms=500)
        assert fn1 is fn2


# ── Correctness tests ───────────────────────────────────────

class TestCorrectness:
    """route_signals must match numpy reference exactly (within float tolerance)."""

    @pytest.mark.parametrize("n_rooms,d_latent", [
        (50, 32),
        (100, 64),
        (250, 128),
        (500, 128),
        (1000, 128),
    ])
    def test_correctness_vs_numpy(self, n_rooms, d_latent):
        rng = np.random.RandomState(42)
        room_vectors = rng.randn(n_rooms, d_latent).astype(np.float32)
        signal_matrix = rng.randn(d_latent, d_latent).astype(np.float32)
        room_mask = rng.rand(n_rooms) > 0.3

        kernel = CudaEinsumKernel(n_sms=20, shared_mem_kb=16)
        result = kernel.route_signals(room_vectors, signal_matrix, room_mask)
        expected = _ref_einsum(room_vectors, signal_matrix, room_mask)

        assert result.shape == expected.shape, \
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        assert result.dtype == expected.dtype
        max_diff = np.max(np.abs(result - expected))
        assert max_diff < 1e-4, f"Max diff too large: {max_diff}"

    def test_empty_mask(self):
        """All False mask → empty result."""
        rng = np.random.RandomState(7)
        room_vectors = rng.randn(100, 64).astype(np.float32)
        signal_matrix = rng.randn(64, 64).astype(np.float32)
        room_mask = np.zeros(100, dtype=bool)

        kernel = CudaEinsumKernel()
        result = kernel.route_signals(room_vectors, signal_matrix, room_mask)
        expected = _ref_einsum(room_vectors, signal_matrix, room_mask)

        assert result.shape == (0, 64)
        assert expected.shape == (0, 64)

    def test_full_mask(self):
        """All True mask → same as full matrix multiply."""
        rng = np.random.RandomState(7)
        room_vectors = rng.randn(50, 32).astype(np.float32)
        signal_matrix = rng.randn(32, 32).astype(np.float32)
        room_mask = np.ones(50, dtype=bool)

        kernel = CudaEinsumKernel()
        result = kernel.route_signals(room_vectors, signal_matrix, room_mask)
        expected = _ref_einsum(room_vectors, signal_matrix, room_mask)

        assert result.shape == (50, 32)
        max_diff = np.max(np.abs(result - expected))
        assert max_diff < 1e-4

    def test_single_selected(self):
        """Only one room selected."""
        rng = np.random.RandomState(7)
        room_vectors = rng.randn(100, 64).astype(np.float32)
        signal_matrix = rng.randn(64, 64).astype(np.float32)
        room_mask = np.zeros(100, dtype=bool)
        room_mask[42] = True

        kernel = CudaEinsumKernel()
        result = kernel.route_signals(room_vectors, signal_matrix, room_mask)
        expected = _ref_einsum(room_vectors, signal_matrix, room_mask)

        assert result.shape == (1, 64)
        max_diff = np.max(np.abs(result - expected))
        assert max_diff < 1e-4

    def test_different_d_out(self):
        """signal_matrix with d_out != d_latent."""
        rng = np.random.RandomState(7)
        room_vectors = rng.randn(100, 64).astype(np.float32)
        signal_matrix = rng.randn(32, 64).astype(np.float32)  # d_out=32, d_latent=64
        room_mask = rng.rand(100) > 0.5

        kernel = CudaEinsumKernel()
        result = kernel.route_signals(room_vectors, signal_matrix, room_mask)
        expected = _ref_einsum(room_vectors, signal_matrix, room_mask)

        assert result.shape[1] == 32
        max_diff = np.max(np.abs(result - expected))
        assert max_diff < 1e-4


# ── Fallback tests ──────────────────────────────────────────

class TestFallback:
    """When CUDA is unavailable, should still produce correct results (slower)."""

    def test_fallback_produces_correct_result(self):
        """Fallback path matches reference."""
        rng = np.random.RandomState(99)
        room_vectors = rng.randn(100, 64).astype(np.float32)
        signal_matrix = rng.randn(64, 64).astype(np.float32)
        room_mask = rng.rand(100) > 0.5

        kernel = CudaEinsumKernel()
        result = kernel.route_signals(room_vectors, signal_matrix, room_mask)
        expected = _ref_einsum(room_vectors, signal_matrix, room_mask)

        assert np.allclose(result, expected, atol=1e-5)

    def test_fallback_slower_than_cuda(self, monkeypatch):
        """When forced fallback, timing info reports cuda_available=False."""
        kernel = CudaEinsumKernel()
        bench = kernel.benchmark(n_rooms=100, d_latent=64, n_iterations=20)
        assert bench["cuda_available"] is False
        assert bench["shape_ok"] is True


# ── Benchmark tests ─────────────────────────────────────────

class TestBenchmark:
    """Benchmark returns structured data with expected keys."""

    def test_benchmark_keys(self):
        kernel = CudaEinsumKernel()
        bench = kernel.benchmark(n_rooms=100, d_latent=64, n_iterations=10)
        expected_keys = {
            "n_rooms", "d_latent", "n_selected", "n_iterations",
            "ms_per_call", "ms_per_call_np", "speedup",
            "cuda_available", "shape_ok", "max_diff",
        }
        assert expected_keys.issubset(bench.keys())

    def test_benchmark_speedup_positive(self):
        """Even on CPU fallback, speedup should be a finite number."""
        kernel = CudaEinsumKernel()
        bench = kernel.benchmark(n_rooms=100, d_latent=64, n_iterations=10)
        assert bench["speedup"] > 0
        assert np.isfinite(bench["speedup"])

    def test_benchmark_n_selected_range(self):
        kernel = CudaEinsumKernel()
        bench = kernel.benchmark(n_rooms=500, d_latent=128, n_iterations=10)
        assert 0 < bench["n_selected"] <= 500

    def test_benchmark_large_array(self):
        """Should handle 1000 rooms × 128 dims without error."""
        kernel = CudaEinsumKernel()
        bench = kernel.benchmark(n_rooms=1000, d_latent=128, n_iterations=5)
        assert bench["shape_ok"] is True
        assert bench["max_diff"] is not None
        assert bench["max_diff"] < 1e-3


# ── Input validation ────────────────────────────────────────

class TestValidation:
    """Graceful handling of malformed inputs."""

    def test_d_latent_mismatch(self):
        rng = np.random.RandomState(1)
        room_vectors = rng.randn(100, 64).astype(np.float32)
        signal_matrix = rng.randn(64, 32).astype(np.float32)  # d_latent mismatch
        room_mask = np.ones(100, dtype=bool)

        kernel = CudaEinsumKernel()
        with pytest.raises(ValueError, match="d_latent"):
            kernel.route_signals(room_vectors, signal_matrix, room_mask)

    def test_mask_length_mismatch(self):
        rng = np.random.RandomState(1)
        room_vectors = rng.randn(100, 64).astype(np.float32)
        signal_matrix = rng.randn(64, 64).astype(np.float32)
        room_mask = np.ones(50, dtype=bool)  # wrong length

        kernel = CudaEinsumKernel()
        with pytest.raises(ValueError, match="room_mask length"):
            kernel.route_signals(room_vectors, signal_matrix, room_mask)

    def test_non_float32_input(self):
        """Should accept float64 and convert internally."""
        rng = np.random.RandomState(1)
        room_vectors = rng.randn(50, 32).astype(np.float64)
        signal_matrix = rng.randn(32, 32).astype(np.float64)
        room_mask = rng.rand(50) > 0.5

        kernel = CudaEinsumKernel()
        result = kernel.route_signals(room_vectors, signal_matrix, room_mask)
        expected = _ref_einsum(room_vectors.astype(np.float32),
                               signal_matrix.astype(np.float32), room_mask)
        assert np.allclose(result, expected, atol=1e-4)


# ── Repr ────────────────────────────────────────────────────

class TestRepr:
    def test_repr_fallback(self):
        kernel = CudaEinsumKernel()
        r = repr(kernel)
        assert "CudaEinsumKernel" in r
        assert "NumPy-fallback" in r
