"""Tests for PersistentCUDAGrid — GPU-accelerated JEPA forward.

The CUDA library is typically absent in CI/test environments.
These tests verify the Python-side API, error handling, and weight layout.
"""

import numpy as np
import pytest

from nerve import cuda_bridge


class TestLibraryLoading:
    def test_cuda_lib_attribute_exists(self):
        assert hasattr(cuda_bridge, "_CUDA_LIB")

    def test_persistent_grid_raises_without_cuda(self):
        # If _CUDA_LIB is None (no .so found), construction must fail
        if cuda_bridge._CUDA_LIB is not None:
            pytest.skip("CUDA library is present — skipping no-CUDA test")
        weights = {
            "w1": np.ones((10, 64, 32)),
            "w2": np.ones((10, 32, 16)),
            "w3": np.ones((10, 16, 16)),
            "b1": np.ones((10, 32)),
            "b2": np.ones((10, 16)),
            "b3": np.ones((10, 16)),
        }
        with pytest.raises(RuntimeError, match="libjepa_cuda.so"):
            cuda_bridge.PersistentCUDAGrid(10, weights)


class TestPersistentCUDAGridWithMockLib:
    def test_init_shapes(self, monkeypatch):
        """Simulate _CUDA_LIB present and verify init copies weights."""

        class FakeLib:
            pass

        monkeypatch.setattr(cuda_bridge, "_CUDA_LIB", FakeLib())
        weights = {
            "w1": np.arange(10 * 64 * 32).reshape(10, 64, 32).astype(np.float64),
            "w2": np.ones((10, 32, 16)),
            "w3": np.ones((10, 16, 16)),
            "b1": np.ones((10, 32)),
            "b2": np.ones((10, 16)),
            "b3": np.ones((10, 16)),
        }
        grid = cuda_bridge.PersistentCUDAGrid(10, weights)
        assert grid.n == 10
        assert grid._out.shape == (10, 16)
        assert grid._out.dtype == np.float32
        # weights were converted to contiguous float32
        assert grid._w1.dtype == np.float32
        assert grid._w1.flags["C_CONTIGUOUS"]

    def test_repr(self, monkeypatch):
        class FakeLib:
            pass

        monkeypatch.setattr(cuda_bridge, "_CUDA_LIB", FakeLib())
        weights = {
            "w1": np.ones((2, 64, 32)),
            "w2": np.ones((2, 32, 16)),
            "w3": np.ones((2, 16, 16)),
            "b1": np.ones((2, 32)),
            "b2": np.ones((2, 16)),
            "b3": np.ones((2, 16)),
        }
        grid = cuda_bridge.PersistentCUDAGrid(2, weights)
        r = repr(grid)
        assert "PersistentCUDAGrid" in r
        assert "n=2" in r
