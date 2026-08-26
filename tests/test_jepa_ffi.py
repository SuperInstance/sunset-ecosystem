"""Tests for nerve/jepa_ffi.py bindings.

Requires libjepa_kernel.so built via cargo build --release.
Run: pytest tests/test_jepa_ffi.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from nerve.jepa_ffi import JEPAKernel


@pytest.fixture
def kernel():
    return JEPAKernel()


class TestJEPAAvailability:
    def test_kernel_detects_so(self):
        kernel = JEPAKernel()
        assert kernel.is_available() is True

    def test_kernel_repr(self):
        kernel = JEPAKernel()
        assert "JEPAKernel" in repr(kernel)
        assert "loaded=True" in repr(kernel)


class TestForwardBatch:
    def test_forward_basic(self, kernel):
        n = 10
        x = np.ones(64, dtype=np.float32)
        w1 = np.random.randn(n, 64, 32).astype(np.float32) * 0.01
        w2 = np.random.randn(n, 32, 16).astype(np.float32) * 0.01
        w3 = np.random.randn(n, 16, 16).astype(np.float32) * 0.01
        b1 = np.zeros((n, 32), dtype=np.float32)
        b2 = np.zeros((n, 16), dtype=np.float32)
        b3 = np.zeros((n, 16), dtype=np.float32)

        out = kernel.forward_batch(x, w1, w2, w3, b1, b2, b3, n)

        assert out.shape == (n, 16)
        assert np.all(np.isfinite(out))

    def test_forward_different_weights_different_outputs(self, kernel):
        n = 2
        x = np.ones(64, dtype=np.float32)
        w1 = np.array(
            [
                np.random.randn(64, 32).astype(np.float32) * 0.01,
                np.random.randn(64, 32).astype(np.float32) * 0.01,
            ]
        )
        w2 = np.array(
            [
                np.random.randn(32, 16).astype(np.float32) * 0.01,
                np.random.randn(32, 16).astype(np.float32) * 0.01,
            ]
        )
        w3 = np.array(
            [
                np.random.randn(16, 16).astype(np.float32) * 0.01,
                np.random.randn(16, 16).astype(np.float32) * 0.01,
            ]
        )
        b1 = np.zeros((2, 32), dtype=np.float32)
        b2 = np.zeros((2, 16), dtype=np.float32)
        b3 = np.zeros((2, 16), dtype=np.float32)

        out = kernel.forward_batch(x, w1, w2, w3, b1, b2, b3, n)

        # Different weights should produce different outputs
        assert np.sum(np.abs(out[0] - out[1])) > 1e-8


class TestErrorHandling:
    def test_missing_so_raises(self):
        with pytest.raises(FileNotFoundError):
            JEPAKernel("/nonexistent/libjepa_kernel.so")
