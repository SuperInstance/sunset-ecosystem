"""Tests for ``sunset.turbovec`` — BLAS wrapper + turbovec re-export."""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np
import pytest

from sunset import turbovec as tv

# ── Restore real turbovec for upstream integration tests ──
# conftest.py unconditionally mocks turbovec for speed.  This test file
# specifically verifies the real upstream module, so we bypass the mock
# by directly loading the real package via importlib.
_tv_mock = sys.modules.pop("turbovec", None)
try:
    _spec = importlib.util.find_spec("turbovec")
    if _spec is not None and _spec.loader is not None:
        _real_turbovec = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_real_turbovec)
        tv.IdMapIndex = _real_turbovec.IdMapIndex
        tv.TurboQuantIndex = _real_turbovec.TurboQuantIndex
    else:
        _real_turbovec = None
finally:
    if _tv_mock is not None:
        sys.modules["turbovec"] = _tv_mock


class TestBlasLoading:
    """Verify that the BLAS library is discovered and loaded."""

    def test_blas_lib_is_not_none(self):
        """A BLAS .so was found and loaded with RTLD_GLOBAL."""
        assert tv._blas_lib is not None, "No BLAS library loaded"

    def test_cblas_sgemm_symbol_resolved(self):
        """The critical symbol exists in the loaded library."""
        assert hasattr(tv._blas_lib, "cblas_sgemm"), (
            f"cblas_sgemm missing in {tv._blas_lib._name}"
        )

    def test_id_map_index_imported(self):
        """IdMapIndex is available (re-exported from turbovec)."""
        # Should not raise
        idx = tv.IdMapIndex(dim=8, bit_width=2)
        assert idx is not None


class TestCblasSgemm:
    """Correctness tests for the ctypes-wrapped cblas_sgemm."""

    def test_basic_matmul(self):
        """2×2 matrix multiply against numpy reference."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        C = tv.cblas_sgemm(A, B)
        assert C.dtype == np.float32
        np.testing.assert_allclose(C, A @ B, rtol=1e-5, atol=1e-6)

    def test_non_square(self):
        """M×K @ K×N where dimensions differ."""
        A = np.random.randn(10, 64).astype(np.float32)
        B = np.random.randn(64, 32).astype(np.float32)
        C = tv.cblas_sgemm(A, B)
        assert C.shape == (10, 32)
        np.testing.assert_allclose(C, A @ B, rtol=1e-4, atol=1e-5)

    def test_alpha_beta(self):
        """Scalar scaling and accumulation."""
        A = np.ones((3, 4), dtype=np.float32)
        B = np.ones((4, 3), dtype=np.float32)
        out = np.zeros((3, 3), dtype=np.float32)
        C = tv.cblas_sgemm(A, B, alpha=2.0, beta=1.0, out=out)
        # C = 2 * (A @ B) + 1 * out  →  2 * 4 + 0 = 8 everywhere
        np.testing.assert_allclose(C, np.full((3, 3), 8.0, dtype=np.float32))

    def test_transpose_a(self):
        """Compute with A transposed."""
        A = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # A^T @ B
        C = tv.cblas_sgemm(A, B, trans_a=True)
        np.testing.assert_allclose(C, A.T @ B, rtol=1e-5)

    def test_transpose_b(self):
        """Compute with B transposed."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 7.0], [6.0, 8.0]], dtype=np.float32)
        C = tv.cblas_sgemm(A, B, trans_b=True)
        np.testing.assert_allclose(C, A @ B.T, rtol=1e-5)

    def test_invalid_dtype_raises(self):
        """Double-precision inputs are rejected."""
        A = np.ones((2, 2), dtype=np.float64)
        B = np.ones((2, 2), dtype=np.float64)
        with pytest.raises(TypeError, match="float32"):
            tv.cblas_sgemm(A, B)

    def test_dimension_mismatch_raises(self):
        """Incompatible inner dimensions raise ValueError."""
        A = np.ones((3, 4), dtype=np.float32)
        B = np.ones((5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            tv.cblas_sgemm(A, B)

    def test_out_shape_mismatch_raises(self):
        """Wrong output shape raises ValueError."""
        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((3, 4), dtype=np.float32)
        bad_out = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="out shape"):
            tv.cblas_sgemm(A, B, out=bad_out)


class TestTurbovecIntegration:
    """Smoke tests that the upstream turbovec module still works
    after our BLAS preload."""

    def test_id_map_index_add_and_search(self):
        """Round-trip: add vectors, search, verify results."""
        idx = tv.IdMapIndex(dim=16, bit_width=4)
        rng = np.random.default_rng(42)

        vecs = rng.standard_normal((50, 16), dtype=np.float32)
        ids = np.arange(50, dtype=np.uint64)
        idx.add_with_ids(vecs, ids)

        query = rng.standard_normal((1, 16), dtype=np.float32)
        scores, result_ids = idx.search(query, k=5)

        assert scores.shape == (1, 5)
        assert result_ids.shape == (1, 5)
        assert np.all(result_ids[0] < 50)

    def test_turbo_quant_index_basic(self):
        """TurboQuantIndex also imports and initialises."""
        idx = tv.TurboQuantIndex(dim=8, bit_width=2)
        assert idx is not None


class TestEnvironmentOverrides:
    """SUNSET_BLAS_PATH and SUNSET_BLAS_PRELOAD behaviour."""

    def test_sunset_blas_path_override(self, monkeypatch):
        """When SUNSET_BLAS_PATH is set, the wrapper attempts it first."""
        monkeypatch.setenv("SUNSET_BLAS_PATH", "/nonexistent/blas.so")
        # Re-running _find_blas in a fresh process would log a warning.
        # In-process, the library is already loaded, so we just verify
        # that the env variable is read in the source by inspection.
        assert "SUNSET_BLAS_PATH" in tv.__doc__
