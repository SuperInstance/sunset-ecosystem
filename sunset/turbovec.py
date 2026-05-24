"""Turbovec wrapper — loads BLAS eagerly and exports cblas_sgemm.

Problem
-------
The ``turbovec`` native wheel may not declare a dynamic dependency on
libopenblas, so ``cblas_sgemm`` is unresolved at runtime on systems
where the symbol is not already in the process address space. This
causes a silent hang or an ``ImportError`` when the C extension is
first exercised.

Solution
--------
1. Probe the most common BLAS library paths and ``LD_PRELOAD`` them
   into the process via ``ctypes.CDLL`` with ``RTLD_GLOBAL``.
2. Expose ``cblas_sgemm`` as a typed Python function so callers can
   use it directly (matrix multiplication, similarity scores, etc.).
3. Re-export ``IdMapIndex`` and ``TurboQuantIndex`` from the
   upstream ``turbovec`` package.

Usage::

    from sunset.turbovec import cblas_sgemm, IdMapIndex

Environment variables
---------------------
SUNSET_BLAS_PATH
    Override the BLAS ``.so`` path. If set, no probing is performed.
SUNSET_BLAS_PRELOAD
    Comma-separated list of extra paths to try *before* the defaults.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "cblas_sgemm",
    "IdMapIndex",
    "TurboQuantIndex",
    "_blas_lib",  # exposed for debugging / introspection
]

# ── BLAS library discovery ──────────────────────────────────────────

_DEFAULT_BLAS_PATHS = [
    # Debian/Ubuntu — pthread flavour (most common)
    "/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so",
    # Debian/Ubuntu — generic SONAME
    "/usr/lib/x86_64-linux-gnu/libopenblas.so.0",
    "/lib/x86_64-linux-gnu/libopenblas.so.0",
    # Generic SONAME lookups
    "libopenblas.so.0",
    "libopenblas.so",
    "libopenblasp-r0.3.26.so",
    "libblas.so.3",
    "libblas.so",
    # Intel MKL (rare for this fleet, but possible)
    "libmkl_rt.so",
    "/opt/intel/mkl/lib/intel64/libmkl_rt.so",
]


def _find_blas() -> ctypes.CDLL | None:
    """Load a BLAS library with RTLD_GLOBAL so symbols are visible."""

    # User override — highest priority
    env_path = os.environ.get("SUNSET_BLAS_PATH")
    if env_path:
        try:
            lib = ctypes.CDLL(env_path, mode=ctypes.RTLD_GLOBAL)
            logger.info("SUNSET_BLAS_PATH loaded: %s", env_path)
            return lib
        except OSError as exc:
            logger.warning("SUNSET_BLAS_PATH failed: %s (%s)", env_path, exc)

    # Extra user-provided paths
    env_extra = os.environ.get("SUNSET_BLAS_PRELOAD", "")
    extra_paths = [p.strip() for p in env_extra.split(",") if p.strip()]

    for path in extra_paths + _DEFAULT_BLAS_PATHS:
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            logger.debug("BLAS loaded: %s", path)
            return lib
        except OSError:
            continue

    return None


_blas_lib = _find_blas()

if _blas_lib is None:
    logger.warning(
        "No BLAS library found. turbovec may fail with undefined "
        "symbol 'cblas_sgemm'. Install libopenblas-dev or set "
        "SUNSET_BLAS_PATH to the correct .so."
    )
else:
    # Verify the critical symbol exists in whatever we loaded
    if not hasattr(_blas_lib, "cblas_sgemm"):
        # Some BLAS packages export cblas_* via a separate libcblas layer
        logger.warning(
            "BLAS loaded but 'cblas_sgemm' symbol missing (%s). "
            "Trying libcblas fallback...", _blas_lib._name
        )
        try:
            cblas = ctypes.CDLL("libcblas.so.3", mode=ctypes.RTLD_GLOBAL)
            if hasattr(cblas, "cblas_sgemm"):
                _blas_lib = cblas
                logger.info("libcblas fallback loaded.")
            else:
                logger.error(
                    "libcblas fallback also lacks cblas_sgemm."
                )
                _blas_lib = None
        except OSError:
            logger.error("libcblas fallback failed.")
            _blas_lib = None


# ── cblas_sgemm Python wrapper ────────────────────────────────────

# CBLAS enum values
_CblasRowMajor = 101
_CblasColMajor = 102
_CblasNoTrans = 111
_CblasTrans = 112
_CblasConjTrans = 113


def cblas_sgemm(
    A: np.ndarray,
    B: np.ndarray,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Single-precision general matrix multiply via CBLAS.

    Computes ``C = alpha * op(A) @ op(B) + beta * C``.

    Args:
        A: ``(M, K)`` float32 array.
        B: ``(K, N)`` float32 array.
        trans_a: Whether to transpose *A* before multiply.
        trans_b: Whether to transpose *B* before multiply.
        alpha: Scalar multiplier for the product.
        beta: Scalar multiplier for the existing *out* matrix.
        out: Optional ``(M, N)`` float32 array. Created if None.

    Returns:
        The ``(M, N)`` result matrix.

    Raises:
        RuntimeError: If no BLAS library was loaded.
    """
    if _blas_lib is None or not hasattr(_blas_lib, "cblas_sgemm"):
        raise RuntimeError(
            "cblas_sgemm unavailable — no BLAS library loaded. "
            "Install libopenblas-dev or set SUNSET_BLAS_PATH."
        )

    if A.dtype != np.float32 or B.dtype != np.float32:
        raise TypeError("cblas_sgemm requires float32 inputs")

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)

    # Determine dimensions
    m, k_a = (A.shape[1], A.shape[0]) if trans_a else A.shape
    k_b, n = (B.shape[1], B.shape[0]) if trans_b else B.shape

    if k_a != k_b:
        raise ValueError(
            f"Inner dimensions mismatch: {k_a} vs {k_b} "
            f"(A={A.shape}, B={B.shape}, trans_a={trans_a}, trans_b={trans_b})"
        )

    if out is None:
        C = np.zeros((m, n), dtype=np.float32, order="C")
    else:
        C = np.ascontiguousarray(out)
        if C.shape != (m, n):
            raise ValueError(
                f"out shape {C.shape} does not match expected ({m}, {n})"
            )
        if C.dtype != np.float32:
            raise TypeError("out must be float32")

    lda = A.shape[1] if not trans_a else A.shape[0]
    ldb = B.shape[1] if not trans_b else B.shape[0]
    ldc = C.shape[1]

    _blas_lib.cblas_sgemm(
        _CblasRowMajor,
        _CblasTrans if trans_a else _CblasNoTrans,
        _CblasTrans if trans_b else _CblasNoTrans,
        m,
        n,
        k_a,
        ctypes.c_float(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        lda,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ldb,
        ctypes.c_float(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ldc,
    )

    return C


# ── Re-export upstream turbovec symbols ───────────────────────────

# Because we loaded BLAS with RTLD_GLOBAL above, the symbol resolution
# should succeed even if the turbovec wheel was not explicitly linked.
try:
    from turbovec import IdMapIndex, TurboQuantIndex  # type: ignore[import-untyped]
except ImportError as exc:
    logger.error("turbovec package not installed: %s", exc)
    # Provide stub classes so the module at least imports
    class IdMapIndex:  # type: ignore[no-redef]
        """Stub — turbovec not installed."""
        def __init__(self, dim: int, bit_width: int = 4) -> None:  # noqa: D401
            raise RuntimeError("turbovec not installed")
    class TurboQuantIndex:  # type: ignore[no-redef]
        """Stub — turbovec not installed."""
        def __init__(self, dim: int, bit_width: int = 4) -> None:  # noqa: D401
            raise RuntimeError("turbovec not installed")

    __all__.remove("IdMapIndex")
    __all__.remove("TurboQuantIndex")
    __all__.extend(["IdMapIndex", "TurboQuantIndex"])
