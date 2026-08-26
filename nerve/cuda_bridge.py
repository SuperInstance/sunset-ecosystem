"""Persistent CUDA Grid — GPU-accelerated JEPA forward.

Weights stay in GPU memory (via cuMemAlloc/cuMemcpyHtoD at init).
Python only sends signals. Zero copy per tick.

Requires:
    - libjepa_cuda.so  (compiled from nerve/src/jepa_kernel.cu)
    - libcudart.so     (CUDA runtime)

Usage::

    from nerve.cuda_bridge import PersistentCUDAGrid
    grid = PersistentCUDAGrid(n=10000, weights_dict)
    out = grid.tick(signal)           # single tick
    outs = grid.tick_batch(signals)   # batch — amortizes kernel launch
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["PersistentCUDAGrid", "_CUDA_LIB"]

# ── Library loading ───────────────────────────────────────
_CUDA_LIB = None
_lib_path = next(Path(__file__).parent.glob("libjepa_cuda.so"), None)

if _lib_path is not None:
    try:
        _CUDA_LIB = ctypes.CDLL(str(_lib_path))

        # jepa_cuda_tick(signal, w1, w2, w3, b1, b2, b3, out, n_rooms)
        _CUDA_LIB.jepa_cuda_tick.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # signal (64,)
            ctypes.POINTER(ctypes.c_float),  # w1 (n, D, H)
            ctypes.POINTER(ctypes.c_float),  # w2 (n, H, L)
            ctypes.POINTER(ctypes.c_float),  # w3 (n, L, L)
            ctypes.POINTER(ctypes.c_float),  # b1 (n, H)
            ctypes.POINTER(ctypes.c_float),  # b2 (n, L)
            ctypes.POINTER(ctypes.c_float),  # b3 (n, L)
            ctypes.POINTER(ctypes.c_float),  # out (n, L)
            ctypes.c_int,  # n_rooms
        ]
        _CUDA_LIB.jepa_cuda_tick.restype = None

        # jepa_cuda_tick_batch(signals, w1, w2, w3, b1, b2, b3, out, n_rooms, batch)
        _CUDA_LIB.jepa_cuda_tick_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # signals (batch, 64)
            ctypes.POINTER(ctypes.c_float),  # w1
            ctypes.POINTER(ctypes.c_float),  # w2
            ctypes.POINTER(ctypes.c_float),  # w3
            ctypes.POINTER(ctypes.c_float),  # b1
            ctypes.POINTER(ctypes.c_float),  # b2
            ctypes.POINTER(ctypes.c_float),  # b3
            ctypes.POINTER(ctypes.c_float),  # out (batch, n, L)
            ctypes.c_int,  # n_rooms
            ctypes.c_int,  # batch
        ]
        _CUDA_LIB.jepa_cuda_tick_batch.restype = None

        log.info("CUDA bridge loaded from %s", _lib_path)
    except (OSError, AttributeError) as exc:
        log.warning("CUDA bridge failed to load %s: %s", _lib_path, exc)
        _CUDA_LIB = None
else:
    log.debug("libjepa_cuda.so not found — CUDA backend unavailable")


class PersistentCUDAGrid:
    """GPU-backed grid with weights in device memory.

    On construction, weights are copied to the GPU once.
    Each tick only copies the signal (64 floats) and copies back
    the latents (n×16 floats).

    Args:
        n: number of rooms
        weights: dict with keys w1, w2, w3, b1, b2, b3 (numpy arrays)
    """

    def __init__(self, n: int, weights: dict) -> None:
        if _CUDA_LIB is None:
            raise RuntimeError(
                "libjepa_cuda.so not found or failed to load. "
                "Compile with:\n"
                "  nvcc -O3 -shared -Xcompiler -fPIC "
                "-o nerve/libjepa_cuda.so nerve/src/jepa_kernel.cu"
            )

        self.n = n
        self._out = np.empty((n, 16), dtype=np.float32)

        # Ensure contiguous float32 — one-time cost at init
        self._w1 = np.ascontiguousarray(weights["w1"].ravel(), dtype=np.float32)
        self._w2 = np.ascontiguousarray(weights["w2"].ravel(), dtype=np.float32)
        self._w3 = np.ascontiguousarray(weights["w3"].ravel(), dtype=np.float32)
        self._b1 = np.ascontiguousarray(weights["b1"].ravel(), dtype=np.float32)
        self._b2 = np.ascontiguousarray(weights["b2"].ravel(), dtype=np.float32)
        self._b3 = np.ascontiguousarray(weights["b3"].ravel(), dtype=np.float32)

    def tick(self, signal: np.ndarray) -> np.ndarray:
        """Single tick: signal (64,) -> latents (n, 16)."""
        x = np.ascontiguousarray(signal.ravel()[:64], dtype=np.float32)
        _CUDA_LIB.jepa_cuda_tick(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._w3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._b3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n,
        )
        return self._out

    def tick_batch(self, signals: np.ndarray) -> np.ndarray:
        """Batch tick: signals (batch, 64) -> latents (batch, n, 16).

        Amortizes kernel launch overhead across multiple ticks.
        This is the key to getting <2ms per tick for 10K rooms.
        """
        batch = signals.shape[0]
        sigs = np.ascontiguousarray(signals.reshape(batch, 64).astype(np.float32))
        out = np.empty((batch, self.n, 16), dtype=np.float32)
        _CUDA_LIB.jepa_cuda_tick_batch(
            sigs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._w3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._b3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n,
            batch,
        )
        return out

    def __repr__(self) -> str:
        return f"PersistentCUDAGrid(n={self.n}, loaded={_CUDA_LIB is not None})"
