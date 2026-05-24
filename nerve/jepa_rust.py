"""Persistent Rust Grid — weights stay in Rust, Python only sends signals.

This eliminates the 7× ascontiguousarray() overhead per tick.
Usage:
    from nerve.jepa_rust import PersistentGrid
    grid = PersistentGrid(n=100, weights_dict)
    out = grid.tick(signal)           # single tick — just signal + pre-allocated buffer
    outs = grid.tick_batch(signals)   # batch — amortizes FFI overhead

See nerve/src/lib.rs for the Rust implementation.
"""

from __future__ import annotations

import ctypes
import numpy as np
from pathlib import Path

# Load the shared library
_so = next(Path(__file__).parent.glob("target/release/libjepa_kernel.so"), None)
if _so is None:
    raise RuntimeError("libjepa_kernel.so not found. Run `cargo build --release` in nerve/.")

_lib = ctypes.CDLL(str(_so))

# jepa_grid_create(n, w1, w2, w3, b1, b2, b3) -> handle
_lib.jepa_grid_create.argtypes = [
    ctypes.c_size_t,                          # n
    ctypes.POINTER(ctypes.c_float),           # w1
    ctypes.POINTER(ctypes.c_float),           # w2
    ctypes.POINTER(ctypes.c_float),           # w3
    ctypes.POINTER(ctypes.c_float),           # b1
    ctypes.POINTER(ctypes.c_float),           # b2
    ctypes.POINTER(ctypes.c_float),           # b3
]
_lib.jepa_grid_create.restype = ctypes.c_void_p

# jepa_grid_tick(handle, signal, out)
_lib.jepa_grid_tick.argtypes = [
    ctypes.c_void_p,                          # handle
    ctypes.POINTER(ctypes.c_float),           # signal (64,)
    ctypes.POINTER(ctypes.c_float),           # out (n*16,)
]
_lib.jepa_grid_tick.restype = None

# jepa_grid_tick_batch(handle, signals, batch, out)
_lib.jepa_grid_tick_batch.argtypes = [
    ctypes.c_void_p,                          # handle
    ctypes.POINTER(ctypes.c_float),           # signals (batch*64,)
    ctypes.c_size_t,                          # batch
    ctypes.POINTER(ctypes.c_float),           # out (batch*n*16,)
]
_lib.jepa_grid_tick_batch.restype = None

# jepa_grid_destroy(handle)
_lib.jepa_grid_destroy.argtypes = [ctypes.c_void_p]
_lib.jepa_grid_destroy.restype = None


class PersistentGrid:
    """Rust-backed grid with persistent weights.

    Args:
        n: number of rooms
        weights: dict with keys w1, w2, w3, b1, b2, b3 (numpy arrays)
    """

    def __init__(self, n: int, weights: dict) -> None:
        self.n = n
        self._handle: ctypes.c_void_p | None = None
        self._out = np.empty((n, 16), dtype=np.float32)  # pre-allocated

        # Ensure contiguous float32 — one-time cost at init
        w1 = np.ascontiguousarray(weights["w1"].ravel(), dtype=np.float32)
        w2 = np.ascontiguousarray(weights["w2"].ravel(), dtype=np.float32)
        w3 = np.ascontiguousarray(weights["w3"].ravel(), dtype=np.float32)
        b1 = np.ascontiguousarray(weights["b1"].ravel(), dtype=np.float32)
        b2 = np.ascontiguousarray(weights["b2"].ravel(), dtype=np.float32)
        b3 = np.ascontiguousarray(weights["b3"].ravel(), dtype=np.float32)

        self._handle = _lib.jepa_grid_create(
            n,
            w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if not self._handle:
            raise RuntimeError("jepa_grid_create failed")

    def tick(self, signal: np.ndarray) -> np.ndarray:
        """Single tick: signal (64,) -> latents (n, 16).
        Zero copy: writes into pre-allocated buffer."""
        x = np.ascontiguousarray(signal.ravel()[:64], dtype=np.float32)
        _lib.jepa_grid_tick(
            self._handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return self._out

    def tick_batch(self, signals: np.ndarray) -> np.ndarray:
        """Batch tick: signals (batch, 64) -> latents (batch, n, 16).
        Amortizes FFI overhead."""
        batch = signals.shape[0]
        sigs = np.ascontiguousarray(signals.reshape(batch, 64).astype(np.float32))
        out = np.empty((batch, self.n, 16), dtype=np.float32)
        _lib.jepa_grid_tick_batch(
            self._handle,
            sigs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def __del__(self) -> None:
        if self._handle:
            _lib.jepa_grid_destroy(self._handle)
            self._handle = None

    def __repr__(self) -> str:
        return f"PersistentGrid(n={self.n}, handle={self._handle is not None})"
