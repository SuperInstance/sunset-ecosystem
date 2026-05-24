"""Python FFI bindings for libjepa_kernel.so.

Wraps the Rust C API exposed by nerve/src/lib.rs:
    jepa_forward_batch(x, w1, w2, w3, b1, b2, b3, n, out)

Usage::

    from nerve.jepa_ffi import JEPAKernel
    kernel = JEPAKernel("nerve/target/release/libjepa_kernel.so")
    out = kernel.forward_batch(x, weights, biases, n_rooms)
"""
from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class JEPAKernel:
    """Python wrapper for the JEPA Rust shared library."""

    def __init__(self, so_path: Optional[str | Path] = None) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._so_path = self._resolve_so(so_path)
        if self._so_path:
            self._load()

    # ── SO discovery ────────────────────────────────────────────

    @staticmethod
    def _resolve_so(path: Optional[str | Path]) -> Optional[Path]:
        if path is not None:
            p = Path(path)
            if p.exists():
                return p
            raise FileNotFoundError(f"libjepa_kernel.so not found: {p}")

        candidates = [
            Path(__file__).with_name("libjepa_kernel.so"),
            Path(__file__).parent / "target" / "release" / "libjepa_kernel.so",
            Path(__file__).parent / "libjepa_kernel.so",
            Path.cwd() / "nerve" / "target" / "release" / "libjepa_kernel.so",
            Path.cwd() / "nerve" / "libjepa_kernel.so",
        ]
        for c in candidates:
            if c.exists():
                return c

        logger.warning("libjepa_kernel.so not found in standard locations")
        return None

    def _load(self) -> None:
        if self._so_path is None:
            return
        try:
            self._lib = ctypes.CDLL(str(self._so_path))
            self._setup_types()
            logger.info("Loaded JEPA kernel from %s", self._so_path)
        except OSError as exc:
            logger.error("Failed to load %s: %s", self._so_path, exc)
            raise

    def _setup_types(self) -> None:
        if self._lib is None:
            return
        # void jepa_forward_batch(float*, float*, float*, float*, float*, float*, float*, usize, float*)
        self._lib.jepa_forward_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # w1
            ctypes.POINTER(ctypes.c_float),  # w2
            ctypes.POINTER(ctypes.c_float),  # w3
            ctypes.POINTER(ctypes.c_float),  # b1
            ctypes.POINTER(ctypes.c_float),  # b2
            ctypes.POINTER(ctypes.c_float),  # b3
            ctypes.c_size_t,                  # n
            ctypes.POINTER(ctypes.c_float),  # out
        ]
        self._lib.jepa_forward_batch.restype = None

    # ── Public API ──────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._lib is not None

    def forward_batch(
        self,
        x: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        w3: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        b3: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Run JEPA forward pass for n rooms.

        Parameters
        ----------
        x : np.ndarray
            (64,) input signal.
        w1 : np.ndarray
            (n × 64 × 32) weights.
        w2 : np.ndarray
            (n × 32 × 16) weights.
        w3 : np.ndarray
            (n × 16 × 16) weights.
        b1, b2, b3 : np.ndarray
            Biases: (n × 32), (n × 16), (n × 16).
        n : int
            Number of rooms.

        Returns
        -------
        np.ndarray
            (n × 16,) latent outputs.
        """
        if self._lib is None:
            raise RuntimeError("libjepa_kernel.so not loaded")

        out = np.zeros(n * 16, dtype=np.float32)

        self._lib.jepa_forward_batch(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(n),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        return out.reshape(n, 16)

    def __repr__(self) -> str:
        return f"JEPAKernel(so_path={self._so_path}, loaded={self.is_available()})"
