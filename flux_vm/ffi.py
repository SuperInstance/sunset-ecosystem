"""Python FFI bindings for libflux_vm.so.

Wraps the C API exposed by flux-vm-v3/src/ffi.rs:
    flux_check_batch(latents, n_rooms, latent_dim,
                     min_bound, max_bound, max_l2, max_var,
                     violations) -> int

Usage::

    from flux_vm.ffi import FluxVM
    vm = FluxVM("flux_vm/libflux_vm.so")
    violations = vm.check_batch(latents, min_bound=-10.0, max_bound=10.0,
                                 max_l2=100.0, max_var=10.0)
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FluxVM:
    """Python wrapper for the FLUX VM shared library."""

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
            raise FileNotFoundError(f"libflux_vm.so not found: {p}")

        # Search common locations
        candidates = [
            Path(__file__).with_name("libflux_vm.so"),
            Path(__file__).parent / "libflux_vm.so",
            Path.cwd() / "flux_vm" / "libflux_vm.so",
            Path.cwd() / "libflux_vm.so",
        ]
        for c in candidates:
            if c.exists():
                return c

        logger.warning("libflux_vm.so not found in standard locations")
        return None

    def _load(self) -> None:
        if self._so_path is None:
            return
        try:
            self._lib = ctypes.CDLL(str(self._so_path))
            self._setup_types()
            logger.info("Loaded FLUX VM from %s", self._so_path)
        except OSError as exc:
            logger.error("Failed to load %s: %s", self._so_path, exc)
            raise

    def _setup_types(self) -> None:
        if self._lib is None:
            return
        # void flux_check_batch(float*, int, int, float, float, float, float, unsigned int*)
        self._lib.flux_check_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # latents
            ctypes.c_int,                     # n_rooms
            ctypes.c_int,                     # latent_dim
            ctypes.c_float,                   # min_bound
            ctypes.c_float,                   # max_bound
            ctypes.c_float,                   # max_l2
            ctypes.c_float,                   # max_var
            ctypes.POINTER(ctypes.c_uint),     # violations
        ]
        self._lib.flux_check_batch.restype = ctypes.c_int

    # ── Public API ──────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._lib is not None

    def check_batch(
        self,
        latents: np.ndarray,
        *,
        min_bound: float = -10.0,
        max_bound: float = 10.0,
        max_l2: float = 100.0,
        max_var: float = 10.0,
    ) -> np.ndarray:
        """Check a batch of room latents against neural bounds.

        Parameters
        ----------
        latents : np.ndarray
            (n_rooms × latent_dim) float32 array.
        min_bound, max_bound : float
            Per-dimension bounds.
        max_l2 : float
            Maximum L2 norm per room.
        max_var : float
            Maximum variance per room.

        Returns
        -------
        np.ndarray
            (n_rooms,) uint8 array — 0 = pass, 1 = fail.
        """
        if self._lib is None:
            raise RuntimeError("libflux_vm.so not loaded")

        if latents.dtype != np.float32:
            latents = latents.astype(np.float32)

        if latents.ndim != 2:
            raise ValueError(f"latents must be 2D, got {latents.ndim}D")

        n_rooms, latent_dim = latents.shape
        violations = np.zeros(n_rooms, dtype=np.uint32)

        ret = self._lib.flux_check_batch(
            latents.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_rooms),
            ctypes.c_int(latent_dim),
            ctypes.c_float(min_bound),
            ctypes.c_float(max_bound),
            ctypes.c_float(max_l2),
            ctypes.c_float(max_var),
            violations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        )

        if ret != 0:
            raise RuntimeError(f"flux_check_batch failed with code {ret}")

        return violations.astype(np.uint8)

    def __repr__(self) -> str:
        return f"FluxVM(so_path={self._so_path}, loaded={self.is_available()})"
