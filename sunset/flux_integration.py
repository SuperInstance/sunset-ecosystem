"""FLUX Integration — Constraint checking for the Sunset Ecosystem.

Provides a bridge between the living grid (sunset-ecosystem) and the
FLUX formally-proven constraint checker (flux-vm-v3).

When FLUX is available:
  - Room outputs are checked against safety constraints
  - Violations increase chaos (exploration) in violating rooms
  - Breeding avoids parents that produce violating offspring

When FLUX is NOT available (development mode):
  - Constraints are enforced in pure Python (slower but no Rust dep)
  - Same API, different backend — seamless fallback

Architecture:
  RoomGrid.tick()  →  FluxConstraintChecker.check_batch()  →  FLUX VM
                              ↓
                    violations  →  chaos[violating_rooms] += 0.1

Usage:
    from sunset.flux_integration import FluxConstraintChecker
    checker = FluxConstraintChecker()

    # After grid tick:
    violations = checker.check_batch(latents, preset="neural_bounds")
    if violations.any():
        chaos[violations] += 0.1

See docs/FLUX_INTEGRATION.md for full specification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────

FLUX_VM_PATH = os.environ.get(
    "FLUX_VM_PATH", "../flux-vm-v3-temp/target/release/flux_vm"
)
FLUST_COMPILER_PATH = os.environ.get(
    "FLUX_COMPILER_PATH", "../flux-compiler-v0.1.0/compiler/fluxc.py"
)


# ── Data structures ───────────────────────────────────────


@dataclass
class ConstraintViolation:
    """A single constraint violation in a room."""

    room_idx: int
    room_id: str
    constraint_name: str
    severity: float  # 0.0-1.0
    details: str


@dataclass
class ConstraintPreset:
    """Named set of constraints with parameters."""

    name: str
    bounds: Tuple[float, float]  # (min, max) for latent values
    max_l2_norm: float
    max_variance: float
    novelty_floor: float  # Minimum novelty score


# Presets tuned for different operational modes
PRESETS: Dict[str, ConstraintPreset] = {
    "neural_bounds": ConstraintPreset(
        name="neural_bounds",
        bounds=(-10.0, 10.0),
        max_l2_norm=25.0,
        max_variance=5.0,
        novelty_floor=0.01,
    ),
    "safe_mode": ConstraintPreset(
        name="safe_mode",
        bounds=(-5.0, 5.0),
        max_l2_norm=15.0,
        max_variance=2.0,
        novelty_floor=0.05,
    ),
    "exploration": ConstraintPreset(
        name="exploration",
        bounds=(-50.0, 50.0),
        max_l2_norm=100.0,
        max_variance=20.0,
        novelty_floor=0.001,
    ),
}


# ── Backends ────────────────────────────────────────────────


class _FluxBackend:
    """Abstract base for constraint checking backends."""

    def check_batch(
        self,
        latents: np.ndarray,
        preset: ConstraintPreset,
    ) -> np.ndarray:
        """Return boolean mask of violating rooms."""
        raise NotImplementedError


class _PythonBackend(_FluxBackend):
    """Pure-Python constraint checker (no Rust dependency)."""

    def check_batch(
        self,
        latents: np.ndarray,
        preset: ConstraintPreset,
    ) -> np.ndarray:
        """Vectorized constraint checking using numpy.

        Checks:
        1. Value bounds: all latent dims within [min, max]
        2. L2 norm: per-room norm ≤ max_l2_norm
        3. Variance: per-room variance ≤ max_variance
        """
        n = latents.shape[0]
        violations = np.zeros(n, dtype=bool)

        # 1. Bounds check
        violations |= (latents < preset.bounds[0]).any(axis=1)
        violations |= (latents > preset.bounds[1]).any(axis=1)

        # 2. L2 norm check
        l2_norms = np.sqrt(np.sum(latents**2, axis=1))
        violations |= l2_norms > preset.max_l2_norm

        # 3. Variance check
        variances = np.var(latents, axis=1)
        violations |= variances > preset.max_variance

        return violations


class _RustBackend(_FluxBackend):
    """FLUX VM backend — calls into compiled Rust constraint checker."""

    def __init__(self, vm_path: str) -> None:
        self.vm_path = vm_path
        self._lib = None
        self._load()

    def _load(self) -> None:
        """Load the FLUX VM shared library via ctypes."""
        import ctypes

        if not os.path.exists(self.vm_path):
            log.warning(
                "FLUX VM not found at %s — falling back to Python", self.vm_path
            )
            return

        try:
            self._lib = ctypes.CDLL(self.vm_path)
            # Setup function signatures
            # flux_check_batch(float* latents, int n_rooms, int latent_dim,
            #                  float min_bound, float max_bound,
            #                  float max_l2, float max_var,
            #                  uint8_t* violations)
            self._lib.flux_check_batch.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_uint8),
            ]
            self._lib.flux_check_batch.restype = ctypes.c_int
            log.info("FLUX VM loaded: %s", self.vm_path)
        except OSError as exc:
            log.warning("Failed to load FLUX VM: %s", exc)
            self._lib = None

    def check_batch(
        self,
        latents: np.ndarray,
        preset: ConstraintPreset,
    ) -> np.ndarray:
        """Call FLUX VM to check constraints.

        Falls back to Python if VM is not available.
        """
        if self._lib is None:
            # Fallback to Python
            return _PythonBackend().check_batch(latents, preset)

        import ctypes

        n, d = latents.shape
        violations = np.zeros(n, dtype=np.uint8)

        self._lib.flux_check_batch(
            latents.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            d,
            preset.bounds[0],
            preset.bounds[1],
            preset.max_l2_norm,
            preset.max_variance,
            violations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )

        return violations.astype(bool)


# ── Public API ──────────────────────────────────────────────


class FluxConstraintChecker:
    """Constraint checker for room grid outputs.

    Automatically selects backend:
    1. FLUX VM (Rust) — if compiled and available
    2. Python fallback — always works

    Usage:
        checker = FluxConstraintChecker(preset="neural_bounds")
        violations = checker.check_batch(latents)
        if violations.any():
            # Increase chaos in violating rooms
            chaos[violations] += 0.1
    """

    def __init__(
        self,
        preset: str = "neural_bounds",
        vm_path: Optional[str] = None,
    ) -> None:
        self.preset = PRESETS.get(preset, PRESETS["neural_bounds"])
        self._python_backend = _PythonBackend()

        # Try Rust backend
        vm = vm_path or FLUX_VM_PATH
        if os.path.exists(vm):
            self._backend: _FluxBackend = _RustBackend(vm)
        else:
            log.info("FLUX VM not found — using Python backend")
            self._backend = self._python_backend

    def check_batch(
        self, latents: np.ndarray, preset_name: Optional[str] = None
    ) -> np.ndarray:
        """Check a batch of room latents against constraints.

        Args:
            latents: (n_rooms, latent_dim) float32 array
            preset_name: Override preset for this call only

        Returns:
            Boolean mask of violating rooms
        """
        preset = self.preset
        if preset_name:
            preset = PRESETS.get(preset_name, self.preset)

        return self._backend.check_batch(latents, preset)

    def get_violations(
        self,
        latents: np.ndarray,
        room_ids: List[str],
        preset_name: Optional[str] = None,
    ) -> List[ConstraintViolation]:
        """Get detailed violation records for each violating room.

        Args:
            latents: (n_rooms, latent_dim) float32 array
            room_ids: List of room IDs corresponding to rows
            preset_name: Optional preset override

        Returns:
            List of ConstraintViolation objects
        """
        mask = self.check_batch(latents, preset_name)
        preset = PRESETS.get(preset_name, self.preset) if preset_name else self.preset

        violations = []
        for idx in np.where(mask)[0]:
            # Determine which constraint was violated
            room = latents[idx]
            reasons = []
            if (room < preset.bounds[0]).any() or (room > preset.bounds[1]).any():
                reasons.append("bounds")
            l2 = float(np.sqrt(np.sum(room**2)))
            if l2 > preset.max_l2_norm:
                reasons.append(f"l2_norm({l2:.1f})")
            var = float(np.var(room))
            if var > preset.max_variance:
                reasons.append(f"variance({var:.1f})")

            violations.append(
                ConstraintViolation(
                    room_idx=idx,
                    room_id=room_ids[idx] if idx < len(room_ids) else f"room-{idx}",
                    constraint_name=";".join(reasons),
                    severity=min(1.0, len(reasons) / 3.0),
                    details=f"bounds={preset.bounds}, max_l2={preset.max_l2_norm}, max_var={preset.max_variance}",
                )
            )

        return violations


# ── RoomGrid integration hook ─────────────────────────────


def apply_constraint_feedback(
    grid: Any,
    checker: FluxConstraintChecker,
    chaos_increase: float = 0.1,
) -> int:
    """Apply constraint checking to a RoomGrid after tick().

    Args:
        grid: RoomGrid instance (has .latents and .chaos attributes)
        checker: FluxConstraintChecker instance
        chaos_increase: How much to increase chaos for violating rooms

    Returns:
        Number of rooms that were penalized
    """
    if not hasattr(grid, "latents") or grid.latents is None:
        return 0

    latents = grid.latents
    violations = checker.check_batch(latents)
    n_violations = int(violations.sum())

    if n_violations > 0 and hasattr(grid, "chaos"):
        # Increase chaos for violating rooms — they need to explore more
        grid.chaos[violations] += chaos_increase
        log.debug(
            "Constraint feedback: %d/%d rooms penalized", n_violations, len(latents)
        )

    return n_violations
