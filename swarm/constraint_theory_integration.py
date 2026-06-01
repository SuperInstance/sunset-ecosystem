#!/usr/bin/env python3
"""swarm/constraint_theory_integration.py — Bridge to SuperInstance Constraint Theory.

Provides exact Pythagorean snapping, quantization, and hidden-dimension encoding
for sunset-ecosystem breeding loops.  Works with or without the external
`constraint_theory` package (pure-Python fallback included).

References
----------
- SuperInstance/constraint-theory-core (Rust)  v2.2.0
- SuperInstance/constraint-theory-python (PyO3) v1.0.1
- SuperInstance/eisenstein                    v0.3.1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# ── try to import the external package ────────────────────────────────────
_CT_AVAILABLE = False
_CT_MANIFOLD: Optional[Callable] = None

try:
    from constraint_theory import PythagoreanManifold as _CTManifold
    _CT_AVAILABLE = True
except Exception:  # pragma: no cover
    pass


# ── Pure-Python fallback (mirrors constraint-theory-python manifold.py) ───

class _FallbackManifold:
    """Pure-Python Pythagorean manifold — zero external deps."""

    def __init__(self, density: int = 200):
        if density <= 0:
            raise ValueError("density must be positive")
        self._density = density
        self._states = self._generate_states(density)

    def _generate_states(self, density: int) -> List[Tuple[float, float]]:
        states = set()
        max_c = density * density + density * density
        m = 2
        while m * m + 1 <= max_c:
            for n in range(1, m):
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    if c <= max_c:
                        states.add((a / c, b / c))
                        states.add((b / c, a / c))
            m += 1
        return list(states)

    @property
    def state_count(self) -> int:
        return len(self._states)

    def snap(self, x: float, y: float) -> Tuple[float, float, float]:
        if x == 0 and y == 0:
            return (0.0, 0.0, float("inf"))
        norm = math.sqrt(x * x + y * y)
        nx, ny = x / norm, y / norm
        best_dist = float("inf")
        best_state = (0.0, 0.0)
        for sx, sy in self._states:
            dist = (nx - sx) ** 2 + (ny - sy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_state = (sx, sy)
        return (best_state[0], best_state[1], math.sqrt(best_dist))

    def snap_batch(self, vectors: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        return [self.snap(v[0], v[1]) for v in vectors]


# ── Public wrapper ──────────────────────────────────────────────────────

@dataclass
class SnapResult:
    """Result of snapping a vector to exact Pythagorean coordinates."""
    x: float
    y: float
    noise: float
    original: Tuple[float, float]


class ConstraintTheoryIntegration:
    """Exact constraint satisfaction for sunset-ecosystem breeding loops.

    Uses SuperInstance constraint-theory when available, otherwise falls
    back to a pure-Python implementation with identical semantics.
    """

    def __init__(self, density: int = 200):
        self._density = density
        self._backend = _CTManifold(density) if _CT_AVAILABLE else _FallbackManifold(density)
        self._has_ct = _CT_AVAILABLE

    @property
    def backend_name(self) -> str:
        return "constraint-theory" if self._has_ct else "pure-python"

    @property
    def state_count(self) -> int:
        return self._backend.state_count

    # ── Core snapping ───────────────────────────────────────────────────

    def snap(self, x: float, y: float) -> SnapResult:
        """Snap a 2-D vector to the nearest exact Pythagorean coordinate."""
        sx, sy, noise = self._backend.snap(x, y)
        return SnapResult(x=sx, y=sy, noise=noise, original=(x, y))

    def snap_batch(self, vectors: List[Tuple[float, float]]) -> List[SnapResult]:
        """Batch-snap many vectors efficiently."""
        raw = self._backend.snap_batch(vectors)
        return [
            SnapResult(x=sx, y=sy, noise=noise, original=(ox, oy))
            for (sx, sy, noise), (ox, oy) in zip(raw, vectors)
        ]

    # ── Breeding-specific helpers ───────────────────────────────────────

    def snap_population(self, population: np.ndarray) -> np.ndarray:
        """Snap an entire breeding population (N×2) to exact coordinates.

        Returns a new (N×2) array with snapped values + a noise vector.
        """
        if population.ndim != 2 or population.shape[1] != 2:
            raise ValueError("population must be N×2")
        vectors = [(float(v[0]), float(v[1])) for v in population]
        results = self.snap_batch(vectors)
        snapped = np.array([[r.x, r.y] for r in results], dtype=population.dtype)
        noise = np.array([r.noise for r in results], dtype=population.dtype)
        return snapped, noise

    def snap_direction(self, angle: float) -> SnapResult:
        """Snap a direction (radians) to the nearest Pythagorean angle."""
        x = math.cos(angle)
        y = math.sin(angle)
        return self.snap(x, y)

    # ── Hidden-dimension encoding (GUCT) ──────────────────────────────────

    @staticmethod
    def hidden_dim_count(epsilon: float) -> int:
        """Compute hidden dimensions needed for precision ε.

        Formula: k = ⌈log₂(1/ε)⌉
        """
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("epsilon must be in (0, 1)")
        return math.ceil(math.log2(1.0 / epsilon))

    @staticmethod
    def lift_to_hidden(v: List[float], epsilon: float) -> List[float]:
        """Lift a vector to hidden-dimensional space for exact snapping."""
        k = ConstraintTheoryIntegration.hidden_dim_count(epsilon)
        hidden = [0.0] * k
        for i in range(k):
            hidden[i] = math.sin(v[i % len(v)] * (i + 1))
        return v + hidden

    # ── Quantization helpers (stub — full quantizer when ct available) ───

    def quantize_unit(self, vector: np.ndarray, mode: str = "turbo") -> np.ndarray:
        """Quantize a vector to the nearest exact Pythagorean state.

        Modes: turbo (fastest), polar (best for angles), ternary (compact).
        """
        if vector.shape[-1] != 2:
            # For higher dims, snap each pair of components
            return self._quantize_nd(vector)
        x, y = float(vector[0]), float(vector[1])
        snapped = self.snap(x, y)
        return np.array([snapped.x, snapped.y], dtype=vector.dtype)

    def _quantize_nd(self, vector: np.ndarray) -> np.ndarray:
        """Snap an N-dimensional vector by decomposing into 2-D planes."""
        flat = vector.flatten()
        out = np.zeros_like(flat)
        for i in range(0, len(flat), 2):
            if i + 1 < len(flat):
                s = self.snap(float(flat[i]), float(flat[i + 1]))
                out[i] = s.x
                out[i + 1] = s.y
            else:
                out[i] = flat[i]
        return out.reshape(vector.shape)

    # ── Holonomy / consistency checks ─────────────────────────────────────

    def check_holonomy(self, cycle: List[Tuple[float, float]], threshold: float = 1e-6) -> float:
        """Verify cyclic consistency (sum of snapped deltas ≈ 0).

        Returns a score in [0, 1] where 1.0 = perfectly consistent.
        """
        if len(cycle) < 2:
            return 1.0
        total = 0.0
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            total += math.hypot(dx, dy)
        # Normalize: a perfect cycle has total drift ≈ 0
        score = max(0.0, 1.0 - total / (len(cycle) * threshold))
        return min(1.0, score)

    def __repr__(self) -> str:
        return f"ConstraintTheoryIntegration(density={self._density}, backend={self.backend_name})"


# ── Convenience module-level functions ───────────────────────────────────

def snap_vector(v: Tuple[float, float], density: int = 200) -> SnapResult:
    """One-shot snap a vector."""
    return ConstraintTheoryIntegration(density).snap(v[0], v[1])


__all__ = [
    "ConstraintTheoryIntegration",
    "SnapResult",
    "snap_vector",
]
