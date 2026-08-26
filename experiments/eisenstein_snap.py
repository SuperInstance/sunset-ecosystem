#!/usr/bin/env python3
"""eisenstein_snap.py — Hexagonal lattice quantization for neural weights.

Eisenstein integers are complex numbers a + bω where ω = e^(2πi/3) = -1/2 + i√3/2.
They form a hexagonal lattice — the densest packing in 2D. Weight snap maps
continuous weights to the nearest lattice point, achieving ~2-4× value compression
with minimal accuracy loss.

Reference: The six units are ±1, ±ω, ±ω² where ω² = ω̄ = -1/2 - i√3/2.
The scout found the bug: (1+ω) = -ω², not an independent unit.

Design eye: M.C. Escher meets Gilbert Strang — hexagons everywhere, but the math
is unforgiving.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ── Eisenstein Integer Class ───────────────────────────────────────────────


class EisensteinInteger:
    """An Eisenstein integer: a + b·ω where ω = e^(2πi/3).

    The six units of the Eisenstein ring are:
        ±1, ±ω, ±ω²
    where ω² = -1 - ω = e^(4πi/3) = ω̄ (complex conjugate).
    """

    # ω = e^(2πi/3) = -1/2 + i·√3/2
    OMEGA_REAL = -0.5
    OMEGA_IMAG = math.sqrt(3.0) / 2.0

    # ω² = e^(4πi/3) = -1/2 - i·√3/2 = ω̄
    OMEGA2_REAL = -0.5
    OMEGA2_IMAG = -math.sqrt(3.0) / 2.0

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)

    def to_complex(self) -> complex:
        """Return as a Python complex number."""
        return complex(
            self.a + self.b * self.OMEGA_REAL,
            self.b * self.OMEGA_IMAG,
        )

    @classmethod
    def from_complex(cls, z: complex) -> "EisensteinInteger":
        """Convert a complex number to the nearest Eisenstein integer.

        Given z = x + iy, we solve for a, b in:
            x = a + b·(-1/2)  →  a = x + b/2
            y = b·(√3/2)      →  b = 2y/√3

        Then round a and b to integers.
        """
        b = round(2.0 * z.imag / math.sqrt(3.0))
        a = round(z.real - b * cls.OMEGA_REAL)
        return cls(a, b)

    def __add__(self, other: "EisensteinInteger") -> "EisensteinInteger":
        return EisensteinInteger(self.a + other.a, self.b + other.b)

    def __mul__(self, other: "EisensteinInteger") -> "EisensteinInteger":
        """(a + bω)(c + dω) = ac + (ad + bc)ω + bd·ω²
        = (ac - bd) + (ad + bc - bd)ω   [since ω² = -1 - ω]
        """
        ac = self.a * other.a
        bd = self.b * other.b
        ad_bc = self.a * other.b + self.b * other.a
        return EisensteinInteger(ac - bd, ad_bc - bd)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EisensteinInteger):
            return NotImplemented
        return self.a == other.a and self.b == other.b

    def __repr__(self) -> str:
        return f"EisensteinInteger({self.a}, {self.b})"

    def norm(self) -> float:
        """Norm: |a + bω|² = a² - ab + b²."""
        return self.a * self.a - self.a * self.b + self.b * self.b

    def is_unit(self) -> bool:
        """True if this is one of the six units (norm == 1)."""
        return self.norm() == 1.0

    @classmethod
    def units(cls) -> Tuple["EisensteinInteger", ...]:
        """Return the six units: ±1, ±ω, ±ω²."""
        return (
            cls(1, 0),  #  1
            cls(-1, 0),  # -1
            cls(0, 1),  #  ω
            cls(0, -1),  # -ω
            cls(-1, -1),  #  ω²  (since -1 - ω = ω²)
            cls(1, 1),  # -ω²
        )


# ── Weight Snap ──────────────────────────────────────────────────────────


def snap_weights_to_eisenstein(
    weights: np.ndarray,
    scale: float = 16.0,
) -> np.ndarray:
    """Snap continuous weights to the nearest Eisenstein lattice points.

    Steps:
        1. Scale weights up by *scale*
        2. Map each scaled value to the nearest Eisenstein integer (as complex)
        3. Map back to real line (take real part — weights are real-valued)
        4. Scale down by *scale*

    This achieves compression because many distinct float values collapse to
    the same lattice point after rounding.
    """
    w_scaled = weights.astype(np.float64) * scale

    # For each element, treat it as a complex number on the real axis
    # and snap to the nearest Eisenstein lattice point
    snapped = np.zeros_like(w_scaled)
    for idx in np.ndindex(w_scaled.shape):
        z = complex(w_scaled[idx], 0.0)
        e = EisensteinInteger.from_complex(z)
        # Map back to real: take the real component of the Eisenstein integer
        snapped[idx] = e.to_complex().real

    return (snapped / scale).astype(np.float32)


def eisenstein_mutation(
    weights: np.ndarray,
    mutation_rate: float = 0.1,
    scale: float = 16.0,
) -> np.ndarray:
    """Apply Eisenstein snap mutation to a random subset of weights.

    Unlike Gaussian mutation (adds continuous noise), this snaps selected
    weights to lattice points — the mutation is discrete and structured.
    """
    mutated = weights.copy()
    mask = np.random.rand(*weights.shape) < mutation_rate
    mutated[mask] = snap_weights_to_eisenstein(mutated[mask], scale=scale)
    return mutated


# ── Compression Stats ──────────────────────────────────────────────────


def compression_stats(weights: np.ndarray, snapped: np.ndarray) -> dict:
    """Return quantization error and effective compression ratio."""
    unique_before = len(np.unique(weights.round(decimals=6)))
    unique_after = len(np.unique(snapped.round(decimals=6)))
    return {
        "mean_error": float(np.abs(weights - snapped).mean()),
        "max_error": float(np.abs(weights - snapped).max()),
        "unique_before": unique_before,
        "unique_after": unique_after,
        "compression_ratio": unique_before / max(unique_after, 1),
    }


# ── CLI / Quick Test ───────────────────────────────────────────────────


def _demo() -> None:
    rng = np.random.default_rng(42)
    w = rng.standard_normal(1000).astype(np.float32)

    snapped = snap_weights_to_eisenstein(w, scale=8.0)
    stats = compression_stats(w, snapped)

    print(f"Mean quantization error: {stats['mean_error']:.6f}")
    print(f"Max quantization error:  {stats['max_error']:.6f}")
    print(f"Unique values before:    {stats['unique_before']}")
    print(f"Unique values after:     {stats['unique_after']}")
    print(f"Effective compression:   {stats['compression_ratio']:.1f}×")

    # Verify units
    units = EisensteinInteger.units()
    print(f"\nSix units: {units}")
    print(f"All have norm 1: {[u.norm() for u in units]}")


if __name__ == "__main__":
    _demo()
