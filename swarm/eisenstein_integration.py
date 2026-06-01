#!/usr/bin/env python3
"""swarm/eisenstein_integration.py — Hex-lattice snapping for breeding vectors.

Pure-Python implementation of Eisenstein hex-lattice math from
SuperInstance/eisenstein (v0.3.1).  Provides exact integer arithmetic
for 2-D direction snapping and hex-grid coordinate conversion.

References
----------
- SuperInstance/eisenstein  v0.3.1  (Rust, no_std)
- swarm/superinstance_ffi.py        (ctypes bridge to compiled .so)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Try to load the compiled Rust FFI for acceleration
_E12_FFI_AVAILABLE = False

try:
    from swarm.superinstance_ffi import eisenstein_norm as _ffi_norm
    _E12_FFI_AVAILABLE = True
except Exception:
    _ffi_norm = None


# ── Pure-Python E12 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class E12:
    """Eisenstein integer a + b·ω  where ω = e^(2πi/3)."""
    a: int
    b: int

    def __post_init__(self):
        # Validate they're integers
        if not isinstance(self.a, int) or not isinstance(self.b, int):
            raise TypeError("E12 coordinates must be integers")

    def norm(self) -> int:
        """N(a,b) = a² − ab + b²."""
        return self.a * self.a - self.a * self.b + self.b * self.b

    def to_cartesian(self, scale: float = 1.0) -> Tuple[float, float]:
        """Convert to 2-D Cartesian coordinates.

        ω = −½ + i·√3/2  →  a + bω = (a − b/2,  b·√3/2)
        """
        x = self.a - self.b * 0.5
        y = self.b * math.sqrt(3) / 2.0
        return (x * scale, y * scale)

    def __add__(self, other: E12) -> E12:
        return E12(self.a + other.a, self.b + other.b)

    def __sub__(self, other: E12) -> E12:
        return E12(self.a - other.a, self.b - other.b)

    def __mul__(self, other: E12) -> E12:
        # (a + bω)(c + dω) = ac + (ad+bc)ω + bdω²
        # ω² = −1 − ω  →  bdω² = −bd − bd·ω
        # Result: (ac − bd) + (ad + bc − bd)ω
        a, b, c, d = self.a, self.b, other.a, other.b
        return E12(a * c - b * d, a * d + b * c - b * d)

    def __repr__(self) -> str:
        return f"E12({self.a}, {self.b})"


# ── Hex direction snapping ─────────────────────────────────────────────────

HEX_DIRECTIONS = [
    E12(1, 0),   # 0°
    E12(1, 1),   # 60°
    E12(0, 1),   # 120°
    E12(-1, 0),  # 180°
    E12(-1, -1), # 240°
    E12(0, -1),  # 300°
]


class HexDisk:
    """Hexagonal disk for snapping directions to exact lattice points."""

    def __init__(self, radius: int = 1):
        self.radius = radius
        self._points = self._generate_disk(radius)

    def _generate_disk(self, r: int) -> List[E12]:
        points = []
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if abs(a) + abs(b) + abs(-a - b) <= 2 * r:
                    points.append(E12(a, b))
        return points

    def snap_direction(self, x: float, y: float) -> E12:
        """Snap a 2-D vector to the nearest hex-lattice point."""
        if x == 0 and y == 0:
            return E12(0, 0)
        # Convert Cartesian to axial coordinates (approximate)
        b = round(y * 2.0 / math.sqrt(3))
        a = round(x + b * 0.5)
        # Refine by checking neighbors
        best = E12(int(a), int(b))
        best_dist = self._distance_sq(x, y, best)
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                candidate = E12(int(a) + da, int(b) + db)
                d = self._distance_sq(x, y, candidate)
                if d < best_dist:
                    best_dist = d
                    best = candidate
        return best

    def _distance_sq(self, x: float, y: float, e: E12) -> float:
        cx, cy = e.to_cartesian()
        return (x - cx) ** 2 + (y - cy) ** 2

    def neighbors(self, e: E12) -> List[E12]:
        """Return the six axial neighbors of a hex point."""
        return [e + d for d in HEX_DIRECTIONS]

    def ring(self, e: E12, distance: int) -> List[E12]:
        """Return all points at exactly `distance` steps from e."""
        if distance == 0:
            return [e]
        # Start at distance steps in one direction, then walk around
        ring_points = []
        current = e + E12(-distance, distance)  # 240° direction
        for direction in HEX_DIRECTIONS:
            for _ in range(distance):
                ring_points.append(current)
                current = current + direction
        return ring_points

    def distance(self, a: E12, b: E12) -> int:
        """Hex distance (minimum steps) between two points."""
        diff = a - b
        return max(abs(diff.a), abs(diff.b), abs(diff.a - diff.b))

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return f"HexDisk(radius={self.radius}, points={len(self)})"


# ── Module-level helpers ──────────────────────────────────────────────────

def eisenstein_norm(a: int, b: int) -> int:
    """N(a,b) = a² − ab + b²."""
    if _E12_FFI_AVAILABLE and _ffi_norm is not None:
        return _ffi_norm(a, b)
    return E12(a, b).norm()


def snap_from_angle(angle: float) -> E12:
    """Snap a polar angle (radians) to the nearest hex direction."""
    # Normalize to [0, 2π)
    angle = angle % (2 * math.pi)
    # Six hex directions at 0°, 60°, 120°, 180°, 240°, 300°
    sector = round(angle / (math.pi / 3)) % 6
    return HEX_DIRECTIONS[sector]


def hex_to_cartesian(a: int, b: int, scale: float = 1.0) -> Tuple[float, float]:
    """Convert hex axial coordinates to Cartesian."""
    return E12(a, b).to_cartesian(scale)


def cartesian_to_hex(x: float, y: float) -> E12:
    """Convert Cartesian to hex axial coordinates (rounded)."""
    return HexDisk().snap_direction(x, y)


__all__ = [
    "E12",
    "HexDisk",
    "eisenstein_norm",
    "snap_from_angle",
    "hex_to_cartesian",
    "cartesian_to_hex",
    "HEX_DIRECTIONS",
]
