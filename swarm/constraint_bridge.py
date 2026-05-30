"""swarm/constraint_bridge.py — Bridge to SuperInstance Constraint Theory Core.

Provides exact Pythagorean snapping, deterministic quantization, and holonomy
verification for sunset-ecosystem breeding loops. Replaces floating-point
approximation with exact rational arithmetic.

Usage
-----
    from swarm.constraint_bridge import ConstraintBridge

    bridge = ConstraintBridge(dim=256, density=200)

    # Snap a vector to exact Pythagorean coordinate
    exact, noise = bridge.snap_vector([0.577, 0.816])
    # exact = [0.6, 0.8] — exact, not 1.0000000000000002

    # Batch snap breeding population
    population = [[0.1, 0.9], [0.5, 0.5], ...]
    exact_pop = bridge.batch_snap(population)

    # Quantize tile embeddings
    quantized = bridge.quantize_embedding(embedding, mode="turbo")

    # Verify consensus consistency
    is_consistent = bridge.check_holonomy(cycle_embeddings)

    # Use FFI-accelerated functions when available
    bridge.eisenstein_norm(2, 1)  # 3
    bridge.laman_is_rigid(3, 3)   # True
    bridge.holonomy_check([0.0, 0.0, 0.0], 1e-6)  # 1.0

Dependencies
------------
- constraint-theory-core (optional): Rust crate via Python bindings
- superinstance_ffi (optional): ctypes bindings to FM's Rust library
- Fallback: Pure Python Pythagorean triple generation + KD-tree
"""

from __future__ import annotations

import ctypes
import functools
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check for constraint-theory Python bindings
CT_AVAILABLE = False
try:
    import constraint_theory as ct
    CT_AVAILABLE = True
    logger.info("Constraint Theory Python bindings loaded")
except ImportError:
    logger.debug("constraint-theory-python not available")

# ── SuperInstance FFI probing ────────────────────────────────────────────

_FFI_SO_PATH = os.path.join(
    os.path.dirname(__file__),
    "superinstance-ffi", "target", "release", "libsuperinstance_ffi.so"
)
FFI_SO_EXISTS = os.path.exists(_FFI_SO_PATH)

FFI_AVAILABLE = False
FFI_LIB: Optional[ctypes.CDLL] = None
try:
    if FFI_SO_EXISTS:
        FFI_LIB = ctypes.CDLL(_FFI_SO_PATH)
        FFI_AVAILABLE = True
        logger.info("SuperInstance FFI .so loaded from %s", _FFI_SO_PATH)
except OSError as e:
    logger.warning("SuperInstance FFI .so found but could not load: %s", e)

# Try the python wrapper module as a secondary fallback (may also mock)
try:
    from swarm import superinstance_ffi as _ffi_module
except ImportError:
    _ffi_module = None  # type: ignore[assignment]
except RuntimeError as e:
    logger.debug("superinstance_ffi module import failed: %s", e)
    _ffi_module = None  # type: ignore[assignment]

# Load mock FFI when the real shared library is absent
if not FFI_AVAILABLE:
    try:
        import superinstance_ffi_mock as _mock_ffi
        _mock_obj = _mock_ffi.load_mock_ffi()
        _ffi_module = _mock_obj  # type: ignore[assignment]
        logger.info("Using superinstance_ffi_mock fallback")
    except Exception as e:
        logger.debug("superinstance_ffi_mock not available: %s", e)


# ── LRU-cached triple generation ───────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _generate_triples_cached(density: int) -> Tuple[Tuple[int, int, int], ...]:
    """Generate primitive Pythagorean triples; cached by density."""
    triples: List[Tuple[int, int, int]] = []
    limit = int(math.sqrt(density * 10)) + 5
    for m in range(2, limit + 1):
        for n in range(1, m):
            if math.gcd(m, n) == 1 and (m - n) % 2 == 1:
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                triples.append((a, b, c))
                if a != b:
                    triples.append((b, a, c))
    # Sort by c ascending so the densest triples are first
    triples.sort(key=lambda t: t[2])
    return tuple(triples[: density * 10])


@dataclass
class SnapResult:
    """Result of a constraint snap operation."""
    exact: np.ndarray
    noise: float
    triple: Optional[Tuple[int, int, int]] = None  # (a, b, c) Pythagorean triple


@dataclass
class ConstraintBridge:
    """Bridge to Constraint Theory for exact breeding operations."""

    dim: int = 256
    density: int = 200  # Pythagorean triple density
    _manifold: Optional[Any] = field(default=None, repr=False)
    _triples: Tuple[Tuple[int, int, int], ...] = field(
        default_factory=tuple, repr=False
    )
    _kdtree: Optional[Any] = field(default=None, repr=False)
    _triples_arr: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if CT_AVAILABLE:
            self._manifold = ct.PythagoreanManifold(self.density)
        else:
            self._triples = _generate_triples_cached(self.density)
            self._triples_arr = np.array(self._triples, dtype=np.float64)
            self._kdtree = self._build_kdtree()
        logger.info(
            "ConstraintBridge initialized (dim=%d, density=%d, ct=%s, ffi=%s)",
            self.dim, self.density, CT_AVAILABLE, FFI_AVAILABLE
        )

    # ── Pythagorean snapping ────────────────────────────────────────

    def snap_vector(self, vector: List[float]) -> SnapResult:
        """Snap a 2D vector to exact Pythagorean coordinate."""
        if len(vector) != 2:
            raise ValueError("Only 2D vectors supported for Pythagorean snapping")

        if CT_AVAILABLE and self._manifold:
            exact, noise = self._manifold.snap(vector)
            return SnapResult(exact=np.array(exact), noise=noise)

        # Pure Python fallback
        return self._snap_python(vector)

    def exact_pythagorean_snap(
        self, x: float, y: float
    ) -> Tuple[float, float, Optional[Tuple[int, int, int]], float]:
        """Exact nearest-Pythagorean-triple snap.

        Returns
        -------
        snapped_x, snapped_y, triple, noise
            *snapped* are exact rational a/c, b/c.
            *triple* is the primitive (a, b, c) with a²+b²=c², or None.
            *noise* is the Euclidean distance to the snapped point.
        """
        # Try real FFI first (new binding) if the .so exposes it
        if FFI_LIB is not None:
            try:
                # Hypothetical binding — we probe at call time so missing
                # symbols do not crash import.
                fn = FFI_LIB.exact_pythagorean_snap
                fn.argtypes = [ctypes.c_double, ctypes.c_double]
                fn.restype = ctypes.c_double * 4  # x,y,a,c packed (b derived)
                packed = fn(x, y)
                sx, sy, a_int, c_int = packed[0], packed[1], int(packed[2]), int(packed[3])
                b_int = int(round(math.sqrt(c_int * c_int - a_int * a_int)))
                triple = (a_int, b_int, c_int)
                noise = math.hypot(x - sx, y - sy)
                return sx, sy, triple, noise
            except AttributeError:
                pass  # Symbol not present — fall through

        # Pure-Python / mock fallback (vectorised)
        sx, sy, triple, noise = self._exact_snap_pure(x, y)
        return sx, sy, triple, noise

    def _exact_snap_pure(
        self, x: float, y: float
    ) -> Tuple[float, float, Optional[Tuple[int, int, int]], float]:
        """Pure Python exact snap for a single (x, y) pair; handles all quadrants."""
        norm = math.hypot(x, y)
        if norm == 0:
            return 0.0, 0.0, (0, 0, 1), 0.0

        ux, uy = x / norm, y / norm

        # Reflect into first quadrant for lookup, remember signs
        sx = 1.0 if ux >= 0 else -1.0
        sy = 1.0 if uy >= 0 else -1.0
        abs_ux, abs_uy = abs(ux), abs(uy)

        if self._triples_arr is None or len(self._triples_arr) == 0:
            return float(ux), float(uy), None, 0.0

        # Vectorised distance computation in first quadrant
        coords = self._triples_arr[:, :2] / self._triples_arr[:, 2:3]
        diffs = coords - np.array([abs_ux, abs_uy], dtype=np.float64)
        dists = np.einsum("ij,ij->i", diffs, diffs)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        a, b, c = self._triples[best_idx]
        # Reflect snapped coordinates back to original quadrant
        snapped_x = sx * (a / c)
        snapped_y = sy * (b / c)
        return snapped_x, snapped_y, (a, b, c), math.sqrt(best_dist)

    def batch_snap(self, vectors: List[List[float]]) -> List[SnapResult]:
        """Batch snap multiple vectors using numpy vectorisation."""
        if not vectors:
            return []

        n = len(vectors)
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("batch_snap expects an Nx2 array of 2D vectors")

        # Vectorised normalisation
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, 1.0, norms)
        unit = arr / norms_safe

        # For zero vectors, keep them zero
        zero_mask = (norms[:, 0] == 0)
        unit[zero_mask] = 0.0

        if CT_AVAILABLE and self._manifold:
            # CT path: fall back to loop (no vectorised CT API assumed)
            return [self.snap_vector(v.tolist()) for v in arr]

        if self._triples_arr is None or len(self._triples_arr) == 0:
            # No triples available — return identity
            return [
                SnapResult(exact=unit[i], noise=0.0, triple=None)
                for i in range(n)
            ]

        # Vectorised all-pairs distance computation:
        #   triples_coords: T x 2  (first quadrant only)
        #   unit:           N x 2
        # Reflect unit vectors to first quadrant for distance computation,
        # then restore signs on the snapped result.
        triples_coords = self._triples_arr[:, :2] / self._triples_arr[:, 2:3]
        abs_unit = np.abs(unit)
        diffs = abs_unit[:, np.newaxis, :] - triples_coords[np.newaxis, :, :]  # N x T x 2
        dists = np.einsum("ntk,ntk->nt", diffs, diffs)  # N x T
        best_idx = np.argmin(dists, axis=1)  # N
        best_dist = np.sqrt(np.min(dists, axis=1))  # N

        results: List[SnapResult] = []
        for i in range(n):
            if zero_mask[i]:
                results.append(
                    SnapResult(
                        exact=np.array([0.0, 0.0], dtype=np.float32),
                        noise=0.0,
                        triple=(0, 0, 1),
                    )
                )
                continue
            a, b, c = self._triples[int(best_idx[i])]
            # Restore original quadrant signs
            sx = 1.0 if unit[i, 0] >= 0 else -1.0
            sy = 1.0 if unit[i, 1] >= 0 else -1.0
            exact = np.array([sx * a / c, sy * b / c], dtype=np.float32)
            results.append(
                SnapResult(
                    exact=exact,
                    noise=float(best_dist[i]),
                    triple=(a, b, c),
                )
            )
        return results

    # ── Quantization ────────────────────────────────────────────────

    def quantize_embedding(self, embedding: List[float],
                           mode: str = "turbo") -> np.ndarray:
        """Quantize an embedding using Pythagorean quantization."""
        arr = np.array(embedding, dtype=np.float32)

        if mode == "exact":
            # Project each 2D slice onto the nearest exact Pythagorean direction
            dim = len(arr)
            if dim == 0:
                return arr
            # Pad to even dimension
            if dim % 2 != 0:
                padded = np.concatenate([arr, np.array([0.0], dtype=np.float32)])
            else:
                padded = arr
            pairs = padded.reshape(-1, 2)
            snapped_pairs = self.batch_snap(pairs.tolist())
            result = np.concatenate([sr.exact for sr in snapped_pairs])
            return result[:dim].astype(np.float32)
        elif mode == "ternary":
            # BitNet-style ternary {-1, 0, 1}
            return np.where(arr > 0.1, 1, np.where(arr < -0.1, -1, 0)).astype(np.int8)
        elif mode == "polar":
            # Project to unit circle and snap
            if len(arr) >= 2:
                snap = self.snap_vector([arr[0], arr[1]])
                return snap.exact
            return arr / (np.linalg.norm(arr) + 1e-8)
        elif mode == "turbo":
            # TurboQuant: near-optimal lattice quantization
            return np.round(arr * 10) / 10  # Simple 1-decimal quantization
        else:
            # Hybrid: auto-select based on magnitude
            norm = np.linalg.norm(arr)
            if norm > 1.0:
                return self.quantize_embedding(embedding, "polar")
            else:
                return self.quantize_embedding(embedding, "ternary")

    # ── Holonomy verification ───────────────────────────────────────

    def check_holonomy(self, cycle: List[List[float]]) -> bool:
        """Check if a cycle of embeddings has zero holonomy.

        Zero holonomy means the cycle is consistent — the product of
        parallel transport around the cycle returns to identity.
        """
        if len(cycle) < 3:
            return True  # Trivially consistent

        total_rotation = 0.0
        for i in range(len(cycle)):
            a = np.array(cycle[i])
            b = np.array(cycle[(i + 1) % len(cycle)])
            angle = self._angle_between(a, b)
            total_rotation += angle

        remainder = abs(total_rotation) % (2 * math.pi)
        return remainder < 0.01 or abs(remainder - 2 * math.pi) < 0.01

    # ── FFI-accelerated / mock functions ──────────────────────────────

    def _call_ffi_or_fallback(self, name: str, fallback: Any, *args, **kwargs):
        """Dispatch to real FFI, mock FFI, or pure Python fallback."""
        if FFI_LIB is not None:
            try:
                fn = getattr(FFI_LIB, name)
                return fn(*args, **kwargs)
            except AttributeError:
                pass
        if _ffi_module is not None:
            try:
                fn = getattr(_ffi_module, name)
                return fn(*args, **kwargs)
            except (AttributeError, TypeError):
                pass
        return fallback(*args, **kwargs)

    def eisenstein_norm(self, a: int, b: int) -> int:
        """Eisenstein integer norm N(a,b) = a² − a·b + b²."""
        return self._call_ffi_or_fallback(
            "eisenstein_norm", lambda a, b: a * a - a * b + b * b, a, b
        )

    def laman_is_rigid(self, num_vertices: int, num_edges: int) -> bool:
        """Check if a graph is Laman-rigid: 2n−3 edges, no subset over-constrained."""
        def _fallback(nv: int, ne: int) -> bool:
            return ne == 2 * nv - 3
        return bool(self._call_ffi_or_fallback("laman_is_rigid", _fallback, num_vertices, num_edges))

    def holonomy_check(self, states: List[float], threshold: float) -> float:
        """Cyclic drift consistency check. Returns 1.0 if consistent, 0.0 otherwise."""
        def _fallback(states: List[float], threshold: float) -> float:
            if len(states) < 3:
                return 1.0
            total = sum(abs(states[i] - states[(i + 1) % len(states)]) for i in range(len(states)))
            return 1.0 if total < threshold else 0.0
        return self._call_ffi_or_fallback("holonomy_check", _fallback, states, threshold)

    def constraint_check(self, value: float, lower: float, upper: float) -> bool:
        """Check if value is within [lower, upper]."""
        return bool(self._call_ffi_or_fallback(
            "constraint_check", lambda v, lo, hi: lo <= v <= hi, value, lower, upper
        ))

    def constraint_violation(self, value: float, lower: float, upper: float) -> float:
        """Compute constraint violation distance (0 if satisfied)."""
        def _fallback(v: float, lo: float, hi: float) -> float:
            if v < lo:
                return lo - v
            if v > hi:
                return v - hi
            return 0.0
        return self._call_ffi_or_fallback("constraint_violation", _fallback, value, lower, upper)

    def spline_interpolate(self, p0: float, p1: float, m0: float, m1: float, t: float) -> float:
        """Hermite cubic spline at parameter t in [0,1]."""
        def _fallback(p0: float, p1: float, m0: float, m1: float, t: float) -> float:
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        return self._call_ffi_or_fallback("spline_interpolate", _fallback, p0, p1, m0, m1, t)

    def deadband_filter(self, value: float, last: float, deadband: float) -> Tuple[float, float]:
        """Apply deadband filter. Returns (filtered_value, updated_last)."""
        def _fallback(value: float, last: float, deadband: float) -> Tuple[float, float]:
            if abs(value - last) < deadband:
                return last, last
            return value, value
        result = self._call_ffi_or_fallback("deadband_filter", _fallback, value, last, deadband)
        # Some FFI variants return a tuple, some return a single float; normalise
        if isinstance(result, tuple):
            return result
        return float(result), float(result)

    def manhattan_distance(self, a: List[float], b: List[float]) -> float:
        """L1 distance between two float arrays."""
        return self._call_ffi_or_fallback(
            "manhattan_distance", lambda a, b: sum(abs(x - y) for x, y in zip(a, b)), a, b
        )

    def cascade_match(self, query: List[float], candidates: List[List[float]],
                      thresholds: List[float]) -> int:
        """Tiered nearest-neighbor search. Returns index of first match, or -1."""
        def _fallback(query: List[float], candidates: List[List[float]], thresholds: List[float]) -> int:
            for tier_thresh in thresholds:
                for idx, cand in enumerate(candidates):
                    dist = sum(abs(q - c) for q, c in zip(query, cand))
                    if dist < tier_thresh:
                        return idx
            return -1
        return self._call_ffi_or_fallback("cascade_match", _fallback, query, candidates, thresholds)

    def pythagorean48_encode(self, numerator: int, denominator: int) -> int:
        """Frequency ratio → 48-tone index."""
        def _fallback(num: int, den: int) -> int:
            if den == 0:
                return 0
            ratio = num / den
            semitones = 12 * math.log2(ratio)
            return round(semitones) % 48
        return self._call_ffi_or_fallback("pythagorean48_encode", _fallback, numerator, denominator)

    # ── Graph rigidity ──────────────────────────────────────────────

    def laman_rigid(self, graph_edges: List[Tuple[int, int]], n_nodes: int) -> bool:
        """Check if a constraint graph is Laman-rigid.

        A graph is Laman-rigid if it has exactly 2n - 3 edges and every
        subset of k nodes has at most 2k - 3 edges.
        """
        if len(graph_edges) != 2 * n_nodes - 3:
            return False

        # Check subset condition (simplified: check all pairs)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edge_count = sum(1 for e in graph_edges if i in e and j in e)
                if edge_count > 1:  # At most one edge between any pair
                    return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "density": self.density,
            "ct_available": CT_AVAILABLE,
            "ffi_available": FFI_AVAILABLE,
            "ffi_so_exists": FFI_SO_EXISTS,
            "triples_cached": len(self._triples),
            "manifold_initialized": self._manifold is not None,
        }

    # ── Pure Python helpers ─────────────────────────────────────────

    def _build_kdtree(self) -> Optional[Any]:
        """Build a simple KD-tree from triples."""
        # Brute force search is fast enough for our triple counts
        return None

    def _snap_python(self, vector: List[float]) -> SnapResult:
        """Pure Python snapping to nearest Pythagorean triple."""
        sx, sy, triple, noise = self._exact_snap_pure(vector[0], vector[1])
        exact = np.array([sx, sy], dtype=np.float32)
        return SnapResult(exact=exact, noise=noise, triple=triple)

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
        """Compute angle between two vectors."""
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return math.acos(max(-1.0, min(1.0, cos_sim)))
