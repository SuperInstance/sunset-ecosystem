#!/usr/bin/env python3
"""tests/test_eisenstein_snap.py — Eisenstein integer correctness suite.

Four gates:
1. Arithmetic: (1+ω) + (2+3ω) = correct
2. Snap lattice: weights move closer to lattice after snap
3. Mutation shape: output shape matches input
4. Roundtrip: snap is idempotent (snap twice = snap once)

Plus: unit verification, norm checks, compression stats.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.eisenstein_snap import (
    EisensteinInteger,
    snap_weights_to_eisenstein,
    eisenstein_mutation,
    compression_stats,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

RNG = np.random.default_rng(2026)


# ── Test 1: Eisenstein Arithmetic ────────────────────────────────────────

class TestEisensteinArithmetic:
    def test_addition(self):
        """(1 + 0ω) + (2 + 3ω) = (3 + 3ω)"""
        a = EisensteinInteger(1, 0)
        b = EisensteinInteger(2, 3)
        c = a + b
        assert c == EisensteinInteger(3, 3)

    def test_multiplication(self):
        """ω · ω = ω² = -1 - ω."""
        omega = EisensteinInteger(0, 1)
        omega2 = omega * omega
        assert omega2 == EisensteinInteger(-1, -1)

    def test_omega_cubed_is_one(self):
        """ω³ = 1."""
        omega = EisensteinInteger(0, 1)
        one = omega * omega * omega
        assert one == EisensteinInteger(1, 0)

    def test_complex_roundtrip(self):
        """from_complex(to_complex(z)) ≈ z for lattice points."""
        e = EisensteinInteger(3, -2)
        z = e.to_complex()
        e2 = EisensteinInteger.from_complex(z)
        assert e == e2

    def test_to_complex_values(self):
        """Specific known values."""
        assert EisensteinInteger(1, 0).to_complex() == 1 + 0j
        # ω = -1/2 + i·√3/2
        omega = EisensteinInteger(0, 1).to_complex()
        assert omega.real == pytest.approx(-0.5)
        assert omega.imag == pytest.approx(np.sqrt(3) / 2)


# ── Test 2: Unit Verification ───────────────────────────────────────────

class TestUnits:
    def test_six_units(self):
        """There are exactly six units with norm 1."""
        units = EisensteinInteger.units()
        assert len(units) == 6
        for u in units:
            assert u.is_unit(), f"{u} has norm {u.norm()}"

    def test_unit_norms_all_one(self):
        """Every unit has norm exactly 1."""
        for u in EisensteinInteger.units():
            assert u.norm() == pytest.approx(1.0)

    def test_non_unit_norm(self):
        """A non-unit has norm > 1."""
        e = EisensteinInteger(2, 0)
        assert e.norm() == 4.0


# ── Test 3: Snap to Lattice ────────────────────────────────────────────

class TestSnapLattice:
    def test_snap_reduces_distance(self):
        """After snap, the weight is closer to some lattice point."""
        w = np.array([0.55], dtype=np.float32)
        snapped = snap_weights_to_eisenstein(w, scale=1.0)
        # 0.55 should snap to 1.0 (the nearest integer on real axis)
        assert snapped[0] == pytest.approx(1.0, abs=0.1)

    def test_snap_idempotent(self):
        """snap(snap(w)) ≈ snap(w)."""
        w = RNG.standard_normal(100).astype(np.float32)
        s1 = snap_weights_to_eisenstein(w, scale=8.0)
        s2 = snap_weights_to_eisenstein(s1, scale=8.0)
        np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-6)

    def test_snap_compression(self):
        """Snap reduces the number of unique values."""
        w = RNG.standard_normal(500).astype(np.float32)
        snapped = snap_weights_to_eisenstein(w, scale=4.0)
        stats = compression_stats(w, snapped)
        assert stats["compression_ratio"] > 1.0
        assert stats["mean_error"] < 0.2  # Reasonable for scale=4


# ── Test 4: Mutation Shape ─────────────────────────────────────────────

class TestMutationShape:
    def test_shape_preserved(self):
        """eisenstein_mutation preserves shape."""
        shapes = [(10,), (3, 4), (2, 3, 4)]
        for shape in shapes:
            w = RNG.standard_normal(shape).astype(np.float32)
            mutated = eisenstein_mutation(w, mutation_rate=0.5)
            assert mutated.shape == shape

    def test_mutation_rate_zero(self):
        """mutation_rate=0 → no change."""
        w = RNG.standard_normal(50).astype(np.float32)
        mutated = eisenstein_mutation(w, mutation_rate=0.0)
        np.testing.assert_array_equal(w, mutated)

    def test_mutation_rate_one(self):
        """mutation_rate=1.0 → all weights snapped."""
        w = RNG.standard_normal(50).astype(np.float32)
        mutated = eisenstein_mutation(w, mutation_rate=1.0, scale=8.0)
        # All values should be on the lattice after snap
        snapped = snap_weights_to_eisenstein(w, scale=8.0)
        np.testing.assert_allclose(mutated, snapped, rtol=1e-5)


# ── Test 5: Compression Stats ──────────────────────────────────────────

class TestCompressionStats:
    def test_stats_keys(self):
        """compression_stats returns expected keys."""
        w = np.array([1.0, 1.1, 1.2, 2.0], dtype=np.float32)
        s = snap_weights_to_eisenstein(w, scale=2.0)
        stats = compression_stats(w, s)
        for key in ["mean_error", "max_error", "unique_before", "unique_after", "compression_ratio"]:
            assert key in stats

    def test_high_scale_high_compression(self):
        """Smaller scale → coarser lattice → more compression."""
        w = RNG.standard_normal(200).astype(np.float32)
        s_small = snap_weights_to_eisenstein(w, scale=2.0)
        s_large = snap_weights_to_eisenstein(w, scale=32.0)
        stats_small = compression_stats(w, s_small)
        stats_large = compression_stats(w, s_large)
        assert stats_small["compression_ratio"] > stats_large["compression_ratio"]


# ── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_weight(self):
        """Zero snaps to zero."""
        w = np.array([0.0], dtype=np.float32)
        snapped = snap_weights_to_eisenstein(w, scale=8.0)
        assert snapped[0] == pytest.approx(0.0, abs=1e-5)

    def test_empty_array(self):
        """Empty array stays empty."""
        w = np.array([], dtype=np.float32)
        snapped = snap_weights_to_eisenstein(w, scale=8.0)
        assert snapped.shape == (0,)
