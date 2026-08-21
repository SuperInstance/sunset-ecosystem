#!/usr/bin/env python3
"""Tests for swarm/constraint_theory_integration.py."""

import math

import numpy as np
import pytest

from swarm.constraint_theory_integration import (
    ConstraintTheoryIntegration,
    SnapResult,
    snap_vector,
)


class TestSnapResult:
    def test_dataclass_fields(self):
        r = SnapResult(x=0.6, y=0.8, noise=0.01, original=(0.577, 0.816))
        assert r.x == 0.6
        assert r.y == 0.8
        assert r.noise == 0.01
        assert r.original == (0.577, 0.816)


class TestConstraintTheoryIntegration:
    def test_init(self):
        ct = ConstraintTheoryIntegration(density=100)
        assert ct.state_count > 0
        assert ct.backend_name in ("constraint-theory", "pure-python")

    def test_snap_exact(self):
        ct = ConstraintTheoryIntegration(density=200)
        # 0.6, 0.8 is already a Pythagorean triple (3/5, 4/5)
        r = ct.snap(0.6, 0.8)
        assert r.x == 0.6
        assert r.y == 0.8
        assert r.noise == 0.0
        assert r.original == (0.6, 0.8)

    def test_snap_approximate(self):
        ct = ConstraintTheoryIntegration(density=200)
        # 0.577, 0.816 should snap to *some* Pythagorean state close by
        r = ct.snap(0.577, 0.816)
        # The snapped point should be on the unit circle (or near it)
        mag = math.hypot(r.x, r.y)
        assert abs(mag - 1.0) < 1e-6
        # Noise should be small (close to original)
        assert r.noise < 0.1

    def test_snap_batch(self):
        ct = ConstraintTheoryIntegration(density=200)
        vectors = [(0.6, 0.8), (0.577, 0.816)]
        results = ct.snap_batch(vectors)
        assert len(results) == 2
        assert results[0].noise == 0.0
        assert results[1].noise < 0.1

    def test_snap_zero_vector(self):
        ct = ConstraintTheoryIntegration(density=200)
        r = ct.snap(0.0, 0.0)
        assert r.x == 0.0
        assert r.y == 0.0
        assert r.noise == float("inf")

    def test_snap_direction(self):
        ct = ConstraintTheoryIntegration(density=200)
        # 0 radians = (1, 0) should snap to itself or very close
        r = ct.snap_direction(0.0)
        assert abs(r.x - 1.0) < 0.05
        assert abs(r.y) < 0.05

    def test_snap_population(self):
        ct = ConstraintTheoryIntegration(density=200)
        pop = np.array([[0.6, 0.8], [0.577, 0.816]], dtype=float)
        snapped, noise = ct.snap_population(pop)
        assert snapped.shape == pop.shape
        assert noise.shape == (2,)
        assert noise[0] == 0.0
        assert noise[1] < 0.1

    def test_snap_population_wrong_shape(self):
        ct = ConstraintTheoryIntegration(density=200)
        pop = np.array([[0.6, 0.8, 0.1]])
        with pytest.raises(ValueError):
            ct.snap_population(pop)

    def test_hidden_dim_count(self):
        assert ConstraintTheoryIntegration.hidden_dim_count(0.5) == 1
        assert ConstraintTheoryIntegration.hidden_dim_count(0.25) == 2
        assert ConstraintTheoryIntegration.hidden_dim_count(0.125) == 3

    def test_hidden_dim_count_invalid(self):
        with pytest.raises(ValueError):
            ConstraintTheoryIntegration.hidden_dim_count(0.0)
        with pytest.raises(ValueError):
            ConstraintTheoryIntegration.hidden_dim_count(1.0)
        with pytest.raises(ValueError):
            ConstraintTheoryIntegration.hidden_dim_count(-0.1)

    def test_lift_to_hidden(self):
        v = [0.5, 0.5]
        lifted = ConstraintTheoryIntegration.lift_to_hidden(v, epsilon=0.25)
        assert len(lifted) == len(v) + 2  # 2 original + 2 hidden dims

    def test_quantize_unit(self):
        ct = ConstraintTheoryIntegration(density=200)
        vec = np.array([0.6, 0.8])
        q = ct.quantize_unit(vec)
        assert q.shape == vec.shape
        assert abs(q[0] - 0.6) < 0.01
        assert abs(q[1] - 0.8) < 0.01

    def test_quantize_nd(self):
        ct = ConstraintTheoryIntegration(density=200)
        vec = np.array([0.6, 0.8, 0.577, 0.816])
        q = ct.quantize_unit(vec)
        assert q.shape == vec.shape

    def test_check_holonomy_perfect(self):
        ct = ConstraintTheoryIntegration(density=200)
        # A single-point cycle is trivially consistent
        cycle = [(0.6, 0.8), (0.6, 0.8)]
        score = ct.check_holonomy(cycle, threshold=1e-3)
        assert score == 1.0

    def test_check_holonomy_inconsistent(self):
        ct = ConstraintTheoryIntegration(density=200)
        cycle = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        score = ct.check_holonomy(cycle, threshold=1e-3)
        assert score < 1.0

    def test_check_holonomy_short(self):
        ct = ConstraintTheoryIntegration(density=200)
        assert ct.check_holonomy([], threshold=1e-3) == 1.0
        assert ct.check_holonomy([(0.6, 0.8)], threshold=1e-3) == 1.0

    def test_repr(self):
        ct = ConstraintTheoryIntegration(density=200)
        assert "density=200" in repr(ct)
        assert "backend=" in repr(ct)


class TestModuleFunctions:
    def test_snap_vector(self):
        r = snap_vector((0.6, 0.8), density=200)
        assert r.x == 0.6
        assert r.y == 0.8
        assert r.noise == 0.0


class TestBackendEquivalence:
    """Ensure fallback and (if present) native backend agree."""

    def test_backends_agree(self):
        # Even if ct is available, we can test both via direct fallback
        from swarm.constraint_theory_integration import _FallbackManifold

        fb = _FallbackManifold(density=200)
        ct = ConstraintTheoryIntegration(density=200)

        vectors = [(0.6, 0.8), (0.577, 0.816), (-0.3, 0.4), (0.0, 1.0)]
        for v in vectors:
            fbx, fby, fbn = fb.snap(v[0], v[1])
            ctr = ct.snap(v[0], v[1])
            assert abs(fbx - ctr.x) < 1e-9
            assert abs(fby - ctr.y) < 1e-9
            assert abs(fbn - ctr.noise) < 1e-9
