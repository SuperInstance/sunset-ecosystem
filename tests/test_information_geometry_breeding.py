"""Tests for InformationGeometryBreeding — natural gradient breeding.

Run: python3 -m pytest tests/test_information_geometry_breeding.py -v --tb=short
"""
from __future__ import annotations

import numpy as np
import pytest

from swarm.information_geometry_breeding import (
    InformationGeometryBreeder,
    FisherMetric,
    natural_gradient_step,
    fisher_rao_distance,
    fisher_information_gaussian,
)


# ── FisherMetric ────────────────────────────────────────────

class TestFisherMetric:
    def test_natural_gradient_vs_euclidean(self):
        theta = np.array([1.0, 2.0])
        # Anisotropic Fisher: more sensitive to first parameter
        I = np.diag([10.0, 1.0])
        metric = FisherMetric(theta=theta, fisher_matrix=I)
        euclidean_grad = np.array([1.0, 1.0])
        nat_grad = metric.natural_gradient(euclidean_grad)
        # Natural gradient should be smaller in direction of high Fisher
        assert abs(nat_grad[0]) < abs(nat_grad[1])

    def test_local_distance(self):
        metric = FisherMetric(
            theta=np.zeros(2),
            fisher_matrix=np.eye(2),
        )
        # With identity Fisher, local distance = Euclidean
        d = metric.local_distance(np.array([3.0, 4.0]))
        assert d == pytest.approx(5.0)

    def test_symmetry_enforced(self):
        I = np.array([[1, 2], [3, 4]], dtype=np.float64)
        metric = FisherMetric(theta=np.zeros(2), fisher_matrix=I)
        assert np.allclose(metric.fisher_matrix, metric.fisher_matrix.T)


# ── Natural Gradient Step ─────────────────────────────────

class TestNaturalGradientStep:
    def test_moves_uphill(self):
        theta = np.array([0.0, 0.0])
        grad = np.array([1.0, 0.0])
        fisher_fn = lambda t: np.eye(2)
        new_theta = natural_gradient_step(theta, grad, fisher_fn, step_size=0.1)
        assert new_theta[0] > theta[0]  # moved in direction of gradient

    def test_step_size_scales_movement(self):
        theta = np.array([0.0, 0.0])
        grad = np.array([1.0, 0.0])
        fisher_fn = lambda t: np.eye(2)
        small = natural_gradient_step(theta, grad, fisher_fn, step_size=0.1)
        large = natural_gradient_step(theta, grad, fisher_fn, step_size=0.5)
        assert abs(large[0]) > abs(small[0])

    def test_damping_prevents_instability(self):
        theta = np.array([0.0, 0.0])
        grad = np.array([1.0, 1.0])
        # Near-singular Fisher
        fisher_fn = lambda t: np.array([[1e-8, 0], [0, 1e-8]])
        # With zero damping, this would explode
        # With damping=1e-4, should be stable
        new_theta = natural_gradient_step(theta, grad, fisher_fn, step_size=1.0, damping=1e-4)
        assert np.all(np.isfinite(new_theta))


# ── Fisher-Rao Distance ───────────────────────────────────

class TestFisherRaoDistance:
    def test_symmetry(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        fisher_fn = lambda t: np.eye(2)
        d_ab = fisher_rao_distance(a, b, fisher_fn, num_steps=20)
        d_ba = fisher_rao_distance(b, a, fisher_fn, num_steps=20)
        assert d_ab == pytest.approx(d_ba, rel=0.01)

    def test_zero_when_same(self):
        a = np.array([1.0, 2.0])
        fisher_fn = lambda t: np.eye(2)
        d = fisher_rao_distance(a, a, fisher_fn, num_steps=10)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_increases_with_separation(self):
        origin = np.array([0.0, 0.0])
        fisher_fn = lambda t: np.eye(2)
        d1 = fisher_rao_distance(origin, np.array([1.0, 0.0]), fisher_fn, num_steps=20)
        d2 = fisher_rao_distance(origin, np.array([2.0, 0.0]), fisher_fn, num_steps=20)
        assert d2 > d1

    def test_fisher_information_gaussian(self):
        mean = np.array([0.0, 0.0])
        cov = np.diag([2.0, 0.5])
        I = fisher_information_gaussian(mean, cov)
        # I(μ) = Σ⁻¹
        expected = np.diag([0.5, 2.0])
        assert np.allclose(I, expected, atol=1e-5)


# ── InformationGeometryBreeder ──────────────────────────────

class TestInformationGeometryBreeder:
    def test_mutate_moves_toward_gradient(self):
        breeder = InformationGeometryBreeder(dim=2)
        parent = np.array([0.0, 0.0])
        grad = np.array([1.0, 0.0])  # fitness increases in +x
        child = breeder.mutate(parent, grad)
        assert child[0] > parent[0]

    def test_mutate_batch(self):
        breeder = InformationGeometryBreeder(dim=2)
        parents = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        grads = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        children = breeder.mutate_batch(parents, grads)
        assert len(children) == 2
        assert children[0][0] > parents[0][0]
        assert children[1][1] > parents[1][1]

    def test_geodesic_crossover_interpolation(self):
        breeder = InformationGeometryBreeder(dim=2)
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 2.0])
        child = breeder.geodesic_crossover(a, b, alpha=0.5)
        assert np.allclose(child, np.array([1.0, 1.0]))

    def test_diversity_matrix_symmetric(self):
        breeder = InformationGeometryBreeder(dim=2)
        pop = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        D = breeder.diversity_matrix(pop)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T)
        assert np.all(np.diag(D) == pytest.approx(0.0))

    def test_novelty_score_increases_with_isolation(self):
        breeder = InformationGeometryBreeder(dim=2)
        pop = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
        # Novelty of point near cluster = low
        near = breeder.novelty_score(np.array([0.1, 0.0]), pop, k=1)
        # Novelty of far point = high
        far = breeder.novelty_score(np.array([10.0, 10.0]), pop, k=1)
        assert far > near

    def test_compute_metrics(self):
        breeder = InformationGeometryBreeder(dim=2)
        pop = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])]
        metrics = breeder.compute_metrics(pop)
        assert metrics["n"] == 3
        assert metrics["mean_fisher_rao"] > 0
        assert metrics["diversity_volume"] > 0
