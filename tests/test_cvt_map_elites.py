"""Tests for CVT_MAP_Elites — Centroidal Voronoi Tessellation archive.

Run: python3 -m pytest tests/test_cvt_map_elites.py -v --tb=short
"""
from __future__ import annotations

import numpy as np
import pytest

from swarm.cvt_map_elites import (
    CVTMAPArchive,
    kmeans_plus_plus,
    lloyd_relaxation,
)


# ── k-means++ ───────────────────────────────────────────────

class TestKMeansPlusPlus:
    def test_centroids_in_data(self):
        points = np.random.randn(100, 2)
        centroids = kmeans_plus_plus(points, k=5)
        assert centroids.shape == (5, 2)
        # All centroids should be actual data points
        for c in centroids:
            assert any(np.allclose(c, p) for p in points)

    def test_spread_out(self):
        # Two well-separated clusters
        rng = np.random.default_rng(42)
        cluster_a = rng.standard_normal((50, 2)) + np.array([0.0, 0.0])
        cluster_b = rng.standard_normal((50, 2)) + np.array([10.0, 10.0])
        points = np.vstack([cluster_a, cluster_b])
        centroids = kmeans_plus_plus(points, k=2, rng=rng)
        # Should pick one from each cluster (with high probability)
        dists = np.linalg.norm(centroids[0] - centroids[1])
        assert dists > 3.0


# ── Lloyd relaxation ────────────────────────────────────────

class TestLloydRelaxation:
    def test_convergence(self):
        # Two Gaussian clusters
        a = np.random.randn(50, 2) + np.array([0.0, 0.0])
        b = np.random.randn(50, 2) + np.array([5.0, 5.0])
        points = np.vstack([a, b])
        centroids = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)
        final_centroids, labels = lloyd_relaxation(points, centroids, max_iters=50)
        # Centroids should move toward cluster centers
        assert np.linalg.norm(final_centroids[0] - np.array([0.0, 0.0])) < 1.0
        assert np.linalg.norm(final_centroids[1] - np.array([5.0, 5.0])) < 1.0

    def test_labels_sum(self):
        points = np.random.randn(100, 2)
        centroids = kmeans_plus_plus(points, k=5)
        _, labels = lloyd_relaxation(points, centroids, max_iters=20)
        assert labels.shape == (100,)
        assert set(labels) <= set(range(5))


# ── CVTMAPArchive ───────────────────────────────────────────

class TestCVTMAPArchive:
    def test_initialization(self):
        archive = CVTMAPArchive(
            n_cells=10,
            behavior_bounds=[(0.0, 1.0), (0.0, 1.0)],
            behavior_samples=1000,
            lloyd_iters=10,
        )
        assert archive.n_cells == 10
        assert archive.behavior_dims == 2
        assert archive.centroids.shape == (10, 2)

    def test_add_and_retrieve(self):
        archive = CVTMAPArchive(
            n_cells=10,
            behavior_bounds=[(0.0, 1.0)],
            behavior_samples=500,
        )
        sol = np.array([1.0, 2.0])
        added = archive.add(sol, fitness=0.8, behavior=np.array([0.5]))
        assert added is True
        assert archive.coverage > 0

    def test_replace_on_better_fitness(self):
        archive = CVTMAPArchive(
            n_cells=10,
            behavior_bounds=[(0.0, 1.0)],
            behavior_samples=500,
        )
        b = np.array([0.5])
        archive.add(np.array([1.0]), fitness=0.5, behavior=b)
        archive.add(np.array([2.0]), fitness=0.7, behavior=b)
        # Should have replaced
        cell = archive._cell_index(b)
        entry = archive.get(cell)
        assert entry is not None
        assert entry.fitness == pytest.approx(0.7)
        assert np.allclose(entry.solution, np.array([2.0]))

    def test_no_replace_on_worse_fitness(self):
        archive = CVTMAPArchive(
            n_cells=10,
            behavior_bounds=[(0.0, 1.0)],
            behavior_samples=500,
        )
        b = np.array([0.5])
        archive.add(np.array([1.0]), fitness=0.8, behavior=b)
        added = archive.add(np.array([2.0]), fitness=0.6, behavior=b)
        assert added is False

    def test_coverage_increases(self):
        archive = CVTMAPArchive(
            n_cells=20,
            behavior_bounds=[(0.0, 1.0), (0.0, 1.0)],
            behavior_samples=2000,
        )
        assert archive.coverage == 0.0
        for i in range(15):
            archive.add(
                np.array([float(i)]),
                fitness=0.5 + 0.01 * i,
                behavior=np.array([i / 15.0, i / 15.0]),
            )
        assert archive.coverage > 0
        assert archive.coverage <= 1.0

    def test_qd_score(self):
        archive = CVTMAPArchive(
            n_cells=10,
            behavior_bounds=[(0.0, 1.0)],
            behavior_samples=500,
        )
        for i in range(5):
            archive.add(np.array([float(i)]), fitness=0.1 * (i + 1), behavior=np.array([i / 5.0]))
        assert archive.qd_score == pytest.approx(0.1 + 0.2 + 0.3 + 0.4 + 0.5)

    def test_sample(self):
        archive = CVTMAPArchive(
            n_cells=10,
            behavior_bounds=[(0.0, 1.0)],
            behavior_samples=500,
        )
        for i in range(5):
            archive.add(np.array([float(i)]), fitness=0.5, behavior=np.array([i / 5.0]))
        samples = archive.sample(n=3)
        assert len(samples) == 3

    def test_grid_report(self):
        archive = CVTMAPArchive(
            n_cells=10,
            behavior_bounds=[(0.0, 1.0)],
            behavior_samples=500,
        )
        archive.add(np.array([1.0]), fitness=0.8, behavior=np.array([0.5]))
        report = archive.grid_report()
        assert report["n_cells"] == 10
        assert report["occupied"] == 1
        assert report["max_fitness"] == pytest.approx(0.8)

    def test_elite_hypervolume(self):
        archive = CVTMAPArchive(
            n_cells=20,
            behavior_bounds=[(0.0, 1.0), (0.0, 1.0)],
            behavior_samples=1000,
        )
        # Add a few scattered points
        for i in range(5):
            archive.add(
                np.array([float(i)]),
                fitness=0.5,
                behavior=np.array([0.2 * i, 0.2 * i]),
            )
        hv = archive.elite_hypervolume()
        assert hv > 0

    def test_high_dimensional(self):
        archive = CVTMAPArchive(
            n_cells=50,
            behavior_bounds=[(0.0, 1.0)] * 10,  # 10D behavior space
            behavior_samples=5000,
            lloyd_iters=20,
        )
        assert archive.centroids.shape == (50, 10)
        archive.add(np.random.randn(5), fitness=0.5, behavior=np.random.rand(10))
        assert archive.coverage > 0
