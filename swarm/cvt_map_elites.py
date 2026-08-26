"""CVT_MAP_Elites — Centroidal Voronoi Tessellation MAP-Elites.

Next-generation Quality Diversity archive using Voronoi tessellation
instead of a fixed grid. Provides:
1. Better coverage of behavior space (non-uniform, adaptive cells)
2. Graceful handling of high-dimensional behavior descriptors
3. Natural emergence of cell density matching fitness landscape

Mathematical foundation:
- Voronoi diagram: cell i = {x | d(x, c_i) ≤ d(x, c_j) ∀ j}
- Centroidal property: c_i = centroid(Voronoi cell i)
- Lloyd relaxation: iteratively move centroids to cell means
- k-means++ initialization for good spread

Reference: Vassiliades et al. (2018) "Discovering the Elite Hypervolume
by Leveraging Interspecies Correlation"
"""

from __future__ import annotations

__all__ = [
    "CVTMAPArchive",
    "lloyd_relaxation",
    "kmeans_plus_plus",
]

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── k-means++ initialization ────────────────────────────────


def kmeans_plus_plus(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """k-means++ centroid initialization for better spread.

    1. Pick first centroid uniformly at random.
    2. For each point, compute D(x)² = min distance to existing centroids.
    3. Pick next centroid with prob ∝ D(x)².
    """
    rng = rng or np.random.default_rng()
    n, d = points.shape
    centroids = np.zeros((k, d), dtype=np.float64)

    # First centroid: random
    centroids[0] = points[rng.integers(n)]

    for i in range(1, k):
        # Squared distances to nearest centroid
        dists = (
            np.min(
                np.linalg.norm(points[:, None, :] - centroids[:i][None, :, :], axis=2),
                axis=1,
            )
            ** 2
        )
        # Probabilities
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = points[idx]

    return centroids


# ── Lloyd relaxation ────────────────────────────────────────


def lloyd_relaxation(
    points: np.ndarray,
    centroids: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm: move centroids to cell means until convergence.

    Returns (centroids, labels).
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    k = centroids.shape[0]

    for _ in range(max_iters):
        # Assign to nearest centroid
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Recompute centroids as cell means
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centroids[i] = points[mask].mean(axis=0)
            else:
                # Empty cell: reinitialize to random point
                new_centroids[i] = points[np.random.randint(len(points))]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    # Final assignment
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return centroids, labels


# ── CVT MAP-Elites Archive ──────────────────────────────────


@dataclass
class CVTArchiveEntry:
    """Single entry in a CVT archive cell."""

    solution: np.ndarray
    fitness: float
    behavior: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class CVTMAPArchive:
    """CVT MAP-Elites archive with Voronoi tessellation.

    Replaces the fixed grid of MAP-Elites with adaptive Voronoi cells
    that better match the structure of the behavior space.
    """

    def __init__(
        self,
        n_cells: int,
        behavior_bounds: list[tuple[float, float]],
        behavior_samples: int = 100_000,
        lloyd_iters: int = 100,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_cells = n_cells
        self.behavior_dims = len(behavior_bounds)
        self.behavior_bounds = behavior_bounds
        self.rng = rng or np.random.default_rng()

        # Generate behavior space samples for CVT construction
        samples = self._sample_behavior_space(behavior_samples)

        # k-means++ init
        centroids = kmeans_plus_plus(samples, n_cells, self.rng)

        # Lloyd relaxation
        self.centroids, _ = lloyd_relaxation(samples, centroids, max_iters=lloyd_iters)

        # Archive: cell_idx -> best entry
        self._archive: dict[int, CVTArchiveEntry] = {}

        # History for analysis
        self._add_count = 0
        self._replace_count = 0

    def _sample_behavior_space(self, n: int) -> np.ndarray:
        """Uniform random samples in behavior space."""
        samples = np.zeros((n, self.behavior_dims), dtype=np.float64)
        for d, (lo, hi) in enumerate(self.behavior_bounds):
            samples[:, d] = self.rng.uniform(lo, hi, size=n)
        return samples

    # ── archive operations ────────────────────────────────

    def _cell_index(self, behavior: np.ndarray) -> int:
        """Find Voronoi cell containing behavior descriptor."""
        b = np.asarray(behavior, dtype=np.float64)
        dists = np.linalg.norm(self.centroids - b, axis=1)
        return int(np.argmin(dists))

    def add(self, solution: np.ndarray, fitness: float, behavior: np.ndarray) -> bool:
        """Add to archive. Returns True if added or improved."""
        cell = self._cell_index(behavior)
        self._add_count += 1

        if cell not in self._archive or fitness > self._archive[cell].fitness:
            self._archive[cell] = CVTArchiveEntry(
                solution=np.asarray(solution, dtype=np.float64),
                fitness=float(fitness),
                behavior=np.asarray(behavior, dtype=np.float64),
            )
            if cell in self._archive:
                self._replace_count += 1
            return True
        return False

    def get(self, cell: int) -> CVTArchiveEntry | None:
        """Get entry for a specific cell."""
        return self._archive.get(cell)

    def sample(self, n: int = 1) -> list[CVTArchiveEntry]:
        """Sample n random cells with contents."""
        filled = list(self._archive.values())
        if not filled:
            return []
        indices = self.rng.choice(len(filled), size=min(n, len(filled)), replace=False)
        return [filled[i] for i in indices]

    # ── properties ──────────────────────────────────────────

    @property
    def coverage(self) -> float:
        """Fraction of cells that are occupied."""
        return len(self._archive) / self.n_cells

    @property
    def qd_score(self) -> float:
        """Sum of fitnesses across all occupied cells."""
        return sum(e.fitness for e in self._archive.values())

    @property
    def max_fitness(self) -> float:
        if not self._archive:
            return 0.0
        return max(e.fitness for e in self._archive.values())

    @property
    def mean_fitness(self) -> float:
        if not self._archive:
            return 0.0
        return sum(e.fitness for e in self._archive.values()) / len(self._archive)

    # ── analysis ────────────────────────────────────────────

    def grid_report(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "n_cells": self.n_cells,
            "occupied": len(self._archive),
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "max_fitness": self.max_fitness,
            "mean_fitness": self.mean_fitness,
            "add_count": self._add_count,
            "replace_count": self._replace_count,
        }

    def behavior_density(self, cell_radius: float = 0.1) -> np.ndarray:
        """Local density of occupied cells around each centroid."""
        densities = np.zeros(self.n_cells)
        for i, c_i in enumerate(self.centroids):
            count = 0
            for j, c_j in enumerate(self.centroids):
                if i != j and j in self._archive:
                    if np.linalg.norm(c_i - c_j) < cell_radius:
                        count += 1
            densities[i] = count
        return densities

    def elite_hypervolume(self) -> float:
        """Approximate hypervolume of occupied cells (rectangle method).

        Each cell contributes a hyper-rectangle around its centroid.
        """
        # Compute average cell size from centroid distances
        if len(self._archive) < 2:
            return 0.0

        # Average nearest-neighbor distance per dimension
        nn_dists = []
        for i in range(self.n_cells):
            if i not in self._archive:
                continue
            dists = [
                np.linalg.norm(self.centroids[i] - self.centroids[j])
                for j in range(self.n_cells)
                if i != j and j in self._archive
            ]
            if dists:
                nn_dists.append(min(dists))

        if not nn_dists:
            return 0.0

        avg_side = np.mean(nn_dists)
        cell_volume = avg_side**self.behavior_dims
        return len(self._archive) * cell_volume
