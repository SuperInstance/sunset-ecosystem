"""swarm/exact_qd_archive.py — Exact QD archive using Pythagorean lattice dimensions.

MAP-Elites archive where behavior characterization uses Pythagorean triples
as exact dimensions. Each cell in the archive is indexed by a Pythagorean
triple, giving exact (not approximate) behavioral coordinates.

Usage
-----
    from swarm.exact_qd_archive import ExactQDArchive, PythagoreanQDIndex

    # 2D archive with Pythagorean behavior dimensions
    archive = ExactQDArchive(
        dimensions=[(3,4,5), (5,12,13)],  # Two behavior axes
        resolution=5  # 5 bins per axis
    )

    # Add genome with behavior vector
    behavior = [0.6, 0.8]  # Will snap to (3,4,5)
    archive.add(genome, behavior, fitness=1.5)

    # Get coverage and QD-score
    print(archive.coverage)
    print(archive.qd_score)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import math
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from swarm.pythagorean_evolution import PythagoreanGenome, PythagoreanTriple
from swarm.constraint_bridge import ConstraintBridge


@dataclass
class PythagoreanQDIndex:
    """Index into QD archive using Pythagorean triples as dimensions."""

    triples: List[PythagoreanTriple]
    resolution: int = 5

    def __post_init__(self):
        self._bin_edges = {}
        for triple in self.triples:
            angle = triple.angle()
            # Create bins around this angle
            half_width = math.pi / self.resolution
            edges = []
            for i in range(self.resolution + 1):
                bin_angle = angle - half_width + (2 * half_width * i / self.resolution)
                edges.append(bin_angle)
            self._bin_edges[(triple.a, triple.b, triple.c)] = edges

    def index(self, behavior_vector: List[float]) -> Tuple[int, ...]:
        """Map behavior vector to exact Pythagorean bin indices."""
        if len(behavior_vector) != len(self.triples):
            raise ValueError(
                f"Behavior vector dim {len(behavior_vector)} != triples {len(self.triples)}"
            )

        indices = []
        for i, triple in enumerate(self.triples):
            # Use the i-th component as the target angle (or first component if only 1D)
            if len(behavior_vector) == 1:
                target_angle = behavior_vector[0]
            else:
                target_angle = (
                    math.atan2(behavior_vector[1], behavior_vector[0])
                    if i < 2
                    else behavior_vector[i]
                    if i < len(behavior_vector)
                    else 0.0
                )
            # Find bin
            edges = self._bin_edges[(triple.a, triple.b, triple.c)]
            for j in range(len(edges) - 1):
                if edges[j] <= target_angle < edges[j + 1]:
                    indices.append(j)
                    break
            else:
                indices.append(self.resolution - 1)
        return tuple(indices)

    def get_bin_center(self, indices: Tuple[int, ...]) -> List[float]:
        """Get the center of a bin as a behavior vector."""
        centers = []
        for i, triple in enumerate(self.triples):
            edges = self._bin_edges[(triple.a, triple.b, triple.c)]
            idx = min(indices[i], len(edges) - 2)
            center_angle = (edges[idx] + edges[idx + 1]) / 2
            centers.extend([math.cos(center_angle), math.sin(center_angle)])
        return centers[: len(self.triples)]


@dataclass
class ExactQDArchive:
    """MAP-Elites archive with Pythagorean lattice dimensions."""

    dimensions: List[Tuple[int, int, int]]  # Pythagorean triples as axes
    resolution: int = 5
    max_genomes_per_bin: int = 1

    _index: PythagoreanQDIndex = field(init=False, repr=False)
    _archive: Dict[Tuple[int, ...], List[PythagoreanGenome]] = field(
        default_factory=dict, repr=False
    )
    _fitnesses: Dict[Tuple[int, ...], List[float]] = field(
        default_factory=dict, repr=False
    )
    _total_added: int = 0

    def __post_init__(self):
        triples = [PythagoreanTriple(a, b, c) for a, b, c in self.dimensions]
        self._index = PythagoreanQDIndex(triples=triples, resolution=self.resolution)

    def add(
        self, genome: PythagoreanGenome, behavior: List[float], fitness: float
    ) -> bool:
        """Add a genome to the archive."""
        idx = self._index.index(behavior)
        if idx not in self._archive:
            self._archive[idx] = []
            self._fitnesses[idx] = []

        # Keep only the best per bin (or top N)
        self._archive[idx].append(genome)
        self._fitnesses[idx].append(fitness)

        # Sort by fitness and keep top N
        combined = list(zip(self._fitnesses[idx], self._archive[idx]))
        combined.sort(reverse=True)
        self._fitnesses[idx] = [f for f, _ in combined[: self.max_genomes_per_bin]]
        self._archive[idx] = [g for _, g in combined[: self.max_genomes_per_bin]]

        self._total_added += 1
        return True

    def get(self, indices: Tuple[int, ...]) -> Optional[PythagoreanGenome]:
        """Get the best genome at a bin."""
        if indices not in self._archive:
            return None
        return self._archive[indices][0]

    def sample(self) -> Optional[Tuple[PythagoreanGenome, float]]:
        """Sample a random genome from the archive."""
        if not self._archive:
            return None
        idx = random.choice(list(self._archive.keys()))
        genome = self._archive[idx][0]
        fitness = self._fitnesses[idx][0]
        return genome, fitness

    @property
    def coverage(self) -> float:
        """Fraction of non-empty bins."""
        total_bins = self.resolution ** len(self.dimensions)
        return len(self._archive) / total_bins if total_bins > 0 else 0.0

    @property
    def qd_score(self) -> float:
        """Quality diversity score: sum of fitnesses across all bins."""
        return sum(sum(fitnesses) for fitnesses in self._fitnesses.values())

    @property
    def num_bins(self) -> int:
        return len(self._archive)

    def get_stats(self) -> dict:
        return {
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "num_bins": self.num_bins,
            "total_added": self._total_added,
            "dimensions": len(self.dimensions),
            "resolution": self.resolution,
        }

    def get_all_genomes(self) -> List[PythagoreanGenome]:
        """Get all genomes in the archive."""
        genomes = []
        for bin_genomes in self._archive.values():
            genomes.extend(bin_genomes)
        return genomes

    def get_elites(self, n: int = 10) -> List[Tuple[PythagoreanGenome, float]]:
        """Get top N genomes across all bins."""
        all_genomes = []
        for idx, fitnesses in self._fitnesses.items():
            for i, fitness in enumerate(fitnesses):
                all_genomes.append((self._archive[idx][i], fitness))
        all_genomes.sort(key=lambda x: x[1], reverse=True)
        return all_genomes[:n]

    def __len__(self) -> int:
        return sum(len(g) for g in self._archive.values())
