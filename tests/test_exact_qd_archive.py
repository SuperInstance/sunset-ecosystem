"""Tests for Exact QD Archive — Pythagorean lattice MAP-Elites.

Covers PythagoreanQDIndex, ExactQDArchive, coverage, QD-score, and sampling.
"""

import math
import random

import pytest
import numpy as np

from swarm.exact_qd_archive import ExactQDArchive, PythagoreanQDIndex
from swarm.pythagorean_evolution import PythagoreanGenome, PythagoreanTriple


class TestPythagoreanQDIndex:
    def test_init(self):
        triples = [PythagoreanTriple(3, 4, 5)]
        index = PythagoreanQDIndex(triples=triples, resolution=5)
        assert len(index.triples) == 1
        assert index.resolution == 5

    def test_index(self):
        triples = [PythagoreanTriple(3, 4, 5)]
        index = PythagoreanQDIndex(triples=triples, resolution=5)
        # Vector with same length as triples (1D)
        idx = index.index([0.6])
        assert isinstance(idx, tuple)
        assert len(idx) == 1

    def test_index_2d(self):
        triples = [PythagoreanTriple(3, 4, 5), PythagoreanTriple(5, 12, 13)]
        index = PythagoreanQDIndex(triples=triples, resolution=5)
        idx = index.index([0.6, 0.8])
        assert len(idx) == 2

    def test_get_bin_center(self):
        triples = [PythagoreanTriple(3, 4, 5)]
        index = PythagoreanQDIndex(triples=triples, resolution=5)
        center = index.get_bin_center((2,))
        assert len(center) == 1


class TestExactQDArchive:
    def test_init(self):
        archive = ExactQDArchive(
            dimensions=[(3, 4, 5), (5, 12, 13)],
            resolution=5
        )
        assert archive.resolution == 5
        assert len(archive.dimensions) == 2

    def test_add(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        result = archive.add(genome, [0.6], fitness=1.5)
        assert result is True

    def test_get(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        archive.add(genome, [0.6], fitness=1.5)
        idx = archive._index.index([0.6])
        retrieved = archive.get(idx)
        assert retrieved is not None

    def test_sample(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        archive.add(genome, [0.6], fitness=1.5)
        result = archive.sample()
        assert result is not None
        assert isinstance(result, tuple)

    def test_coverage(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        # Add one genome
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        archive.add(genome, [0.6], fitness=1.5)
        coverage = archive.coverage
        assert coverage > 0
        assert coverage <= 1.0

    def test_qd_score(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        genome1 = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        genome2 = PythagoreanGenome(triples=[PythagoreanTriple(5, 12, 13)])
        archive.add(genome1, [0.6], fitness=1.5)
        archive.add(genome2, [0.38], fitness=2.0)
        assert archive.qd_score > 0

    def test_stats(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        archive.add(genome, [0.6], fitness=1.5)
        stats = archive.get_stats()
        assert stats["num_bins"] == 1
        assert stats["total_added"] == 1
        assert stats["dimensions"] == 1

    def test_get_elites(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        for i in range(5):
            genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
            archive.add(genome, [0.6], fitness=float(i))
        elites = archive.get_elites(n=3)
        assert len(elites) <= 3
        assert elites[0][1] == 4.0  # Highest fitness

    def test_multiple_bins(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        # Add genomes with different behaviors to fill different bins
        for angle in [0.0, 0.5, 1.0, 1.5, 2.0]:
            genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
            behavior = [math.cos(angle)]
            archive.add(genome, behavior, fitness=1.0)
        assert archive.num_bins >= 1

    def test_len(self):
        archive = ExactQDArchive(dimensions=[(3, 4, 5)], resolution=5)
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        archive.add(genome, [0.6], fitness=1.5)
        assert len(archive) == 1
