"""Tests for FluxVectorTable diversity search methods.

Mocks turbovec so tests run without the native extension.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

# ── Mock turbovec before any swarm.vector_table import ──
_mock_turbovec = types.ModuleType("turbovec")


class _MockIdMapIndex:
    """Minimal stand-in for turbovec.IdMapIndex."""

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self._vectors: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for vec, aid in zip(vectors, ids):
            self._vectors[int(aid)] = vec.copy()

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        allowlist: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.zeros((1, k), dtype=np.float32),
                np.zeros((1, k), dtype=np.uint64),
            )
        q = query[0]
        candidates = list(self._vectors.items())
        if allowlist is not None:
            allowed = set(int(a) for a in allowlist)
            candidates = [(aid, v) for aid, v in candidates if aid in allowed]

        qn = q / (np.linalg.norm(q) + 1e-8)
        sims: list[tuple[int, float]] = []
        for aid, vec in candidates:
            vn = vec / (np.linalg.norm(vec) + 1e-8)
            sims.append((aid, float(np.dot(qn, vn))))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:k]
        while len(top) < k:
            top.append((0, 0.0))
        scores = np.array([[s for _, s in top]], dtype=np.float32)
        ids_arr = np.array([[aid for aid, _ in top]], dtype=np.uint64)
        return scores, ids_arr

    def remove(self, agent_id: int) -> bool:
        return self._vectors.pop(agent_id, None) is not None

    def contains(self, agent_id: int) -> bool:
        return agent_id in self._vectors

    def write(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "_MockIdMapIndex":
        return cls(dim=256)


_mock_turbovec.IdMapIndex = _MockIdMapIndex  # type: ignore[attr-defined]
sys.modules["turbovec"] = _mock_turbovec

from swarm.flux_vector_table import AgentVector, FluxVectorTable


@pytest.fixture
def vector_table():
    """Pre-populated vector table with 12 agents across 3 niches."""
    vt = FluxVectorTable(dim=64, bit_width=4)
    rng = np.random.RandomState(42)

    # Niche A: agents 0-3 (high fitness, similar vectors)
    base_a = rng.randn(64).astype(np.float32) * 2.0
    for i in range(4):
        vec = base_a + rng.randn(64).astype(np.float32) * 0.1
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec.tolist(),
                fitness=0.9 - i * 0.02,
                generation=1,
                capability_mask=0xFFFF,
                thermal_pressure=0.1,
            )
        )

    # Niche B: agents 4-7 (medium fitness, different cluster)
    base_b = rng.randn(64).astype(np.float32) * 2.0 + 5.0
    for i in range(4, 8):
        vec = base_b + rng.randn(64).astype(np.float32) * 0.1
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec.tolist(),
                fitness=0.6 - (i - 4) * 0.02,
                generation=1,
                capability_mask=0xFFFF,
                thermal_pressure=0.2,
            )
        )

    # Niche C: agents 8-11 (lower fitness, third cluster)
    base_c = rng.randn(64).astype(np.float32) * 2.0 - 5.0
    for i in range(8, 12):
        vec = base_c + rng.randn(64).astype(np.float32) * 0.1
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec.tolist(),
                fitness=0.4 - (i - 8) * 0.02,
                generation=1,
                capability_mask=0xFFFF,
                thermal_pressure=0.3,
            )
        )

    return vt


class TestComputeDiversityMatrix:
    """compute_diversity_matrix returns correct shape and values."""

    def test_shape_matches_population(self, vector_table):
        n = len(vector_table)
        dists, ids = vector_table.compute_diversity_matrix()
        assert dists.shape == (n, n)
        assert len(ids) == n

    def test_self_distance_is_zero(self, vector_table):
        dists, ids = vector_table.compute_diversity_matrix()
        id_to_idx = {aid: i for i, aid in enumerate(ids)}
        for aid in ids:
            idx = id_to_idx[aid]
            assert dists[idx, idx] == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self, vector_table):
        dists, ids = vector_table.compute_diversity_matrix()
        n = len(ids)
        for i in range(n):
            for j in range(n):
                assert dists[i, j] == pytest.approx(dists[j, i], abs=1e-5)

    def test_range_zero_to_two(self, vector_table):
        dists, _ = vector_table.compute_diversity_matrix()
        assert dists.min() >= 0.0 - 1e-5
        assert dists.max() <= 2.0 + 1e-5

    def test_different_niches_have_higher_distance(self, vector_table):
        dists, ids = vector_table.compute_diversity_matrix()
        id_to_idx = {aid: i for i, aid in enumerate(ids)}
        # Agent 0 (niche A) vs agent 4 (niche B) should be farther
        # than agent 0 vs agent 1 (both niche A)
        d_0_4 = dists[id_to_idx[0], id_to_idx[4]]
        d_0_1 = dists[id_to_idx[0], id_to_idx[1]]
        assert d_0_4 > d_0_1


class TestFindNicheCentroids:
    """find_niche_centroids discovers population clusters."""

    def test_returns_centroids_and_assignments(self, vector_table):
        centroids, assignments = vector_table.find_niche_centroids(k=3)
        assert centroids.shape == (3, 64)
        assert len(assignments) == len(vector_table)

    def test_every_agent_assigned(self, vector_table):
        _, assignments = vector_table.find_niche_centroids(k=3)
        for aid in vector_table._meta.keys():
            assert aid in assignments
            assert 0 <= assignments[aid] < 3

    def test_niches_are_distinct(self, vector_table):
        centroids, _ = vector_table.find_niche_centroids(k=3)
        # Centroids should be far apart
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                assert dist > 0.1

    def test_reduces_k_when_population_small(self, vector_table):
        small_vt = FluxVectorTable(dim=64, bit_width=4)
        vec = np.random.randn(64).astype(np.float32).tolist()
        small_vt.add(AgentVector(agent_id=0, vector=vec, fitness=0.5))
        centroids, assignments = small_vt.find_niche_centroids(k=3)
        assert centroids.shape[0] <= 1


class TestSearchDiverseParents:
    """search_diverse_parents returns maximally diverse pairs."""

    def test_returns_pairs(self, vector_table):
        pairs = vector_table.search_diverse_parents(n_results=2)
        assert len(pairs) == 2
        for a, b in pairs:
            assert a != b
            assert a in vector_table._meta
            assert b in vector_table._meta

    def test_pairs_are_unique(self, vector_table):
        pairs = vector_table.search_diverse_parents(n_results=3)
        used = set()
        for a, b in pairs:
            assert a not in used
            assert b not in used
            used.add(a)
            used.add(b)

    def test_pairs_are_diverse(self, vector_table):
        pairs = vector_table.search_diverse_parents(n_results=1)
        assert len(pairs) == 1
        a, b = pairs[0]
        vec_a = vector_table._get_vector(a)
        vec_b = vector_table._get_vector(b)
        # Cosine distance should be relatively high
        sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8))
        dist = 1.0 - sim
        assert dist > 0.1

    def test_returns_empty_for_single_agent(self):
        vt = FluxVectorTable(dim=64, bit_width=4)
        vt.add(AgentVector(agent_id=0, vector=np.random.randn(64).astype(np.float32).tolist(), fitness=0.5))
        pairs = vt.search_diverse_parents(n_results=1)
        assert pairs == []


class TestRecommendBreedPair:
    """recommend_breed_pair picks parents from different niches."""

    def test_returns_tuple(self, vector_table):
        pair = vector_table.recommend_breed_pair()
        assert pair is not None
        a, b = pair
        assert a != b
        assert a in vector_table._meta
        assert b in vector_table._meta

    def test_parents_from_different_niches(self, vector_table):
        _, assignments = vector_table.find_niche_centroids(k=3)
        pair = vector_table.recommend_breed_pair()
        assert pair is not None
        a, b = pair
        niche_a = assignments.get(a, 0)
        niche_b = assignments.get(b, 0)
        assert niche_a != niche_b, f"Parents {a} and {b} are from same niche {niche_a}"

    def test_high_fitness_parents_preferred(self, vector_table):
        pair = vector_table.recommend_breed_pair()
        assert pair is not None
        a, b = pair
        fitness_a = vector_table._meta[a].fitness
        fitness_b = vector_table._meta[b].fitness
        # Both should be reasonably high (top half of population)
        assert fitness_a >= 0.3
        assert fitness_b >= 0.3

    def test_fallback_for_single_niche(self):
        vt = FluxVectorTable(dim=64, bit_width=4)
        base = np.random.randn(64).astype(np.float32)
        for i in range(5):
            vec = base + np.random.randn(64).astype(np.float32) * 0.01
            vt.add(AgentVector(agent_id=i, vector=vec.tolist(), fitness=0.5 - i * 0.01))
        pair = vt.recommend_breed_pair()
        assert pair is not None
        a, b = pair
        assert a != b

    def test_returns_none_for_empty_table(self):
        vt = FluxVectorTable(dim=64, bit_width=4)
        assert vt.recommend_breed_pair() is None


class TestGetVector:
    """_get_vector retrieves raw vectors."""

    def test_retrieves_existing_vector(self, vector_table):
        vec = vector_table._get_vector(0)
        assert vec is not None
        assert len(vec) == 64
        assert vec.dtype == np.float32

    def test_returns_none_for_missing_agent(self, vector_table):
        assert vector_table._get_vector(999) is None

    def test_retrieves_all_agents(self, vector_table):
        for aid in vector_table._meta.keys():
            vec = vector_table._get_vector(aid)
            assert vec is not None
            assert len(vec) == 64
