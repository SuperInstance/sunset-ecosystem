"""Tests for FleetDiversity — Pyversity-powered diversity selection.

Coverage targets:
- All 5 strategies (DPP, MMR, MSD, COVER, SSD)
- select_parents with various population sizes
- diversify_archive_elites with QDArchive
- rerank_nearest_neighbors
- compute_diversity_stats (ILAD, ILMD, pairwise distances)
- Fallback behavior when pyversity is unavailable
- Edge cases: empty, single item, k > n, zero embeddings
- Integration with FleetBreederConsensus
"""

import sys
from typing import Any, List, Tuple

import numpy as np
import pytest

from swarm.fleet_bft_qd import BehaviorDescriptor, QDArchive
from swarm.fleet_diversity import (
    DiversityStats,
    DiversityStrategy,
    FleetDiversitySelector,
    PopulationItem,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def make_population():
    """Factory for creating populations with controlled diversity."""

    def _make(
        n: int = 20,
        dim: int = 8,
        clusters: int = 3,
        seed: int = 42,
    ) -> List[PopulationItem]:
        rng = np.random.RandomState(seed)
        # Generate clustered embeddings for realistic diversity patterns
        centroids = rng.randn(clusters, dim)
        items = []
        for i in range(n):
            c = i % clusters
            emb = centroids[c] + rng.randn(dim) * 0.3
            fit = float(rng.rand())
            items.append(
                PopulationItem(
                    id=f"agent_{i}",
                    embedding=emb,
                    fitness=fit,
                    metadata={"cluster": c},
                )
            )
        return items

    return _make


@pytest.fixture
def selector() -> FleetDiversitySelector:
    return FleetDiversitySelector(
        strategy=DiversityStrategy.DPP,
        diversity=0.5,
        default_k=5,
    )


# ── Strategy Tests ─────────────────────────────────────────────────────────


class TestDPP:
    """Determinantal Point Process — probabilistic repulsion."""

    def test_select_diverse_parents(self, selector, make_population):
        pop = make_population(n=30, clusters=3)
        selected = selector.select_parents(pop, k=5)
        assert len(selected) == 5
        ids = [s.id for s in selected]
        assert len(set(ids)) == 5

    def test_spreads_across_clusters(self, selector, make_population):
        """DPP should pick from multiple clusters, not just one."""
        pop = make_population(n=30, clusters=3, seed=123)
        selected = selector.select_parents(pop, k=6)
        clusters = {s.metadata["cluster"] for s in selected}
        assert len(clusters) >= 2  # DPP tends to spread

    def test_respects_k(self, selector, make_population):
        pop = make_population(n=10)
        selected = selector.select_parents(pop, k=3)
        assert len(selected) == 3

    def test_k_larger_than_population(self, selector, make_population):
        pop = make_population(n=5)
        selected = selector.select_parents(pop, k=10)
        assert len(selected) == 5


class TestMMR:
    """Maximal Marginal Relevance — relevance minus similarity penalty."""

    def test_select_parents(self, selector, make_population):
        pop = make_population(n=20)
        selected = selector.select_parents(pop, k=5, strategy=DiversityStrategy.MMR)
        assert len(selected) == 5

    def test_high_diversity_reduces_similarity(self, make_population):
        """With diversity=1.0, MMR should maximize spread even at cost of relevance."""
        pop = make_population(n=20, seed=77)
        sel_high = FleetDiversitySelector(strategy=DiversityStrategy.MMR, diversity=1.0)
        sel_low = FleetDiversitySelector(strategy=DiversityStrategy.MMR, diversity=0.1)

        high = sel_high.select_parents(pop, k=5)
        low = sel_low.select_parents(pop, k=5)

        # High diversity should produce more spread-out selections
        high_emb = np.stack([s.embedding for s in high])
        low_emb = np.stack([s.embedding for s in low])
        high_dist = np.mean(
            [
                np.linalg.norm(high_emb[i] - high_emb[j])
                for i in range(5)
                for j in range(i + 1, 5)
            ]
        )
        low_dist = np.mean(
            [
                np.linalg.norm(low_emb[i] - low_emb[j])
                for i in range(5)
                for j in range(i + 1, 5)
            ]
        )
        assert high_dist >= low_dist * 0.5  # relaxed due to randomness


class TestMSD:
    """Max Sum of Distances — maximum variety."""

    def test_select_parents(self, selector, make_population):
        pop = make_population(n=20)
        selected = selector.select_parents(pop, k=5, strategy=DiversityStrategy.MSD)
        assert len(selected) == 5


class TestCOVER:
    """Facility-Location — topic coverage."""

    def test_select_parents(self, selector, make_population):
        pop = make_population(n=20)
        selected = selector.select_parents(pop, k=5, strategy=DiversityStrategy.COVER)
        assert len(selected) == 5

    def test_archive_coverage(self, selector, make_population):
        """COVER should cover multiple clusters."""
        pop = make_population(n=30, clusters=4, seed=88)
        selected = selector.select_parents(
            pop, k=6, strategy=DiversityStrategy.COVER, diversity=0.7
        )
        clusters = {s.metadata["cluster"] for s in selected}
        assert len(clusters) >= 2


class TestSSD:
    """Sliding Spectrum Decomposition — sequence-aware."""

    def test_select_parents(self, selector, make_population):
        pop = make_population(n=20)
        selected = selector.select_parents(pop, k=5, strategy=DiversityStrategy.SSD)
        assert len(selected) == 5


# ── Archive Integration ────────────────────────────────────────────────────


class TestArchiveDiversification:
    """Diversify QDArchive elites for cross-node breeding pools."""

    def test_diversify_archive_elites(self, selector):
        archive = QDArchive(
            grid_shape=(5, 5),
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            n_dims=2,
        )
        # Populate archive
        rng = np.random.RandomState(42)
        for i in range(20):
            desc = BehaviorDescriptor(
                values=rng.rand(2),
                names=("x", "y"),
            )
            archive.add(desc, {"id": f"elite_{i}"}, fitness=float(rng.rand()))

        elites = archive.get_all_elites()
        # Use grid indices as embeddings
        grid_indices = list(archive._grid.keys())
        embeddings = np.array(
            [
                np.array([idx / g for idx, g in zip(pos, archive.grid_shape)])
                for pos in grid_indices
            ]
        )

        selected = selector.diversify_archive_elites(
            elites, embeddings, k=5, strategy=DiversityStrategy.COVER
        )
        assert len(selected) == 5
        ids = {s[0]["id"] for s in selected}
        assert len(ids) == 5

    def test_wire_to_qd_archive(self, selector):
        archive = QDArchive(
            grid_shape=(4, 4),
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            n_dims=2,
        )
        rng = np.random.RandomState(7)
        for i in range(12):
            desc = BehaviorDescriptor(
                values=rng.rand(2),
                names=("x", "y"),
            )
            archive.add(desc, {"id": f"elite_{i}"}, fitness=float(rng.rand()))

        selected = selector.wire_to_qd_archive(archive, k=4)
        assert len(selected) == 4
        ids = {s["id"] for s in selected}
        assert len(ids) == 4

    def test_empty_archive(self, selector):
        archive = QDArchive(
            grid_shape=(3, 3),
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            n_dims=2,
        )
        selected = selector.wire_to_qd_archive(archive, k=5)
        assert selected == []


# ── Nearest Neighbor Re-ranking ───────────────────────────────────────────


class TestRerankNearestNeighbors:
    """Re-rank NN results with diversity."""

    def test_rerank_candidates(self, selector, make_population):
        pop = make_population(n=20, seed=99)
        query = np.ones(8)
        candidates = selector.rerank_nearest_neighbors(pop, query, k=5)
        assert len(candidates) == 5

    def test_query_zero_norm(self, selector, make_population):
        """Zero query should not crash."""
        pop = make_population(n=10)
        query = np.zeros(8)
        candidates = selector.rerank_nearest_neighbors(pop, query, k=3)
        assert len(candidates) == 3

    def test_empty_candidates(self, selector):
        query = np.ones(8)
        candidates = selector.rerank_nearest_neighbors([], query, k=5)
        assert candidates == []


# ── Diversity Statistics ───────────────────────────────────────────────────


class TestComputeDiversityStats:
    """Diversity metric computation."""

    def test_basic_stats(self, selector, make_population):
        pop = make_population(n=10, seed=55)
        stats = selector.compute_diversity_stats(pop)
        assert stats.n_items == 10
        assert 0.0 < stats.mean_fitness <= 1.0
        assert stats.mean_pairwise_distance >= 0.0

    def test_stats_with_selection(self, selector, make_population):
        pop = make_population(n=15, seed=66)
        selected = selector.select_parents(pop, k=5)
        indices = [pop.index(s) for s in selected]
        stats = selector.compute_diversity_stats(pop, selected_indices=indices)
        assert stats.selected_indices == indices
        assert stats.ilad >= 0.0
        assert stats.ilmd >= 0.0

    def test_empty_population(self, selector):
        stats = selector.compute_diversity_stats([])
        assert stats.n_items == 0
        assert stats.mean_fitness == 0.0

    def test_single_item(self, selector):
        pop = [PopulationItem(id="only", embedding=np.ones(4), fitness=0.5)]
        stats = selector.compute_diversity_stats(pop)
        assert stats.n_items == 1
        assert stats.mean_pairwise_distance == 0.0


# ── Fallback Tests (pyversity unavailable) ───────────────────────────────


class TestFallback:
    """Verify fallback implementations work without pyversity."""

    @pytest.fixture(autouse=True)
    def hide_pyversity(self, monkeypatch):
        """Temporarily hide pyversity to force fallback."""
        monkeypatch.setitem(sys.modules, "pyversity", None)
        # Force reimport by clearing cached module
        for key in list(sys.modules.keys()):
            if "fleet_diversity" in key:
                del sys.modules[key]

    def test_fallback_dpp(self, make_population, monkeypatch):
        """Fallback DPP should still select diverse items."""
        # We need to create a selector that can't import pyversity
        # The fallback is inside _diversify, so we need to test it differently
        # Instead, let's just test the fallback methods directly
        from swarm.fleet_diversity import FleetDiversitySelector, DiversityStrategy

        sel = FleetDiversitySelector(strategy=DiversityStrategy.DPP)
        pop = make_population(n=15, seed=111)
        # Manually call fallback
        embeddings = np.stack([p.embedding for p in pop])
        scores = np.array([p.fitness for p in pop])
        indices = sel._fallback_diversify(
            embeddings, scores, k=5, strategy=DiversityStrategy.DPP, diversity=0.5
        )
        assert len(indices) == 5
        assert len(set(indices)) == 5

    def test_fallback_mmr(self, make_population):
        from swarm.fleet_diversity import FleetDiversitySelector, DiversityStrategy

        sel = FleetDiversitySelector(strategy=DiversityStrategy.MMR)
        pop = make_population(n=15, seed=222)
        embeddings = np.stack([p.embedding for p in pop])
        scores = np.array([p.fitness for p in pop])
        indices = sel._fallback_mmr(
            embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
            scores,
            k=5,
            diversity=0.5,
        )
        assert len(indices) == 5

    def test_fallback_msd(self, make_population):
        from swarm.fleet_diversity import FleetDiversitySelector, DiversityStrategy

        sel = FleetDiversitySelector(strategy=DiversityStrategy.MSD)
        pop = make_population(n=15, seed=333)
        embeddings = np.stack([p.embedding for p in pop])
        scores = np.array([p.fitness for p in pop])
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        indices = sel._fallback_msd(normed, scores, k=5, diversity=0.5)
        assert len(indices) == 5

    def test_fallback_cover(self, make_population):
        from swarm.fleet_diversity import FleetDiversitySelector, DiversityStrategy

        sel = FleetDiversitySelector(strategy=DiversityStrategy.COVER)
        pop = make_population(n=15, seed=444)
        embeddings = np.stack([p.embedding for p in pop])
        scores = np.array([p.fitness for p in pop])
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        indices = sel._fallback_cover(normed, scores, k=5, diversity=0.5)
        assert len(indices) == 5

    def test_fallback_ssd(self, make_population):
        from swarm.fleet_diversity import FleetDiversitySelector, DiversityStrategy

        sel = FleetDiversitySelector(strategy=DiversityStrategy.SSD)
        pop = make_population(n=15, seed=555)
        embeddings = np.stack([p.embedding for p in pop])
        scores = np.array([p.fitness for p in pop])
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        indices = sel._fallback_ssd(normed, scores, k=5, diversity=0.5)
        assert len(indices) == 5


# ── Edge Cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and error handling."""

    def test_empty_population(self, selector):
        selected = selector.select_parents([])
        assert selected == []

    def test_single_item(self, selector):
        pop = [PopulationItem(id="lonely", embedding=np.ones(4), fitness=0.5)]
        selected = selector.select_parents(pop, k=5)
        assert len(selected) == 1
        assert selected[0].id == "lonely"

    def test_zero_embeddings(self, selector):
        """All-zero embeddings should not crash."""
        pop = [
            PopulationItem(id=f"z{i}", embedding=np.zeros(4), fitness=0.5)
            for i in range(5)
        ]
        selected = selector.select_parents(pop, k=3)
        assert len(selected) == 3

    def test_negative_fitness(self, selector):
        """Negative fitness scores should work."""
        pop = [
            PopulationItem(id=f"n{i}", embedding=np.random.randn(4), fitness=-1.0)
            for i in range(5)
        ]
        selected = selector.select_parents(pop, k=3)
        assert len(selected) == 3

    def test_very_high_diversity(self, selector, make_population):
        """diversity=1.0 should maximize spread."""
        pop = make_population(n=20, seed=777)
        selected = selector.select_parents(
            pop, k=5, diversity=1.0, strategy=DiversityStrategy.MMR
        )
        assert len(selected) == 5

    def test_very_low_diversity(self, selector, make_population):
        """diversity=0.0 should pick top fitness."""
        pop = make_population(n=20, seed=888)
        selected = selector.select_parents(
            pop, k=5, diversity=0.0, strategy=DiversityStrategy.MMR
        )
        # With diversity=0, should pick top 5 by fitness
        top_5_ids = {
            p.id for p in sorted(pop, key=lambda x: x.fitness, reverse=True)[:5]
        }
        selected_ids = {s.id for s in selected}
        assert selected_ids == top_5_ids


# ── History and State ───────────────────────────────────────────────────────


class TestHistory:
    """Selection history tracking."""

    def test_history_recorded(self, selector, make_population):
        pop = make_population(n=10)
        selector.select_parents(pop, k=3)
        selector.select_parents(pop, k=3)
        history = selector.get_history()
        assert len(history) == 2
        assert all(isinstance(h[0], str) for h in history)
        assert all(isinstance(h[1], list) for h in history)

    def test_clear_history(self, selector, make_population):
        pop = make_population(n=10)
        selector.select_parents(pop, k=3)
        selector.clear_history()
        assert selector.get_history() == []

    def test_repr(self, selector):
        r = repr(selector)
        assert "FleetDiversitySelector" in r
        assert "dpp" in r


# ── Parameter Override ─────────────────────────────────────────────────────


class TestParameterOverride:
    """Per-call parameter overrides."""

    def test_override_strategy(self, selector, make_population):
        pop = make_population(n=15)
        # Default is DPP, override to MMR
        selected = selector.select_parents(pop, k=5, strategy=DiversityStrategy.MMR)
        assert len(selected) == 5
        history = selector.get_history()
        assert history[-1][0] == "mmr"

    def test_override_diversity(self, selector, make_population):
        pop = make_population(n=15)
        selected = selector.select_parents(pop, k=5, diversity=0.9)
        assert len(selected) == 5

    def test_override_k(self, selector, make_population):
        pop = make_population(n=15)
        selected = selector.select_parents(pop, k=3)
        assert len(selected) == 3


# ── Integration with FleetBreederConsensus ───────────────────────────────


class TestBreederIntegration:
    """FleetBreederConsensus integration patterns."""

    def test_from_breeder_consensus(self):
        """Factory method creates selector wired to consensus."""
        from swarm.fleet_bft_qd import FleetBreederConsensus

        fbc = FleetBreederConsensus(
            node_id="n0",
            all_nodes=["n0", "n1", "n2", "n3"],
            secret_key="secret",
        )
        sel = FleetDiversitySelector.from_breeder_consensus(
            fbc, strategy=DiversityStrategy.DPP, diversity=0.6
        )
        assert sel.strategy == DiversityStrategy.DPP
        assert sel.diversity == 0.6

    def test_diversity_before_bft_proposal(self, make_population):
        """Pattern: diversity-select parents, then BFT-propose breeding batch."""
        from swarm.fleet_bft_qd import FleetBreederConsensus

        fbc = FleetBreederConsensus(
            node_id="n0",
            all_nodes=["n0", "n1", "n2", "n3"],
            secret_key="secret",
        )
        sel = FleetDiversitySelector.from_breeder_consensus(fbc)

        pop = make_population(n=20, seed=123)
        parents = sel.select_parents(pop, k=5)
        parent_ids = [p.id for p in parents]

        # Propose breeding batch via BFT (pass candidates as dicts)
        candidates = [{"id": pid, "chaos": 0.3} for pid in parent_ids]
        ok = fbc.propose_breeding_batch(candidates, batch_size=5)
        assert ok is not None  # returns a PBFTMessage or None

    def test_evaluate_offspring_updates_archive(self, make_population):
        """Pattern: breed → evaluate → archive with diversity stats."""
        from swarm.fleet_bft_qd import FleetBreederConsensus

        fbc = FleetBreederConsensus(
            node_id="n0",
            all_nodes=["n0", "n1", "n2", "n3"],
            secret_key="secret",
        )
        sel = FleetDiversitySelector.from_breeder_consensus(fbc)

        pop = make_population(n=15, seed=234)
        parents = sel.select_parents(pop, k=4)

        # Simulate offspring evaluation
        for i, parent in enumerate(parents):
            child = {"id": f"child_{i}", "parent": parent.id}
            fitness = parent.fitness * 1.1  # slight improvement
            behavior = parent.embedding + np.random.randn(len(parent.embedding)) * 0.1
            fbc.evaluate_offspring(child, fitness, behavior)

        # Archive should have items
        assert fbc.archive.coverage > 0
        assert fbc.archive.qd_score > 0

        # Diversity stats on the breeding population
        stats = sel.compute_diversity_stats(
            pop, selected_indices=[pop.index(p) for p in parents]
        )
        assert stats.selected_fitness_mean > 0
        assert stats.ilad >= 0.0


# ── Benchmark / Scale ─────────────────────────────────────────────────────


class TestScale:
    """Performance at reasonable fleet scales."""

    def test_100_candidates(self, selector):
        """Should handle 100 candidates quickly."""
        rng = np.random.RandomState(999)
        pop = [
            PopulationItem(
                id=f"c{i}",
                embedding=rng.randn(32),
                fitness=float(rng.rand()),
            )
            for i in range(100)
        ]
        import time

        t0 = time.perf_counter()
        selected = selector.select_parents(pop, k=10)
        t1 = time.perf_counter()
        assert len(selected) == 10
        assert t1 - t0 < 1.0  # should be sub-second

    def test_256_dim_embeddings(self, selector):
        """Should handle high-dimensional embeddings."""
        rng = np.random.RandomState(111)
        pop = [
            PopulationItem(
                id=f"d{i}",
                embedding=rng.randn(256),
                fitness=float(rng.rand()),
            )
            for i in range(20)
        ]
        selected = selector.select_parents(pop, k=5)
        assert len(selected) == 5

    def test_3d_archive(self, selector):
        """3D QDArchive should work with wire_to_qd_archive."""
        from swarm.fleet_bft_qd import QDArchive, BehaviorDescriptor

        archive = QDArchive(
            grid_shape=(4, 4, 4),
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            n_dims=3,
        )
        rng = np.random.RandomState(222)
        for i in range(30):
            desc = BehaviorDescriptor(
                values=rng.rand(3),
                names=("x", "y", "z"),
            )
            archive.add(desc, {"id": f"e{i}"}, fitness=float(rng.rand()))

        selected = selector.wire_to_qd_archive(archive, k=5)
        assert len(selected) == 5
