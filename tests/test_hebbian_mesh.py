"""Tests for HebbianMeshLayer — diversity-aware stochastic routing.

Run with: pytest tests/test_hebbian_mesh.py -v
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from swarm.hebbian_mesh import (
    HebbianAffinity,
    HebbianMeshLayer,
    HebbianOutcome,
    DiversityError,
    BLACKLIST_THRESHOLD,
    CHAOS_MIN,
    CHAOS_MAX,
    DELTA_SUCCESS,
    DELTA_TIMEOUT,
    DELTA_VIOLATION,
    DELTA_NOVELTY,
)
from swarm.mesh_vector_gossip import MeshVectorGossip, GossipResult
from swarm.flux_vector_table import FluxVectorTable, AgentVector


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_table_64() -> FluxVectorTable:
    """Return a FluxVectorTable with dim=64, 4 diverse agents."""
    table = FluxVectorTable(dim=64, bit_width=4)
    for i in range(4):
        vec = np.random.randn(64).astype(np.float32).tolist()
        av = AgentVector(
            agent_id=i + 100,
            vector=vec,
            fitness=0.5 + i * 0.1,
            generation=1,
            capability_mask=0xFFFF,
            thermal_pressure=0.2,
        )
        table.add(av)
    return table


@pytest.fixture
def mock_gossip(mock_table_64: FluxVectorTable) -> MeshVectorGossip:
    """Return a MeshVectorGossip wired to the mock table."""
    return MeshVectorGossip(
        node_id="Oracle1",
        local_table=mock_table_64,
        max_peers_per_round=2,
    )


@pytest.fixture
def mesh_layer(mock_gossip: MeshVectorGossip) -> HebbianMeshLayer:
    """Return a HebbianMeshLayer wrapping the mock gossip."""
    return HebbianMeshLayer(mock_gossip)


# ── Affinity update tests ─────────────────────────────────────


class TestAffinityUpdates:
    def test_success_increases_strength(self, mesh_layer: HebbianMeshLayer):
        """SUCCESS outcome should increase affinity strength by +0.1."""
        mesh_layer.update_affinity("ProArt", HebbianOutcome.SUCCESS)
        aff = mesh_layer.get_affinity("ProArt")
        assert aff.strength == pytest.approx(0.50 + DELTA_SUCCESS, abs=0.001)
        assert aff.interaction_count == 1

    def test_timeout_decreases_strength(self, mesh_layer: HebbianMeshLayer):
        """TIMEOUT outcome should decrease affinity strength by -0.2."""
        mesh_layer.update_affinity("ProArt", HebbianOutcome.TIMEOUT)
        aff = mesh_layer.get_affinity("ProArt")
        assert aff.strength == pytest.approx(0.50 + DELTA_TIMEOUT, abs=0.001)

    def test_violation_decreases_strength(self, mesh_layer: HebbianMeshLayer):
        """VIOLATION outcome should decrease affinity strength by -0.3."""
        mesh_layer.update_affinity("ProArt", HebbianOutcome.VIOLATION)
        aff = mesh_layer.get_affinity("ProArt")
        assert aff.strength == pytest.approx(0.50 + DELTA_VIOLATION, abs=0.001)

    def test_novelty_increases_strength(self, mesh_layer: HebbianMeshLayer):
        """NOVELTY outcome should increase affinity strength by +0.15."""
        mesh_layer.update_affinity("ProArt", HebbianOutcome.NOVELTY)
        aff = mesh_layer.get_affinity("ProArt")
        assert aff.strength == pytest.approx(0.50 + DELTA_NOVELTY, abs=0.001)

    def test_success_capped_at_1_0(self, mesh_layer: HebbianMeshLayer):
        """Repeated SUCCESS should cap at 1.0."""
        for _ in range(10):
            mesh_layer.update_affinity("ProArt", HebbianOutcome.SUCCESS)
        aff = mesh_layer.get_affinity("ProArt")
        assert aff.strength == pytest.approx(1.0, abs=0.001)

    def test_timeout_floored_at_0_0(self, mesh_layer: HebbianMeshLayer):
        """Repeated TIMEOUT should floor at 0.0."""
        for _ in range(10):
            mesh_layer.update_affinity("ProArt", HebbianOutcome.TIMEOUT)
        aff = mesh_layer.get_affinity("ProArt")
        assert aff.strength == pytest.approx(0.0, abs=0.001)


# ── Blacklist tests ─────────────────────────────────────────────


class TestBlacklist:
    def test_violation_blacklists_below_threshold(self, mesh_layer: HebbianMeshLayer):
        """VIOLATION that drops strength below 0.1 should blacklist peer."""
        # Start at 0.5, one VIOLATION → 0.2, second VIOLATION → -0.1 → 0.0 → blacklisted
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        assert not mesh_layer.is_blacklisted("BadPeer")  # 0.2 still above threshold
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        assert mesh_layer.is_blacklisted("BadPeer")

    def test_blacklisted_peer_ignored_for_non_novelty(self, mesh_layer: HebbianMeshLayer):
        """Blacklisted peer should not get affinity updates from SUCCESS/TIMEOUT."""
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        assert mesh_layer.is_blacklisted("BadPeer")
        # SUCCESS should be ignored
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.SUCCESS)
        aff = mesh_layer.get_affinity("BadPeer")
        assert aff.strength == pytest.approx(0.0, abs=0.001)  # unchanged

    def test_novelty_unblacklists_peer(self, mesh_layer: HebbianMeshLayer):
        """NOVELTY outcome should un-blacklist a peer."""
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        assert mesh_layer.is_blacklisted("BadPeer")
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.NOVELTY)
        assert not mesh_layer.is_blacklisted("BadPeer")
        aff = mesh_layer.get_affinity("BadPeer")
        assert aff.strength == pytest.approx(0.0 + DELTA_NOVELTY, abs=0.001)

    def test_list_blacklisted(self, mesh_layer: HebbianMeshLayer):
        """list_blacklisted should return all blacklisted peer IDs."""
        mesh_layer.update_affinity("A", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("A", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("B", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("B", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("C", HebbianOutcome.SUCCESS)
        blacklisted = mesh_layer.list_blacklisted()
        assert "A" in blacklisted
        assert "B" in blacklisted
        assert "C" not in blacklisted


# ── Diversity score tests ───────────────────────────────────────


class TestDiversityScore:
    def test_diversity_computed_from_vectors(self, mesh_layer: HebbianMeshLayer):
        """get_diversity_score should return a value between 0 and 1."""
        score = mesh_layer.get_diversity_score()
        assert 0.0 <= score <= 1.0

    def test_diversity_empty_table_raises(self, mock_gossip: MeshVectorGossip):
        """Empty table should raise DiversityError."""
        empty_table = FluxVectorTable(dim=64, bit_width=4)
        gossip = MeshVectorGossip(node_id="EmptyNode", local_table=empty_table)
        mesh = HebbianMeshLayer(gossip)
        with pytest.raises(DiversityError):
            mesh.get_diversity_score()

    def test_low_diversity_collapsed_vectors(self, mock_gossip: MeshVectorGossip):
        """Nearly identical vectors should produce low diversity."""
        table = FluxVectorTable(dim=64, bit_width=4)
        base_vec = np.random.randn(64).astype(np.float32).tolist()
        for i in range(4):
            # Tiny perturbation
            vec = [v + np.random.normal(0, 0.001) for v in base_vec]
            table.add(AgentVector(agent_id=200 + i, vector=vec, fitness=0.5))
        gossip = MeshVectorGossip(node_id="CollapseNode", local_table=table)
        mesh = HebbianMeshLayer(gossip)
        score = mesh.get_diversity_score()
        assert score < 0.15  # Should be very low

    def test_high_diversity_diverse_vectors(self, mock_gossip: MeshVectorGossip):
        """Widely spread vectors should produce high diversity."""
        table = FluxVectorTable(dim=64, bit_width=4)
        for i in range(4):
            # Each vector in a different quadrant
            vec = np.random.randn(64).astype(np.float32) * (2.0 + i * 0.5)
            table.add(AgentVector(agent_id=300 + i, vector=vec.tolist(), fitness=0.5))
        gossip = MeshVectorGossip(node_id="SpreadNode", local_table=table)
        mesh = HebbianMeshLayer(gossip)
        score = mesh.get_diversity_score()
        assert score > 0.30  # Should be noticeably high


# ── Chaos factor tests ──────────────────────────────────────────


class TestChaosFactor:
    def test_chaos_increases_when_diversity_drops(self, mesh_layer: HebbianMeshLayer):
        """Low diversity should push chaos toward CHAOS_MAX."""
        # Mock low diversity by using collapsed vectors
        table = FluxVectorTable(dim=64, bit_width=4)
        base_vec = np.random.randn(64).astype(np.float32).tolist()
        for i in range(4):
            vec = [v + np.random.normal(0, 0.001) for v in base_vec]
            table.add(AgentVector(agent_id=400 + i, vector=vec, fitness=0.5))
        gossip = MeshVectorGossip(node_id="LowDivNode", local_table=table)
        mesh = HebbianMeshLayer(gossip)
        chaos = mesh.chaos_factor
        assert chaos >= CHAOS_MAX * 0.8  # Should be near max

    def test_chaos_decreases_when_diversity_recovers(self, mesh_layer: HebbianMeshLayer):
        """High diversity should push chaos toward CHAOS_MIN."""
        # Default mock_table_64 has reasonably diverse vectors
        chaos = mesh_layer.chaos_factor
        assert chaos <= CHAOS_MAX * 0.6  # Should be well below max

    def test_chaos_bounds(self, mesh_layer: HebbianMeshLayer):
        """Chaos factor should always stay within [CHAOS_MIN, CHAOS_MAX]."""
        chaos = mesh_layer.chaos_factor
        assert CHAOS_MIN <= chaos <= CHAOS_MAX


# ── Routing tests ───────────────────────────────────────────────


class TestRouting:
    def test_route_with_chaos_returns_correct_count(self, mesh_layer: HebbianMeshLayer):
        """route_with_chaos should return exactly n_routes peers."""
        pool = [f"peer_{i}" for i in range(10)]
        selected = mesh_layer.route_with_chaos(pool, n_routes=3)
        assert len(selected) == 3
        assert all(p in pool for p in selected)

    def test_route_with_chaos_no_duplicates(self, mesh_layer: HebbianMeshLayer):
        """route_with_chaos should never return duplicate peers."""
        pool = [f"peer_{i}" for i in range(20)]
        selected = mesh_layer.route_with_chaos(pool, n_routes=5)
        assert len(selected) == len(set(selected))

    def test_route_with_chaos_empty_pool(self, mesh_layer: HebbianMeshLayer):
        """route_with_chaos with empty pool should return empty list."""
        assert mesh_layer.route_with_chaos([], n_routes=3) == []

    def test_select_peers_weights_by_affinity(self, mesh_layer: HebbianMeshLayer):
        """Peers with higher affinity should be selected more frequently."""
        pool = ["GoodPeer", "BadPeer"]
        # Boost GoodPeer, hurt BadPeer
        for _ in range(5):
            mesh_layer.update_affinity("GoodPeer", HebbianOutcome.SUCCESS)
        for _ in range(5):
            mesh_layer.update_affinity("BadPeer", HebbianOutcome.TIMEOUT)

        # Run many selections and count
        counts = {"GoodPeer": 0, "BadPeer": 0}
        for _ in range(200):
            picked = mesh_layer.select_peers_for_gossip(pool, k=1)
            if picked:
                counts[picked[0]] += 1

        # GoodPeer should be selected significantly more often
        assert counts["GoodPeer"] > counts["BadPeer"]

    def test_blacklisted_peer_never_selected(self, mesh_layer: HebbianMeshLayer):
        """Blacklisted peers should never appear in route_with_chaos output."""
        pool = ["A", "B", "BadPeer"]
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("BadPeer", HebbianOutcome.VIOLATION)
        assert mesh_layer.is_blacklisted("BadPeer")

        for _ in range(50):
            selected = mesh_layer.route_with_chaos(pool, n_routes=2)
            assert "BadPeer" not in selected


# ── Thread-safety tests ─────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_affinity_updates(self, mesh_layer: HebbianMeshLayer):
        """Concurrent updates from multiple threads should not corrupt state."""
        errors = []

        def updater(outcome: HebbianOutcome) -> None:
            try:
                for _ in range(50):
                    mesh_layer.update_affinity("ConcurrentPeer", outcome)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=updater, args=(HebbianOutcome.SUCCESS,)),
            threading.Thread(target=updater, args=(HebbianOutcome.TIMEOUT,)),
            threading.Thread(target=updater, args=(HebbianOutcome.NOVELTY,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        aff = mesh_layer.get_affinity("ConcurrentPeer")
        # Strength should still be within valid bounds
        assert 0.0 <= aff.strength <= 1.0
        assert aff.interaction_count == 150

    def test_concurrent_routing_and_updates(self, mesh_layer: HebbianMeshLayer):
        """Routing while affinity updates happen concurrently should be safe."""
        results = []

        def router() -> None:
            for _ in range(50):
                pool = [f"peer_{i}" for i in range(10)]
                selected = mesh_layer.route_with_chaos(pool, n_routes=2)
                results.append(len(selected))

        def updater() -> None:
            for i in range(50):
                peer = f"peer_{i % 10}"
                mesh_layer.update_affinity(peer, HebbianOutcome.SUCCESS)

        t1 = threading.Thread(target=router)
        t2 = threading.Thread(target=updater)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All routing calls should have returned valid counts
        assert all(0 <= r <= 2 for r in results)
        assert len(results) == 50

    def test_route_with_chaos_concurrent(self, mesh_layer: HebbianMeshLayer):
        """Concurrent routing from many threads must not crash and must respect blacklist."""
        # Seed 1000 peers with varied affinities; blacklist ~10%
        pool = [f"peer_{i}" for i in range(1000)]
        blacklisted = set()
        for i, pid in enumerate(pool):
            if i % 10 == 0:
                # Blacklist this peer
                mesh_layer.update_affinity(pid, HebbianOutcome.VIOLATION)
                mesh_layer.update_affinity(pid, HebbianOutcome.VIOLATION)
                blacklisted.add(pid)
            elif i % 3 == 0:
                mesh_layer.update_affinity(pid, HebbianOutcome.SUCCESS)
            else:
                mesh_layer.update_affinity(pid, HebbianOutcome.TIMEOUT)

        errors = []
        selected_counts: list[int] = []
        all_selected: list[list[str]] = []

        def router() -> None:
            for _ in range(1000):
                try:
                    selected = mesh_layer.route_with_chaos(pool, n_routes=10)
                    selected_counts.append(len(selected))
                    all_selected.append(selected)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=router) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Routing crashed: {errors[:3]}"
        assert len(selected_counts) == 10_000
        assert all(0 <= c <= 10 for c in selected_counts)

        # Verify blacklist was respected across all selections
        for selected in all_selected:
            for pid in selected:
                assert pid not in blacklisted, f"Blacklisted peer {pid} was selected"

    def test_lock_free_read_path(self, mesh_layer: HebbianMeshLayer):
        """route_with_chaos() must not acquire self._lock on the read path.

        We hold the lock in the main thread and spawn a thread that calls
        route_with_chaos().  If the read path were not lock-free, the
        spawned thread would deadlock waiting for the lock.
        """
        # Pre-populate some affinities so the cache is non-empty
        for i in range(10):
            mesh_layer.update_affinity(f"peer_{i}", HebbianOutcome.SUCCESS)

        result: list[Any] = [None]

        def target():
            pool = [f"peer_{i}" for i in range(10)]
            result[0] = mesh_layer.route_with_chaos(pool, n_routes=3)

        # Hold the lock — a non-lock-free read would deadlock here
        with mesh_layer._lock:
            t = threading.Thread(target=target)
            t.start()
            t.join(timeout=1.0)

        assert not t.is_alive(), (
            "route_with_chaos() deadlocked — it is NOT lock-free"
        )
        assert result[0] is not None
        assert len(result[0]) == 3
        assert all(p in [f"peer_{i}" for i in range(10)] for p in result[0])


# ── Integration / wrapper tests ─────────────────────────────────


class TestGossipWrapper:
    def test_gossip_round_updates_affinity(self, mesh_layer: HebbianMeshLayer, mock_table_64: FluxVectorTable):
        """The gossip_round wrapper should auto-update affinities from results."""
        # Inject a remote delta so merge succeeds
        remote_deltas = {
            "agents": [
                {
                    "agent_id": 888,
                    "vector": [0.1] * 64,
                    "fitness": 0.88,
                    "generation": 1,
                    "capability_mask": 0xFFFF,
                    "thermal_pressure": 0.1,
                    "wall_time": 1.0,
                }
            ]
        }
        mesh_layer.gossip._fetch_peer_deltas = lambda peer, digest: remote_deltas

        mesh_layer.gossip_round(["ProArt"])
        aff = mesh_layer.get_affinity("ProArt")
        # Should have gotten at least one SUCCESS update
        assert aff.interaction_count >= 1
        assert aff.strength > 0.50  # SUCCESS boosted it

    def test_stats_snapshot(self, mesh_layer: HebbianMeshLayer):
        """stats property should return a coherent snapshot."""
        mesh_layer.update_affinity("A", HebbianOutcome.SUCCESS)
        mesh_layer.update_affinity("B", HebbianOutcome.VIOLATION)
        mesh_layer.update_affinity("B", HebbianOutcome.VIOLATION)

        s = mesh_layer.stats
        assert s["peer_count"] == 2
        assert s["blacklisted_count"] == 1  # B should be blacklisted
        assert 0.0 <= s["avg_strength"] <= 1.0
        assert 0.0 <= s["avg_trust"] <= 1.0
        assert "chaos_factor" in s
        assert "last_diversity" in s

    def test_reset_affinity(self, mesh_layer: HebbianMeshLayer):
        """reset_affinity should restore a peer to default state."""
        mesh_layer.update_affinity("Peer", HebbianOutcome.SUCCESS)
        mesh_layer.update_affinity("Peer", HebbianOutcome.SUCCESS)
        assert mesh_layer.get_affinity("Peer").strength > 0.50
        mesh_layer.reset_affinity("Peer")
        aff = mesh_layer.get_affinity("Peer")
        assert aff.strength == pytest.approx(0.50, abs=0.001)
        assert aff.interaction_count == 0
        assert not aff.blacklisted


# ── Edge-case tests ─────────────────────────────────────────────


class TestEdgeCases:
    def test_affinity_defaults(self, mesh_layer: HebbianMeshLayer):
        """Requesting affinity for unseen peer should return defaults."""
        aff = mesh_layer.get_affinity("NeverSeen")
        assert aff.peer_id == "NeverSeen"
        assert aff.strength == pytest.approx(0.50, abs=0.001)
        assert aff.trust_score == pytest.approx(0.50, abs=0.001)
        assert not aff.blacklisted

    def test_chaos_with_empty_diversity(self, mock_gossip: MeshVectorGossip):
        """When diversity cannot be computed, chaos should default to max."""
        empty_table = FluxVectorTable(dim=64, bit_width=4)
        gossip = MeshVectorGossip(node_id="Empty", local_table=empty_table)
        mesh = HebbianMeshLayer(gossip)
        chaos = mesh.chaos_factor
        assert chaos == pytest.approx(CHAOS_MAX, abs=0.01)

    def test_route_with_all_blacklisted(self, mesh_layer: HebbianMeshLayer):
        """When all peers blacklisted, routing should fall back to random."""
        pool = ["A", "B", "C"]
        for p in pool:
            mesh_layer.update_affinity(p, HebbianOutcome.VIOLATION)
            mesh_layer.update_affinity(p, HebbianOutcome.VIOLATION)
        selected = mesh_layer.route_with_chaos(pool, n_routes=2)
        assert len(selected) == 2
        assert all(s in pool for s in selected)
