"""Tests for MeshVectorGossip — anti-entropy CRDT gossip layer.

Run with: pytest tests/test_mesh_vector_gossip.py -v
"""

from __future__ import annotations

import pytest
import numpy as np
import time
from typing import Any

from swarm.mesh_vector_gossip import (
    MeshVectorGossip,
    GossipDigest,
    DeltaBatch,
    GossipResult,
    ThermalRoutingError,
)
from swarm.flux_vector_table import FluxVectorTable, AgentVector, AgentMeta


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_table_64() -> FluxVectorTable:
    """Return a FluxVectorTable with dim=64, 4 agents."""
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
def mock_wal() -> Any:
    """Return a simple mock WAL that records append calls."""
    class MockWAL:
        def __init__(self):
            self.entries = []
        def append(self, entry):
            self.entries.append(entry)
    return MockWAL()


# ── CRDT merge tests ────────────────────────────────────────────


class TestCRDTMerge:
    def test_crdt_merge_higher_score_wins(self, mock_table_64):
        """When remote has higher fitness, remote vector should win."""
        local = mock_table_64
        agent_id = 100
        local_fitness = local._meta[agent_id].fitness
        assert local_fitness == pytest.approx(0.5, abs=0.01)

        # Remote dict with higher fitness
        remote_vec = np.random.randn(64).astype(np.float32).tolist()
        remote = {
            "agents": [
                {
                    "agent_id": agent_id,
                    "vector": remote_vec,
                    "fitness": 0.99,
                    "generation": 2,
                    "capability_mask": 0xFFFF,
                    "thermal_pressure": 0.1,
                    "wall_time": 1.0,
                }
            ]
        }

        merged = MeshVectorGossip.merge_vector_tables(local, remote)
        assert merged._meta[agent_id].fitness == pytest.approx(0.99, abs=0.001)

    def test_crdt_merge_tiebreak_wall_time(self, mock_table_64):
        """When fitness is equal, earlier wall_time wins."""
        local = mock_table_64
        agent_id = 100
        local._meta[agent_id].extra["wall_time"] = 10.0

        local_vec = local._vectors[agent_id].tolist()
        remote_vec = np.random.randn(64).astype(np.float32).tolist()

        remote = {
            "agents": [
                {
                    "agent_id": agent_id,
                    "vector": remote_vec,
                    "fitness": local._meta[agent_id].fitness,
                    "generation": 2,
                    "capability_mask": 0xFFFF,
                    "thermal_pressure": 0.1,
                    "wall_time": 5.0,  # earlier than local 10.0
                }
            ]
        }

        merged = MeshVectorGossip.merge_vector_tables(local, remote)
        # Since remote wall_time (5.0) < local (10.0), remote wins
        merged_vec = merged._vectors[agent_id].tolist()
        assert merged_vec == pytest.approx(remote_vec, abs=1e-5)
        assert merged._meta[agent_id].extra["wall_time"] == 5.0


# ── Gossip round tests ──────────────────────────────────────────


class TestGossipRound:
    def test_gossip_round_merges_remote(self, mock_table_64, mock_wal):
        """A gossip round should merge remote deltas into local table."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
            signed_wal=mock_wal,
            max_peers_per_round=2,
        )

        remote_vec = np.random.randn(64).astype(np.float32).tolist()
        remote_deltas = {
            "agents": [
                {
                    "agent_id": 999,
                    "vector": remote_vec,
                    "fitness": 0.95,
                    "generation": 3,
                    "capability_mask": 0xFFFF,
                    "thermal_pressure": 0.1,
                    "wall_time": 1.0,
                }
            ]
        }

        # Monkey-patch transport to return remote deltas
        gossip._fetch_peer_deltas = lambda peer, digest: remote_deltas

        results = gossip.gossip_round(["ProArt"])
        assert "ProArt" in results
        assert results["ProArt"].merged_count == 1
        assert 999 in mock_table_64._meta
        assert mock_table_64._meta[999].fitness == pytest.approx(0.95, abs=0.001)

    def test_gossip_with_no_peers(self, mock_table_64):
        """gossip_round with empty peer list should return empty dict."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
        )
        results = gossip.gossip_round([])
        assert results == {}

    def test_thermal_aware_routing_rejects_hot_node(self, mock_table_64):
        """A peer above thermal_threshold should be rejected."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
            thermal_threshold=0.80,
        )
        gossip.set_peer_thermal("JetsonClaw1", 0.95)

        results = gossip.gossip_round(["JetsonClaw1"])
        assert results["JetsonClaw1"].thermal_rejected is True
        assert len(results["JetsonClaw1"].errors) >= 1
        assert "thermal" in results["JetsonClaw1"].errors[0].lower()


# ── Digest & bandwidth tests ────────────────────────────────────


class TestDigest:
    def test_digest_reduces_bandwidth(self, mock_table_64):
        """Digest should be smaller than full table representation."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
        )
        # Seed the version vector so digest has some bytes
        for i in range(4):
            gossip.publish_delta(room_id=100 + i, vector=[0.1] * 64, score=0.5, timestamp=1.0)

        digest = gossip.get_digest()
        assert isinstance(digest, GossipDigest)
        assert digest.agent_count == 4

        # Estimate bandwidth saved: digest is a version_vector, not full vectors
        full_size = 4 * (64 * 4 + 64)  # 4 agents, 64 floats (4 bytes each) + metadata
        saved = digest.estimate_bandwidth_saved(full_size)
        assert saved > 0.0
        assert saved < 1.0


# ── Delta batching tests ────────────────────────────────────────


class TestDeltaBatching:
    def test_delta_batching(self, mock_table_64):
        """publish_delta should queue deltas; batch fills at threshold."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
            delta_batch_size=3,
        )
        assert len(gossip._delta_queue) == 0

        for i in range(5):
            gossip.publish_delta(
                room_id=200 + i,
                vector=[0.1] * 64,
                score=0.8,
                timestamp=time.time(),
            )

        # After 5 publishes with batch_size=3, queue should have 5
        assert len(gossip._delta_queue) == 5
        assert gossip._version_vector[200] == 1
        assert gossip._version_vector[204] == 1

    def test_delta_batch_flush(self, mock_table_64):
        """clear_delta_queue should flush and return count."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
        )
        for i in range(3):
            gossip.publish_delta(300 + i, [0.2] * 64, 0.7)
        flushed = gossip.clear_delta_queue()
        assert flushed == 3
        assert len(gossip._delta_queue) == 0


# ── Rebirth tests ───────────────────────────────────────────────


class TestRebirth:
    def test_rebirth_from_mesh_vector(self, mock_table_64, mock_wal):
        """A vector received via gossip can be used for rebirth on this node.

        Simulates: Oracle1 goes down; ProArt had previously gossipped the
        vector.  A new node (Alibaba) joins, gossips with ProArt, and
        recovers the orphaned vector.
        """
        # Simulate Oracle1 original vector
        orphan_vec = np.random.randn(64).astype(np.float32).tolist()
        orphan_agent_id = 777

        # ProArt table has the orphan
        proart_table = FluxVectorTable(dim=64, bit_width=4)
        proart_table.add(
            AgentVector(
                agent_id=orphan_agent_id,
                vector=orphan_vec,
                fitness=0.91,
                generation=5,
                capability_mask=0xABCD,
                thermal_pressure=0.1,
            )
        )

        # Alibaba new node, empty table
        alibaba_table = FluxVectorTable(dim=64, bit_width=4)
        alibaba_gossip = MeshVectorGossip(
            node_id="Alibaba",
            local_table=alibaba_table,
        )

        # Simulate Alibaba pulling from ProArt via gossip round
        remote_deltas = {
            "agents": [
                {
                    "agent_id": orphan_agent_id,
                    "vector": orphan_vec,
                    "fitness": 0.91,
                    "generation": 5,
                    "capability_mask": 0xABCD,
                    "thermal_pressure": 0.1,
                    "wall_time": 42.0,
                }
            ]
        }

        # Use the internal method that also updates version vector
        alibaba_gossip._apply_remote_deltas(remote_deltas, peer_id="ProArt")

        assert orphan_agent_id in alibaba_table._meta
        assert alibaba_table._meta[orphan_agent_id].fitness == pytest.approx(0.91, abs=0.001)
        assert alibaba_table._meta[orphan_agent_id].capability_mask == 0xABCD
        np.testing.assert_allclose(
            alibaba_table._vectors[orphan_agent_id],
            np.array(orphan_vec, dtype=np.float32),
            atol=1e-5,
        )

        # Verify digest reflects the reborn agent
        digest = alibaba_gossip.get_digest()
        assert digest.agent_count == 1
        assert digest.version_vector[orphan_agent_id] >= 1


# ── WAL integration test ───────────────────────────────────────


class TestWALIntegration:
    def test_wal_callback_receives_gossip_event(self, mock_table_64):
        """When a gossip round completes, WAL callback should be invoked."""
        received = []

        def callback(op, agent_id, meta):
            received.append((op, agent_id, meta))

        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
            wal_callback=callback,
        )

        # Inject a remote delta so merge happens
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
        gossip._fetch_peer_deltas = lambda peer, digest: remote_deltas

        gossip.gossip_round(["ProArt"])

        assert len(received) >= 1
        assert received[0][0] == "gossip_delta"
        assert received[0][2]["peer_id"] == "ProArt"
        assert received[0][2]["merged"] == 1


# ── Peer selection tests ────────────────────────────────────────


class TestPeerSelection:
    def test_select_peers_respects_max(self, mock_table_64):
        """_select_peers should not return more than max_peers_per_round."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
            max_peers_per_round=2,
        )
        peers = [f"peer_{i}" for i in range(10)]
        selected = gossip._select_peers(peers)
        assert len(selected) == 2
        assert all(p in peers for p in selected)

    def test_get_mesh_wide_vectors_filters(self, mock_table_64):
        """get_mesh_wide_vectors should respect min_fitness and max_thermal."""
        gossip = MeshVectorGossip(
            node_id="Oracle1",
            local_table=mock_table_64,
        )
        # Mock table: 100=0.5, 101=0.6, 102=0.7, 103=0.8 fitness; all 0.2 thermal
        # Only agent 103 should pass min_fitness=0.75
        results = gossip.get_mesh_wide_vectors(min_fitness=0.75, max_thermal=0.5)
        assert len(results) == 1
        assert results[0]["agent_id"] == 103
        assert results[0]["fitness"] == pytest.approx(0.8, abs=0.01)
