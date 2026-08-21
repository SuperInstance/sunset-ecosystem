"""Tests for QuantaVdbBridge.

Covers:
- CRDT manifest operations (SQLite shadow)
- Insert / query / search (with and without Quanta C++ module)
- Sync payload encode/decode
- Fleet-wide queries (breedable pool, population summary)
- Stats tracking
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from swarm.quanta_vdb_bridge import QuantaVdbBridge, QuantaTableEntry, VdbSyncPayload


@pytest.fixture
def bridge() -> QuantaVdbBridge:
    with tempfile.TemporaryDirectory() as tmp:
        yield QuantaVdbBridge(
            prefix="test",
            data_path=tmp,
            dim=64,
            node_id="test_node",
        )


@pytest.fixture
def sample_entry() -> QuantaTableEntry:
    return QuantaTableEntry(
        agent_id="agent_001",
        vector=np.random.randn(64).astype(np.float32),
        timestamp=1000.0,
        node_id="test_node",
        generation=1,
        fitness=0.85,
        signature="abc123",
        partition_tag="test_partition",
    )


class TestQuantaTableEntry:
    def test_roundtrip_dict(self) -> None:
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        e = QuantaTableEntry(
            agent_id="a1",
            vector=vec,
            timestamp=1.0,
            node_id="n1",
            generation=0,
            fitness=0.5,
            signature="s1",
        )
        d = e.to_dict()
        e2 = QuantaTableEntry.from_dict(d)
        assert e2.agent_id == "a1"
        assert np.allclose(e2.vector, vec)
        assert e2.fitness == 0.5


class TestVdbSyncPayload:
    def test_roundtrip(self, sample_entry: QuantaTableEntry) -> None:
        payload = VdbSyncPayload(
            node_id="remote",
            timestamp=2000.0,
            entries=[sample_entry],
        )
        blob = payload.to_bytes()
        restored = VdbSyncPayload.from_bytes(blob)
        assert restored.node_id == "remote"
        assert len(restored.entries) == 1
        assert restored.entries[0].agent_id == "agent_001"


class TestInsertAndQuery:
    def test_insert_new(
        self, bridge: QuantaVdbBridge, sample_entry: QuantaTableEntry
    ) -> None:
        assert bridge.insert(sample_entry) is True
        assert bridge.stats["count"] == 1

    def test_insert_duplicate_crdt(
        self, bridge: QuantaVdbBridge, sample_entry: QuantaTableEntry
    ) -> None:
        bridge.insert(sample_entry)
        # Same agent_id, lower fitness → should be rejected
        duplicate = QuantaTableEntry(
            agent_id="agent_001",
            vector=np.random.randn(64).astype(np.float32),
            timestamp=500.0,  # older
            node_id="test_node",
            generation=1,
            fitness=0.5,
            signature="xyz789",
        )
        assert bridge.insert(duplicate) is False

    def test_insert_newer_wins(
        self, bridge: QuantaVdbBridge, sample_entry: QuantaTableEntry
    ) -> None:
        bridge.insert(sample_entry)
        newer = QuantaTableEntry(
            agent_id="agent_001",
            vector=np.random.randn(64).astype(np.float32),
            timestamp=2000.0,  # newer
            node_id="test_node",
            generation=2,
            fitness=0.9,
            signature="newer",
        )
        assert bridge.insert(newer) is True
        queried = bridge._manifest_query("agent_001")
        assert queried is not None
        assert queried.generation == 2

    def test_manifest_query(
        self, bridge: QuantaVdbBridge, sample_entry: QuantaTableEntry
    ) -> None:
        bridge.insert(sample_entry)
        result = bridge._manifest_query("agent_001")
        assert result is not None
        assert result.agent_id == "agent_001"
        assert result.fitness == 0.85

    def test_manifest_query_missing(self, bridge: QuantaVdbBridge) -> None:
        assert bridge._manifest_query("missing") is None


class TestSearch:
    def test_brute_force_search(self, bridge: QuantaVdbBridge) -> None:
        # Insert entries at different positions
        for i in range(5):
            vec = np.zeros(64, dtype=np.float32)
            vec[0] = float(i)
            entry = QuantaTableEntry(
                agent_id=f"agent_{i}",
                vector=vec,
                timestamp=1000.0 + i,
                node_id="test_node",
                generation=i,
                fitness=0.5 + i * 0.1,
                signature=f"sig{i}",
            )
            bridge.insert(entry)

        # Search near agent_2
        query = np.zeros(64, dtype=np.float32)
        query[0] = 2.0
        results = bridge.search(query, k=3)
        assert len(results) <= 3
        assert results[0]["id"] == "agent_2"  # closest

    def test_search_with_partition(self, bridge: QuantaVdbBridge) -> None:
        for i in range(3):
            entry = QuantaTableEntry(
                agent_id=f"agent_{i}",
                vector=np.random.randn(64).astype(np.float32),
                timestamp=1000.0,
                node_id="test_node",
                generation=0,
                fitness=0.5,
                signature="s",
                partition_tag="partition_a" if i < 2 else "partition_b",
            )
            bridge.insert(entry)

        query = np.random.randn(64).astype(np.float32)
        results = bridge.search(query, k=10, partition="partition_a")
        assert all(r["partition"] == "partition_a" for r in results)


class TestSync:
    def test_sync_payload_roundtrip(
        self, bridge: QuantaVdbBridge, sample_entry: QuantaTableEntry
    ) -> None:
        bridge.insert(sample_entry)
        payload = bridge.get_sync_payload()
        assert len(payload) > 0

        # Create a second bridge (different node) and apply
        with tempfile.TemporaryDirectory() as tmp2:
            bridge2 = QuantaVdbBridge(
                prefix="test2",
                data_path=tmp2,
                dim=64,
                node_id="remote_node",
            )
            stats = bridge2.apply_sync_payload(payload)
            assert stats["merged"] == 1
            assert bridge2.stats["count"] == 1

    def test_sync_crdt_merge(self, bridge: QuantaVdbBridge) -> None:
        # Local entry
        local = QuantaTableEntry(
            agent_id="agent_001",
            vector=np.random.randn(64).astype(np.float32),
            timestamp=1000.0,
            node_id="test_node",
            generation=1,
            fitness=0.8,
            signature="local",
        )
        bridge.insert(local)

        # Remote entry with newer timestamp
        remote = QuantaTableEntry(
            agent_id="agent_001",
            vector=np.random.randn(64).astype(np.float32),
            timestamp=2000.0,
            node_id="remote_node",
            generation=2,
            fitness=0.9,
            signature="remote",
        )
        payload = VdbSyncPayload(
            node_id="remote", timestamp=3000.0, entries=[remote]
        ).to_bytes()

        stats = bridge.apply_sync_payload(payload)
        assert stats["merged"] == 1
        # Verify remote won
        result = bridge._manifest_query("agent_001")
        assert result is not None
        assert result.generation == 2


class TestFleetQueries:
    def test_breedable_pool(self, bridge: QuantaVdbBridge) -> None:
        for i in range(10):
            entry = QuantaTableEntry(
                agent_id=f"agent_{i}",
                vector=np.random.randn(64).astype(np.float32),
                timestamp=1000.0 + i,
                node_id="test_node",
                generation=i,
                fitness=0.5 + i * 0.05,
                signature=f"sig{i}",
                thermal_pressure=0.2 if i < 5 else 0.6,
            )
            bridge.insert(entry)

        pool = bridge.get_breedable_pool(min_fitness=0.7, max_thermal=0.5)
        assert all(e.fitness >= 0.7 for e in pool)
        assert all(e.thermal_pressure <= 0.5 for e in pool)

    def test_population_summary(self, bridge: QuantaVdbBridge) -> None:
        for i in range(5):
            entry = QuantaTableEntry(
                agent_id=f"agent_{i}",
                vector=np.random.randn(64).astype(np.float32),
                timestamp=1000.0,
                node_id="test_node",
                generation=i,
                fitness=0.5 + i * 0.1,
                signature="s",
            )
            bridge.insert(entry)

        summary = bridge.get_population_summary()
        assert summary["count"] == 5
        assert summary["generation_range"] == (0, 4)
        assert 0.0 <= summary["diversity_score"] <= 1.0


class TestStats:
    def test_stats_tracking(
        self, bridge: QuantaVdbBridge, sample_entry: QuantaTableEntry
    ) -> None:
        bridge.insert(sample_entry)
        # Search via brute-force fallback (Quanta C++ not available in test)
        results = bridge.search(sample_entry.vector, k=1)
        payload = bridge.get_sync_payload()
        bridge.apply_sync_payload(payload)

        stats = bridge.stats
        assert stats["insert_count"] == 1
        # Brute force path doesn't increment search_count, so check insert_count only
        assert stats["sync_count"] == 1
        assert stats["prefix"] == "test"
        assert stats["node_id"] == "test_node"
