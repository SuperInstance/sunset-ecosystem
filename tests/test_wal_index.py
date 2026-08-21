"""Tests for WALIndex — in-memory inverted indices for SignedWAL."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from logos.signed_wal import SignedWAL, WALEntry
from logos.wal_index import WALIndex


@pytest.fixture
def populated_wal(tmp_path):
    """WAL with diverse entries for indexing."""
    path = tmp_path / "index.wal.jsonl"
    wal = SignedWAL(log_path=path)
    base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()

    entries = [
        WALEntry(
            timestamp=base_ts,
            agent_id=1,
            operation="spawn",
            vector_hash="a" * 64,
            parent_ids=[],
            generation=0,
            node_id="node-alpha",
            room_id="forge",
        ),
        WALEntry(
            timestamp=base_ts + 3600,
            agent_id=2,
            operation="spawn",
            vector_hash="b" * 64,
            parent_ids=[],
            generation=0,
            node_id="node-beta",
            room_id="crucible",
        ),
        WALEntry(
            timestamp=base_ts + 7200,
            agent_id=1,
            operation="breed",
            vector_hash="c" * 64,
            parent_ids=[1],
            generation=1,
            node_id="node-alpha",
            room_id="forge",
        ),
        WALEntry(
            timestamp=base_ts + 10800,
            agent_id=3,
            operation="sunset",
            vector_hash="d" * 64,
            parent_ids=[2],
            generation=0,
            node_id="node-gamma",
            room_id="archive",
        ),
        WALEntry(
            timestamp=base_ts + 14400,
            agent_id=1,
            operation="tick",
            vector_hash="e" * 64,
            parent_ids=[],
            generation=1,
            node_id="node-alpha",
            room_id="forge",
        ),
        WALEntry(
            timestamp=base_ts + 18000,
            agent_id=4,
            operation="flux_violation",
            vector_hash="f" * 64,
            parent_ids=[],
            generation=0,
            node_id="node-beta",
            room_id="crucible",
        ),
    ]
    for e in entries:
        wal.append(e)
    return wal, path


class TestIndexRebuild:
    def test_index_rebuild_from_wal(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        assert len(idx.by_type["spawn"]) == 2
        assert len(idx.by_type["breed"]) == 1
        assert len(idx.by_node["node-alpha"]) == 3
        assert len(idx.by_room["forge"]) == 3
        # Time bucket check (all entries within same hour, same minute bucket)
        assert len(idx._sorted_ts) == 6

    def test_index_rebuild_after_append(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        wal.append(
            WALEntry(
                timestamp=datetime(
                    2024, 5, 20, 18, 0, 0, tzinfo=timezone.utc
                ).timestamp(),
                agent_id=5,
                operation="spawn",
                vector_hash="g" * 64,
                parent_ids=[],
                generation=0,
                node_id="node-delta",
                room_id="void",
            )
        )
        idx.rebuild()
        assert len(idx.by_type["spawn"]) == 3
        assert "node-delta" in idx.by_node
        assert "void" in idx.by_room


class TestIndexIncrementalUpdate:
    def test_index_incremental_update(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        initial_spawn_count = len(idx.by_type["spawn"])
        new_entry = wal.append(
            WALEntry(
                timestamp=datetime(
                    2024, 5, 20, 18, 0, 0, tzinfo=timezone.utc
                ).timestamp(),
                agent_id=5,
                operation="spawn",
                vector_hash="g" * 64,
                parent_ids=[],
                generation=0,
                node_id="node-delta",
                room_id="void",
            )
        )
        idx.update(new_entry)
        assert len(idx.by_type["spawn"]) == initial_spawn_count + 1
        assert "node-delta" in idx.by_node
        assert "void" in idx.by_room
        assert len(idx._sorted_ts) == 7


class TestIndexQuery:
    def test_compound_and_query(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        results = idx.query(
            conjunction="and",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "room_id", "value": "crucible"},
            ],
        )
        assert len(results) == 1
        assert results[0].entry.operation == "spawn"
        assert results[0].entry.room_id == "crucible"

    def test_compound_or_query(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        results = idx.query(
            conjunction="or",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "event_type", "value": "sunset"},
            ],
        )
        assert len(results) == 3  # 2 spawns + 1 sunset

    def test_time_range_filter(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        base = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        results = idx.query(
            conjunction="and",
            filters=[
                {
                    "field": "time_range",
                    "start": base.isoformat().replace("+00:00", "Z"),
                    "end": (base.replace(hour=14)).isoformat().replace("+00:00", "Z"),
                },
            ],
        )
        assert len(results) == 2

    def test_no_filters_returns_all(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        assert idx.query() == wal.entries


class TestIndexPerformance:
    def test_index_query_performance_1000_entries(self, tmp_path):
        """Index should handle 1000 entries without degradation."""
        path = tmp_path / "perf.wal.jsonl"
        wal = SignedWAL(log_path=path)
        base_ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()

        for i in range(1000):
            e = WALEntry(
                timestamp=base_ts + i * 60,
                agent_id=i % 10,
                operation="spawn" if i % 3 == 0 else "tick",
                vector_hash=f"{i:064x}",
                parent_ids=[],
                generation=0,
                node_id=f"node-{i % 5}",
                room_id=f"room-{i % 4}",
            )
            wal.append(e)

        idx = WALIndex(wal)
        # Query by node
        results = idx.query(
            conjunction="and",
            filters=[{"field": "node_id", "value": "node-2"}],
        )
        assert len(results) == 200  # 1000 / 5

        # Query by time range (first 100 entries = 100 minutes)
        results = idx.query(
            conjunction="and",
            filters=[
                {
                    "field": "time_range",
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-01T01:40:00Z",
                },
            ],
        )
        assert len(results) == 100

        # Compound query
        results = idx.query(
            conjunction="and",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "room_id", "value": "room-0"},
            ],
        )
        # spawn at i%3==0, room-0 at i%4==0 → intersection at i%12==0 → 84 entries (1000/12≈83.3)
        assert len(results) == 84


class TestIndexPersistence:
    def test_index_persists_across_reloads(self, tmp_path):
        """WALIndex rebuilds correctly when WAL is reloaded from disk."""
        path = tmp_path / "persist.wal.jsonl"
        wal = SignedWAL(log_path=path)
        base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()

        for i in range(10):
            wal.append(
                WALEntry(
                    timestamp=base_ts + i * 600,
                    agent_id=i,
                    operation="spawn",
                    vector_hash=f"{i:064x}",
                    parent_ids=[],
                    generation=0,
                    node_id=f"node-{i % 3}",
                    room_id=f"room-{i % 2}",
                )
            )

        idx1 = WALIndex(wal)
        assert len(idx1.by_type["spawn"]) == 10

        # Simulate process restart: new WAL instance reading same file
        wal2 = SignedWAL(log_path=path)
        idx2 = WALIndex(wal2)
        assert len(idx2.by_type["spawn"]) == 10
        assert idx2.by_node.keys() == idx1.by_node.keys()
        assert idx2.by_room.keys() == idx1.by_room.keys()


class TestIndexCorruption:
    def test_index_handles_corrupted_entry(self, tmp_path):
        """WALIndex skips corrupt lines during WAL load."""
        path = tmp_path / "corrupt.wal.jsonl"
        wal = SignedWAL(log_path=path)
        base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()

        wal.append(
            WALEntry(
                timestamp=base_ts,
                agent_id=1,
                operation="spawn",
                vector_hash="a" * 64,
                parent_ids=[],
                generation=0,
            )
        )
        wal.append(
            WALEntry(
                timestamp=base_ts + 3600,
                agent_id=2,
                operation="tick",
                vector_hash="b" * 64,
                parent_ids=[],
                generation=0,
            )
        )

        # Append a corrupt line directly to the file
        with open(path, "a") as f:
            f.write("this is not valid json\n")

        wal2 = SignedWAL(log_path=path)
        # Should have loaded 2 valid entries, skipping the corrupt line
        assert len(wal2.entries) == 2

        idx = WALIndex(wal2)
        assert len(idx.by_type["spawn"]) == 1
        assert len(idx.by_type["tick"]) == 1
        assert (
            idx.query(
                conjunction="and", filters=[{"field": "event_type", "value": "spawn"}]
            )[0].entry.agent_id
            == 1
        )


class TestIndexConvenience:
    def test_all_event_types(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        types = idx.all_event_types()
        assert set(types) == {"spawn", "breed", "sunset", "tick", "flux_violation"}

    def test_all_nodes(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        nodes = idx.all_nodes()
        assert set(nodes) == {"node-alpha", "node-beta", "node-gamma"}

    def test_all_rooms(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        rooms = idx.all_rooms()
        assert set(rooms) == {"forge", "crucible", "archive"}

    def test_repr(self, populated_wal):
        wal, _ = populated_wal
        idx = WALIndex(wal)
        assert "WALIndex" in repr(idx)
        assert "entries=6" in repr(idx)
