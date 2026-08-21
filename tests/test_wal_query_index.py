"""Tests for WALIndex compound queries and WALBatchQuery filter-based queries."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from logos.signed_wal import SignedWAL, WALEntry
from logos.wal_index import WALIndex
from logos.wal_query import WALBatchQuery, WALQueryFilter, WALQueryIndex


@pytest.fixture
def wal_with_entries(tmp_path):
    """WAL with diverse entries for querying."""
    path = tmp_path / "query.wal.jsonl"
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
    return wal


# ── WALIndex compound queries ───────────────────────────────


class TestWALIndex:
    def test_rebuild_indexes_all(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        assert len(idx.by_type) == 5  # spawn, breed, sunset, tick, flux_violation
        assert idx.by_type["spawn"] == [0, 1]
        assert idx.by_type["breed"] == [2]

    def test_query_by_event_type(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        results = idx.query(filters=[{"field": "event_type", "value": "spawn"}])
        assert len(results) == 2
        assert results[0].entry.operation == "spawn"

    def test_query_by_node_id(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        results = idx.query(filters=[{"field": "node_id", "value": "node-alpha"}])
        assert len(results) == 3

    def test_query_by_room_id(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        results = idx.query(filters=[{"field": "room_id", "value": "forge"}])
        assert len(results) == 3

    def test_compound_and(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        results = idx.query(
            conjunction="and",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "room_id", "value": "crucible"},
            ],
        )
        assert len(results) == 1
        assert results[0].entry.agent_id == 2

    def test_compound_or(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        results = idx.query(
            conjunction="or",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "event_type", "value": "sunset"},
            ],
        )
        assert len(results) == 3

    def test_time_range(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        base = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        results = idx.query(
            filters=[
                {
                    "field": "time_range",
                    "start": base.isoformat().replace("+00:00", "Z"),
                    "end": base.replace(hour=14).isoformat().replace("+00:00", "Z"),
                }
            ]
        )
        assert len(results) == 2

    def test_update_increments_index(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        new_entry = WALEntry(
            timestamp=time.time(),
            agent_id=5,
            operation="spawn",
            vector_hash="g" * 64,
            parent_ids=[],
            generation=0,
            node_id="node-delta",
            room_id="pool",
        )
        wal.append(new_entry)
        idx.update(wal.entries[-1])
        assert "node-delta" in idx.by_node
        assert len(idx.by_type["spawn"]) == 3

    def test_all_accessors(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        assert set(idx.all_event_types()) == {
            "spawn",
            "breed",
            "sunset",
            "tick",
            "flux_violation",
        }
        assert set(idx.all_nodes()) == {"node-alpha", "node-beta", "node-gamma"}
        assert set(idx.all_rooms()) == {"forge", "crucible", "archive"}

    def test_repr(self, wal_with_entries):
        wal = wal_with_entries
        idx = WALIndex(wal)
        assert "entries=6" in repr(idx)


# ── WALQueryIndex (secondary indexes) ───────────────────────


class TestWALQueryIndex:
    def test_hint_agent(self, wal_with_entries):
        wal = wal_with_entries
        qidx = WALQueryIndex()
        qidx.rebuild(wal.entries)
        assert qidx.hint_agent(1) == [0, 2, 4]
        assert qidx.hint_agent(99) == []

    def test_hint_operation(self, wal_with_entries):
        wal = wal_with_entries
        qidx = WALQueryIndex()
        qidx.rebuild(wal.entries)
        assert sorted(qidx.hint_operation("spawn")) == [0, 1]

    def test_hint_time_range(self, wal_with_entries):
        wal = wal_with_entries
        qidx = WALQueryIndex()
        qidx.rebuild(wal.entries)
        base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        results = qidx.hint_time_range(base_ts, base_ts + 4000)
        assert results == [0, 1]

    def test_hint_generation_range(self, wal_with_entries):
        wal = wal_with_entries
        qidx = WALQueryIndex()
        qidx.rebuild(wal.entries)
        assert qidx.hint_generation_range(0, 0) == [0, 1, 3, 5]
        assert qidx.hint_generation_range(1, 1) == [2, 4]

    def test_plan_picks_smallest(self, wal_with_entries):
        wal = wal_with_entries
        qidx = WALQueryIndex()
        qidx.rebuild(wal.entries)
        # agent 3 only has 1 entry vs agent 1 has 3
        filt = WALQueryFilter(agent_id=3, operation="sunset")
        plan = qidx.plan(filt)
        # Should pick the smallest candidate set
        assert plan is not None
        assert len(plan) <= 3

    def test_plan_none_for_empty_filter(self, wal_with_entries):
        wal = wal_with_entries
        qidx = WALQueryIndex()
        qidx.rebuild(wal.entries)
        filt = WALQueryFilter()
        assert qidx.plan(filt) is None


# ── WALBatchQuery filter-based queries ──────────────────────


class TestWALBatchQuery:
    def test_filter_by_agent(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.filter(WALQueryFilter(agent_id=1))
        assert len(results) == 3
        assert results[0].entry.operation == "spawn"

    def test_filter_by_operation(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.filter(WALQueryFilter(operation="spawn"))
        assert len(results) == 2

    def test_filter_by_node(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.filter(WALQueryFilter(node_id="node-alpha"))
        assert len(results) == 3

    def test_filter_by_room(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.filter(WALQueryFilter(room_id="forge"))
        assert len(results) == 3

    def test_filter_compound(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        filt = WALQueryFilter(agent_id=1, operation="breed")
        results = bq.filter(filt)
        assert len(results) == 1
        assert results[0].entry.operation == "breed"

    def test_filter_time_range(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        filt = WALQueryFilter(time_start=base_ts, time_end=base_ts + 4000)
        results = bq.filter(filt)
        assert len(results) == 2

    def test_filter_generation_range(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        filt = WALQueryFilter(generation_min=1, generation_max=1)
        results = bq.filter(filt)
        assert len(results) == 2
        assert all(se.entry.generation == 1 for se in results)

    def test_filter_parent_ids(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        filt = WALQueryFilter(parent_ids={1})
        results = bq.filter(filt)
        assert len(results) == 1
        assert results[0].entry.operation == "breed"

    def test_filter_custom_callback(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        filt = WALQueryFilter(custom=lambda e: e.agent_id > 2)
        results = bq.filter(filt)
        assert len(results) == 2  # agent 3 and 4

    def test_filter_limit_offset(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.filter(WALQueryFilter(), limit=2, offset=1)
        assert len(results) == 2
        # chronological order, offset 1 means skip first entry
        assert results[0].entry.agent_id == 2  # second entry

    def test_count(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        assert bq.count(WALQueryFilter(operation="spawn")) == 2
        assert bq.count(WALQueryFilter()) == 6

    def test_by_agent(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        assert len(bq.by_agent(1)) == 3
        assert len(bq.by_agent(99)) == 0

    def test_by_operation(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        assert len(bq.by_operation("spawn")) == 2

    def test_by_time_range(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        assert len(bq.by_time_range(base_ts, base_ts + 4000)) == 2

    def test_by_agent_time_range(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        results = bq.by_agent_time_range(1, base_ts, base_ts + 8000)
        assert len(results) == 2  # spawn + breed

    def test_latest_by_agent(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.latest_by_agent(1, n=1)
        assert len(results) == 1
        assert results[0].entry.operation == "tick"

    def test_descendants(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.descendants(1)
        assert len(results) == 1
        assert results[0].entry.operation == "breed"
        assert results[0].entry.parent_ids == [1]

    def test_genealogy(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.genealogy(1)
        # own: spawn, breed, tick (3) + descendants: breed (already counted)
        assert len(results) == 3
        ops = [se.entry.operation for se in results]
        assert "spawn" in ops
        assert "breed" in ops
        assert "tick" in ops

    def test_batch_verify(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        verified, failed, failed_idx = bq.batch_verify()
        assert verified == 6
        assert failed == 0
        assert failed_idx == []

    def test_batch_verify_subset(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        filt = WALQueryFilter(operation="spawn")
        verified, failed, failed_idx = bq.batch_verify(filt)
        assert verified == 2
        assert failed == 0

    def test_range_scan(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        results = bq.range_scan(1, 4)
        assert len(results) == 3
        assert results[0].entry.agent_id == 2

    def test_len(self, wal_with_entries):
        wal = wal_with_entries
        bq = WALBatchQuery(wal.entries)
        assert len(bq) == 6

    def test_auto_rebuild_index(self, wal_with_entries):
        wal = wal_with_entries
        # Pass entries with no pre-built index
        bq = WALBatchQuery(wal.entries)
        # Internal index should have been auto-rebuilt
        assert len(bq._index) == 6

    def test_filter_with_index_hints(self, wal_with_entries):
        wal = wal_with_entries
        qidx = WALQueryIndex()
        qidx.rebuild(wal.entries)
        bq = WALBatchQuery(wal.entries, qidx)
        results = bq.filter(WALQueryFilter(agent_id=1))
        assert len(results) == 3
