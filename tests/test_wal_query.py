"""Tests for WALQueryIndex and WALBatchQuery.

Covers index building, query planning, batch verification, and range scans.
"""

from __future__ import annotations

import time

import pytest

from logos.signed_wal import SignedWAL, WALEntry
from logos.wal_query import WALQueryIndex, WALBatchQuery, WALQueryFilter


@pytest.fixture
def wal():
    w = SignedWAL(algorithm="hmac-sha256")
    # Seed with diverse entries
    for i in range(20):
        w.append(
            WALEntry(
                timestamp=time.time() - (20 - i) * 60,  # spaced 1 min apart
                agent_id=i % 5,  # agents 0-4
                operation=["spawn", "breed", "sunset", "mutate"][i % 4],
                vector_hash=f"hash-{i}",
                parent_ids=[i - 1] if i > 0 else [],
                generation=i // 5,
                node_id=f"node-{i % 3}",
                room_id=f"room-{i % 4}",
            )
        )
    return w


@pytest.fixture
def query(wal):
    idx = WALQueryIndex()
    idx.rebuild(wal.entries)
    return WALBatchQuery(wal.entries, idx)


# ── WALQueryIndex ─────────────────────────────────────────


def test_index_rebuild_counts(wal):
    idx = WALQueryIndex()
    idx.rebuild(wal.entries)
    assert len(idx) == 20
    assert len(idx.by_agent[0]) == 4  # agents 0, 5, 10, 15


def test_hint_agent(wal, query):
    indices = query._index.hint_agent(0)
    assert len(indices) == 4
    for i in indices:
        assert wal.entries[i].entry.agent_id == 0


def test_hint_operation(wal, query):
    spawn_indices = query._index.hint_operation("spawn")
    assert len(spawn_indices) == 5  # indices 0, 4, 8, 12, 16
    for i in spawn_indices:
        assert wal.entries[i].entry.operation == "spawn"


def test_hint_time_range(wal, query):
    now = time.time()
    # Query last 5 minutes
    indices = query._index.hint_time_range(now - 300, now)
    assert len(indices) >= 4  # at least the last 4 entries


def test_hint_generation_range(wal, query):
    indices = query._index.hint_generation_range(0, 1)
    assert len(indices) == 10  # generations 0 and 1


def test_plan_picks_smallest(wal, query):
    # agent_id=0 has 4 entries; operation="spawn" has 5
    # plan should pick agent hint (smaller)
    filt = WALQueryFilter(agent_id=0, operation="spawn")
    plan = query._index.plan(filt)
    assert plan is not None
    assert len(plan) == 4  # agent hint is smaller


# ── WALBatchQuery.filter ──────────────────────────────────


def test_filter_by_agent(query):
    results = query.filter(WALQueryFilter(agent_id=2))
    assert len(results) == 4
    for se in results:
        assert se.entry.agent_id == 2


def test_filter_by_operation(query):
    results = query.filter(WALQueryFilter(operation="breed"))
    assert len(results) == 5
    for se in results:
        assert se.entry.operation == "breed"


def test_filter_by_time_range(query):
    now = time.time()
    results = query.filter(WALQueryFilter(time_start=now - 200, time_end=now))
    assert len(results) >= 3
    for se in results:
        assert now - 200 <= se.entry.timestamp <= now


def test_filter_by_agent_and_operation(query):
    results = query.filter(WALQueryFilter(agent_id=1, operation="sunset"))
    # agent 1 entries at indices 1, 6, 11, 16; operation "sunset" at 3, 7, 11, 15
    # intersection: index 11 only
    assert len(results) == 1
    assert results[0].entry.agent_id == 1
    assert results[0].entry.operation == "sunset"


def test_filter_with_limit(query):
    results = query.filter(WALQueryFilter(operation="spawn"), limit=3)
    assert len(results) == 3


def test_filter_with_offset(query):
    results = query.filter(WALQueryFilter(agent_id=0), offset=2)
    all_results = query.filter(WALQueryFilter(agent_id=0))
    assert results == all_results[2:]


def test_filter_no_match(query):
    results = query.filter(WALQueryFilter(agent_id=999))
    assert results == []


def test_filter_chronological_order(query):
    results = query.filter(WALQueryFilter(operation="spawn"))
    indices = [i for i, se in enumerate(query._entries) if se in results]
    assert indices == sorted(indices)


def test_filter_by_generation_range(query):
    results = query.filter(WALQueryFilter(generation_min=1, generation_max=2))
    assert len(results) == 10  # generations 1 and 2
    for se in results:
        assert 1 <= se.entry.generation <= 2


def test_filter_by_parent_ids(query):
    # Entry 5 has parent_id 4; entry 10 has parent_id 9
    results = query.filter(WALQueryFilter(parent_ids={4}))
    assert len(results) >= 1
    for se in results:
        assert 4 in se.entry.parent_ids


def test_filter_by_node(query):
    results = query.filter(WALQueryFilter(node_id="node-0"))
    # entries at indices 0, 3, 6, 9, 12, 15, 18 have node-0
    assert len(results) == 7


def test_filter_custom_predicate(query):
    results = query.filter(
        WALQueryFilter(custom=lambda e: e.generation >= 2 and e.operation == "spawn")
    )
    for se in results:
        assert se.entry.generation >= 2
        assert se.entry.operation == "spawn"


# ── convenience queries ───────────────────────────────────


def test_by_agent(query):
    results = query.by_agent(3)
    assert len(results) == 4


def test_by_operation(query):
    results = query.by_operation("mutate")
    assert len(results) == 5


def test_by_time_range(query):
    now = time.time()
    results = query.by_time_range(now - 300, now)
    assert len(results) >= 4


def test_by_agent_time_range(query):
    now = time.time()
    results = query.by_agent_time_range(0, now - 300, now)
    assert len(results) <= 4


def test_latest_by_agent(query):
    results = query.latest_by_agent(0, n=2)
    assert len(results) == 2
    # Should be the last 2 entries for agent 0 (indices 15, 20 would be)
    # Actually agent 0 at indices 0, 5, 10, 15
    assert results[-1].entry.agent_id == 0


def test_descendants(query):
    results = query.descendants(2)
    assert len(results) >= 1
    for se in results:
        assert 2 in se.entry.parent_ids


def test_genealogy(query):
    results = query.genealogy(1)
    # Includes agent 1's own entries + entries where 1 is a parent
    assert len(results) >= 4  # at least own entries


# ── count ─────────────────────────────────────────────────


def test_count(query):
    assert query.count(WALQueryFilter(agent_id=0)) == 4
    assert query.count(WALQueryFilter(operation="spawn")) == 5
    assert query.count(WALQueryFilter(agent_id=999)) == 0


# ── batch verify ──────────────────────────────────────────


def test_batch_verify_all(query):
    verified, failed, failed_indices = query.batch_verify()
    assert verified == 20
    assert failed == 0
    assert failed_indices == []


def test_batch_verify_subset(query):
    filt = WALQueryFilter(operation="spawn")
    verified, failed, failed_indices = query.batch_verify(filt)
    assert verified == 5
    assert failed == 0


# ── range scan ────────────────────────────────────────────


def test_range_scan(query):
    results = query.range_scan(5, 10)
    assert len(results) == 5
    assert results[0].entry.agent_id == 0  # index 5
    assert results[-1].entry.agent_id == 4  # index 9


def test_range_scan_empty(query):
    results = query.range_scan(100, 200)
    assert results == []


# ── integration with SignedWAL append ───────────────────


def test_index_keeps_up_with_appends():
    wal = SignedWAL(algorithm="hmac-sha256")
    idx = WALQueryIndex()

    for i in range(10):
        wal.append(
            WALEntry(
                timestamp=time.time(),
                agent_id=i % 3,
                operation="spawn",
                vector_hash=f"h{i}",
                parent_ids=[],
                generation=0,
            )
        )
        idx.append(len(wal.entries) - 1, wal.entries[-1].entry)

    query = WALBatchQuery(wal.entries, idx)
    assert query.count(WALQueryFilter(agent_id=0)) == 4  # indices 0, 3, 6, 9
    assert query.count(WALQueryFilter(operation="spawn")) == 10


# ── edge cases ────────────────────────────────────────────


def test_empty_wal():
    wal = SignedWAL(algorithm="hmac-sha256")
    query = WALBatchQuery(wal.entries)
    assert query.filter(WALQueryFilter()) == []
    assert len(query) == 0


def test_filter_with_all_conditions(query):
    now = time.time()
    results = query.filter(
        WALQueryFilter(
            agent_id=0,
            operation="spawn",
            time_start=now - 2000,
            time_end=now,
            generation_min=0,
            generation_max=1,
            node_id="node-0",
        )
    )
    # agent 0, spawn: index 0 only
    assert len(results) == 1
    assert results[0].entry.agent_id == 0
    assert results[0].entry.operation == "spawn"
