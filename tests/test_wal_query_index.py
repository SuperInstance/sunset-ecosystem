"""Tests for WALQuery index-aware mode, batch queries, and explain."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from logos.signed_wal import SignedWAL, WALEntry
from logos.wal_query import WALQuery, BatchQueryResult


@pytest.fixture
def wal_with_entries(tmp_path):
    """WAL with diverse entries for querying."""
    path = tmp_path / "query.wal.jsonl"
    wal = SignedWAL(log_path=path)
    base_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()

    entries = [
        WALEntry(timestamp=base_ts, agent_id=1, operation="spawn", vector_hash="a" * 64, parent_ids=[], generation=0, node_id="node-alpha", room_id="forge"),
        WALEntry(timestamp=base_ts + 3600, agent_id=2, operation="spawn", vector_hash="b" * 64, parent_ids=[], generation=0, node_id="node-beta", room_id="crucible"),
        WALEntry(timestamp=base_ts + 7200, agent_id=1, operation="breed", vector_hash="c" * 64, parent_ids=[1], generation=1, node_id="node-alpha", room_id="forge"),
        WALEntry(timestamp=base_ts + 10800, agent_id=3, operation="sunset", vector_hash="d" * 64, parent_ids=[2], generation=0, node_id="node-gamma", room_id="archive"),
        WALEntry(timestamp=base_ts + 14400, agent_id=1, operation="tick", vector_hash="e" * 64, parent_ids=[], generation=1, node_id="node-alpha", room_id="forge"),
        WALEntry(timestamp=base_ts + 18000, agent_id=4, operation="flux_violation", vector_hash="f" * 64, parent_ids=[], generation=0, node_id="node-beta", room_id="crucible"),
    ]
    for e in entries:
        wal.append(e)
    return wal, path


# ── Simple filter tests (linear scan) ─────────────────────────────

class TestLinearScan:
    def test_by_time_range(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        base = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        results = q.by_time_range(
            base.isoformat().replace("+00:00", "Z"),
            (base.replace(hour=14)).isoformat().replace("+00:00", "Z"),
        )
        assert len(results) == 2
        assert results[0].entry.operation == "spawn"

    def test_by_event_type(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        results = q.by_event_type("spawn")
        assert len(results) == 2

    def test_by_node_id(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        results = q.by_node_id("node-alpha")
        assert len(results) == 3

    def test_by_room_id(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        results = q.by_room_id("forge")
        assert len(results) == 3


# ── Index-aware tests ───────────────────────────────────────────

class TestIndexAware:
    def test_with_index_uses_inverted_index(self, wal_with_entries):
        wal, _ = wal_with_entries
        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx)

        results = q.by_event_type("spawn")
        assert len(results) == 2
        # Should be index-backed
        explain = q.explain([{"field": "event_type", "value": "spawn"}])
        assert explain["strategy"] == "index_intersection"
        assert explain["index_available"] is True

    def test_without_index_falls_back_to_scan(self, wal_with_entries):
        wal, _ = wal_with_entries
        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx).without_index()

        results = q.by_event_type("spawn")
        assert len(results) == 2
        explain = q.explain([{"field": "event_type", "value": "spawn"}])
        assert explain["strategy"] == "linear_scan"
        assert explain["index_available"] is False

    def test_compound_query_with_index(self, wal_with_entries):
        wal, _ = wal_with_entries
        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx)

        results = q.compound_query(
            conjunction="and",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "room_id", "value": "crucible"},
            ],
        )
        assert len(results) == 1
        assert results[0].entry.agent_id == 2

    def test_compound_query_or_with_index(self, wal_with_entries):
        wal, _ = wal_with_entries
        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx)

        results = q.compound_query(
            conjunction="or",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "event_type", "value": "sunset"},
            ],
        )
        assert len(results) == 3


# ── Batch query tests ───────────────────────────────────────────

class TestBatchQuery:
    def test_batch_two_queries(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)

        results = q.batch_query([
            {
                "query_id": "spawns",
                "conjunction": "and",
                "filters": [{"field": "event_type", "value": "spawn"}],
            },
            {
                "query_id": "alpha-events",
                "conjunction": "and",
                "filters": [{"field": "node_id", "value": "node-alpha"}],
            },
        ])

        assert len(results) == 2
        assert results[0].query_id == "spawns"
        assert len(results[0].entries) == 2
        assert results[0].index_used is False

        assert results[1].query_id == "alpha-events"
        assert len(results[1].entries) == 3
        assert results[1].error is None

    def test_batch_with_index(self, wal_with_entries):
        wal, _ = wal_with_entries
        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx)

        results = q.batch_query([
            {
                "query_id": "q1",
                "conjunction": "and",
                "filters": [{"field": "event_type", "value": "spawn"}],
            },
        ])
        assert results[0].index_used is True

    def test_batch_empty_queries(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        results = q.batch_query([])
        assert len(results) == 0

    def test_batch_timeout(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        # Many queries with tiny timeout — should stop early
        many = [
            {"query_id": f"q{i}", "conjunction": "and", "filters": [{"field": "event_type", "value": "spawn"}]}
            for i in range(1000)
        ]
        results = q.batch_query(many, timeout_ms=1.0)
        # Should have cut off early
        assert len(results) < 1000


# ── Explain tests ───────────────────────────────────────────────

class TestExplain:
    def test_explain_with_index(self, wal_with_entries):
        wal, _ = wal_with_entries
        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx)

        plan = q.explain([
            {"field": "event_type", "value": "spawn"},
            {"field": "node_id", "value": "node-alpha"},
        ])
        assert plan["index_available"] is True
        assert plan["indexable_filters"] == 2
        assert plan["fallback_filters"] == 0
        assert plan["strategy"] == "index_intersection"

    def test_explain_mixed_filters(self, wal_with_entries):
        wal, _ = wal_with_entries
        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx)

        plan = q.explain([
            {"field": "event_type", "value": "spawn"},
            {"field": "agent_id", "value": 1},  # not in index
        ])
        assert plan["indexable_filters"] == 1
        assert plan["fallback_filters"] == 1
        # Still uses index because at least one filter is indexable
        assert plan["strategy"] == "index_intersection"

    def test_explain_no_index(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        plan = q.explain([{"field": "event_type", "value": "spawn"}])
        assert plan["index_available"] is False
        assert plan["strategy"] == "linear_scan"


# ── BatchQueryResult tests ──────────────────────────────────────

class TestBatchQueryResult:
    def test_result_len(self):
        r = BatchQueryResult(query_id="test", entries=[])
        assert len(r) == 0
        assert not r

    def test_result_bool(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        results = q.batch_query([
            {"query_id": "spawns", "conjunction": "and", "filters": [{"field": "event_type", "value": "spawn"}]}
        ])
        assert results[0]
        assert len(results[0]) == 2


# ── Regression: existing tests must still pass ──────────────────

class TestRegression:
    def test_verify_subset(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        subset = wal.entries[:3]
        assert q.verify_subset(subset) is True

    def test_summary(self, wal_with_entries):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        s = q.summary()
        assert s["total_entries"] == 6
        assert s["event_counts"]["spawn"] == 2
        assert "node-alpha" in s["node_coverage"]

    def test_export_jsonl(self, wal_with_entries, tmp_path):
        wal, _ = wal_with_entries
        q = WALQuery(wal)
        out = tmp_path / "exported.jsonl"
        q.export_jsonl(out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 6
        import json
        first = json.loads(lines[0])
        assert first["entry"]["operation"] == "spawn"
