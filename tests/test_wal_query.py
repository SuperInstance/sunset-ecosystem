"""Tests for WALQuery — searchable interface over SignedWAL."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from logos.signed_wal import SignedWAL, WALEntry, SignedEntry
from logos.wal_query import WALQuery, _parse_iso8601, _format_iso8601


@pytest.fixture
def tmp_wal(tmp_path):
    """Provide a SignedWAL backed by a temp file."""
    path = tmp_path / "signed.wal.jsonl"
    wal = SignedWAL(log_path=path)
    return wal, path


@pytest.fixture
def populated_wal(tmp_path):
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


class TestQueryByTimeRange:
    def test_query_by_time_range(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        base = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        # Request 2-hour window starting at base
        start = base.isoformat().replace("+00:00", "Z")
        end = (base.replace(hour=14)).isoformat().replace("+00:00", "Z")
        results = q.by_time_range(start, end)
        assert len(results) == 2
        assert results[0].entry.operation == "spawn"
        assert results[1].entry.operation == "spawn"

    def test_query_by_time_range_empty(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        # Window far in the future
        results = q.by_time_range("2099-01-01T00:00:00Z", "2099-01-02T00:00:00Z")
        assert results == []


class TestQueryByEventType:
    def test_query_by_event_type(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        spawns = q.by_event_type("spawn")
        assert len(spawns) == 2
        assert all(se.entry.operation == "spawn" for se in spawns)

    def test_query_by_event_type_no_match(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        assert q.by_event_type("nonexistent") == []


class TestQueryByNodeId:
    def test_query_by_node_id(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        alpha = q.by_node_id("node-alpha")
        assert len(alpha) == 3
        assert all(se.entry.node_id == "node-alpha" for se in alpha)

    def test_query_by_node_id_empty(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        assert q.by_node_id("node-delta") == []


class TestQueryByRoomId:
    def test_query_by_room_id(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        forge = q.by_room_id("forge")
        assert len(forge) == 3
        assert all(se.entry.room_id == "forge" for se in forge)

    def test_query_by_room_id_empty(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        assert q.by_room_id("void") == []


class TestVerifySubset:
    def test_verify_subset_signatures(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        subset = q.by_event_type("spawn")
        assert q.verify_subset(subset) is True

    def test_verify_subset_tampered(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        subset = q.by_event_type("spawn")
        # Tamper one entry's vector_hash but keep signature
        tampered_entry = WALEntry(
            timestamp=subset[0].entry.timestamp,
            agent_id=subset[0].entry.agent_id,
            operation=subset[0].entry.operation,
            vector_hash="TAMPERED" * 8,
            parent_ids=subset[0].entry.parent_ids,
            generation=subset[0].entry.generation,
        )
        bad = SignedEntry(
            entry=tampered_entry,
            signature=subset[0].signature,
            previous_hash=subset[0].previous_hash,
            public_key=subset[0].public_key,
        )
        assert q.verify_subset([bad]) is False

    def test_verify_subset_chain_break(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        subset = q.by_event_type("spawn")
        # Reorder to break the chain
        reordered = list(reversed(subset))
        assert q.verify_subset(reordered) is False


class TestSummary:
    def test_summary_counts_by_type(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        s = q.summary()
        assert s["total_entries"] == 6
        assert s["event_counts"]["spawn"] == 2
        assert s["event_counts"]["breed"] == 1
        assert s["event_counts"]["sunset"] == 1
        assert s["event_counts"]["tick"] == 1
        assert s["event_counts"]["flux_violation"] == 1

    def test_summary_time_range(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        s = q.summary()
        assert s["time_range"]["start"] is not None
        assert s["time_range"]["end"] is not None
        start_ts = _parse_iso8601(s["time_range"]["start"])
        end_ts = _parse_iso8601(s["time_range"]["end"])
        assert start_ts < end_ts

    def test_summary_coverage(self, populated_wal):
        wal, _ = populated_wal
        q = WALQuery(wal)
        s = q.summary()
        assert "node-alpha" in s["node_coverage"]
        assert "node-beta" in s["node_coverage"]
        assert "node-gamma" in s["node_coverage"]
        assert "forge" in s["room_coverage"]
        assert "crucible" in s["room_coverage"]
        assert "archive" in s["room_coverage"]

    def test_summary_empty_wal(self, tmp_wal):
        wal, _ = tmp_wal
        q = WALQuery(wal)
        s = q.summary()
        assert s["total_entries"] == 0
        assert s["event_counts"] == {}
        assert s["time_range"]["start"] is None


class TestExportJsonl:
    def test_export_all_entries(self, populated_wal, tmp_path):
        wal, _ = populated_wal
        q = WALQuery(wal)
        out = tmp_path / "export.jsonl"
        q.export_jsonl(out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 6
        first = json.loads(lines[0])
        assert first["entry"]["operation"] == "spawn"
        assert "signature" in first

    def test_export_subset(self, populated_wal, tmp_path):
        wal, _ = populated_wal
        q = WALQuery(wal)
        out = tmp_path / "subset.jsonl"
        subset = q.by_event_type("spawn")
        q.export_jsonl(out, entries=subset)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert all(json.loads(line)["entry"]["operation"] == "spawn" for line in lines)

    def test_export_includes_new_fields(self, populated_wal, tmp_path):
        wal, _ = populated_wal
        q = WALQuery(wal)
        out = tmp_path / "export.jsonl"
        q.export_jsonl(out)
        first = json.loads(out.read_text().strip().split("\n")[0])
        assert "node_id" in first["entry"]
        assert "room_id" in first["entry"]


class TestISO8601Helpers:
    def test_parse_iso8601_utc_z(self):
        assert _parse_iso8601("2024-05-25T12:00:00Z") == datetime(2024, 5, 25, 12, 0, 0, tzinfo=timezone.utc).timestamp()

    def test_parse_iso8601_offset(self):
        assert _parse_iso8601("2024-05-25T14:00:00+02:00") == datetime(2024, 5, 25, 12, 0, 0, tzinfo=timezone.utc).timestamp()

    def test_format_roundtrip(self):
        ts = 1716640800.0
        assert _parse_iso8601(_format_iso8601(ts)) == ts
