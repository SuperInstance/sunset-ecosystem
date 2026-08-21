"""Tests for log_aggregator.py — Centralized log collection and querying.

Run: python3 -m pytest tests/test_log_aggregator.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.log_aggregator import LogAggregator, LogEntry


class TestLogAggregator:
    def test_create(self):
        logs = LogAggregator()
        assert logs.count() == 0

    def test_ingest(self):
        logs = LogAggregator()
        entry = logs.ingest({"level": "ERROR", "message": "boom", "source": "agent-1"})
        assert entry.level == "ERROR"
        assert entry.message == "boom"
        assert entry.source == "agent-1"
        assert logs.count() == 1

    def test_query_level(self):
        logs = LogAggregator()
        logs.ingest({"level": "ERROR", "message": "boom"})
        logs.ingest({"level": "INFO", "message": "ok"})
        logs.ingest({"level": "ERROR", "message": "crash"})
        errors = logs.query(level="ERROR")
        assert len(errors) == 2

    def test_query_source(self):
        logs = LogAggregator()
        logs.ingest({"source": "a", "message": "msg"})
        logs.ingest({"source": "b", "message": "msg"})
        result = logs.query(source="a")
        assert len(result) == 1
        assert result[0].source == "a"

    def test_query_message_contains(self):
        logs = LogAggregator()
        logs.ingest({"message": "connection failed"})
        logs.ingest({"message": "timeout"})
        result = logs.query(message_contains="connection")
        assert len(result) == 1

    def test_query_since_minutes(self):
        logs = LogAggregator()
        logs.ingest({"message": "old", "timestamp": time.time() - 600})
        logs.ingest({"message": "new", "timestamp": time.time()})
        result = logs.query(since_minutes=5)
        assert len(result) == 1
        assert result[0].message == "new"

    def test_query_limit(self):
        logs = LogAggregator()
        for i in range(5):
            logs.ingest({"message": f"msg-{i}"})
        result = logs.query(limit=2)
        assert len(result) == 2
        assert result[-1].message == "msg-4"

    def test_levels(self):
        logs = LogAggregator()
        logs.ingest({"level": "ERROR"})
        logs.ingest({"level": "ERROR"})
        logs.ingest({"level": "INFO"})
        levels = logs.levels()
        assert levels["ERROR"] == 2
        assert levels["INFO"] == 1

    def test_sources(self):
        logs = LogAggregator()
        logs.ingest({"source": "agent-1"})
        logs.ingest({"source": "agent-2"})
        assert logs.sources() == ["agent-1", "agent-2"]

    def test_latest(self):
        logs = LogAggregator()
        logs.ingest({"message": "first"})
        logs.ingest({"message": "second"})
        latest = logs.latest()
        assert latest is not None
        assert latest.message == "second"

    def test_latest_source(self):
        logs = LogAggregator()
        logs.ingest({"message": "a", "source": "s1"})
        logs.ingest({"message": "b", "source": "s2"})
        latest = logs.latest(source="s1")
        assert latest is not None
        assert latest.message == "a"

    def test_max_entries_eviction(self):
        logs = LogAggregator(max_entries=3)
        for i in range(5):
            logs.ingest({"message": f"msg-{i}"})
        assert logs.count() == 3
        assert logs.latest().message == "msg-4"

    def test_source_cleanup_on_eviction(self):
        logs = LogAggregator(max_entries=2)
        logs.ingest({"source": "a", "message": "1"})
        logs.ingest({"source": "b", "message": "2"})
        logs.ingest({"source": "c", "message": "3"})
        assert logs.sources() == ["b", "c"]

    def test_clear(self):
        logs = LogAggregator()
        logs.ingest({"message": "test"})
        logs.clear()
        assert logs.count() == 0
        assert logs.sources() == []

    def test_stats(self):
        logs = LogAggregator()
        logs.ingest({"level": "ERROR", "source": "a"})
        logs.ingest({"level": "INFO", "source": "b"})
        stats = logs.stats()
        assert stats["total_entries"] == 2
        assert stats["sources"] == 2
        assert stats["levels"]["ERROR"] == 1

    def test_repr(self):
        logs = LogAggregator()
        assert "LogAggregator" in repr(logs)
