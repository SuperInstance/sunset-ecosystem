"""Tests for request_deduplicator.py — Request deduplication by key with TTL.

Run: python3 -m pytest tests/test_request_deduplicator.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.request_deduplicator import RequestDeduplicator


class TestRequestDeduplicator:
    def test_create(self):
        dedup = RequestDeduplicator(ttl_sec=60, clock=lambda: 0)
        assert dedup.stats()["size"] == 0

    def test_is_duplicate_first(self):
        dedup = RequestDeduplicator(clock=lambda: 0)
        assert dedup.is_duplicate("req-1") is False

    def test_is_duplicate_second(self):
        dedup = RequestDeduplicator(clock=lambda: 0)
        dedup.is_duplicate("req-1")
        assert dedup.is_duplicate("req-1") is True

    def test_expiration(self):
        dedup = RequestDeduplicator(ttl_sec=10, clock=lambda: 0)
        dedup.is_duplicate("req-1")
        assert dedup.is_duplicate("req-1") is True
        dedup._clock = lambda: 15
        assert dedup.is_duplicate("req-1") is False  # Expired

    def test_forget(self):
        dedup = RequestDeduplicator(clock=lambda: 0)
        dedup.is_duplicate("req-1")
        assert dedup.forget("req-1") is True
        assert dedup.is_duplicate("req-1") is False
        assert dedup.forget("missing") is False

    def test_forget_pattern(self):
        dedup = RequestDeduplicator(clock=lambda: 0)
        dedup.is_duplicate("user:1:action")
        dedup.is_duplicate("user:2:action")
        dedup.is_duplicate("post:1")
        count = dedup.forget_pattern("user:")
        assert count == 2
        assert dedup.is_duplicate("user:1:action") is False
        assert dedup.is_duplicate("post:1") is True

    def test_clear(self):
        dedup = RequestDeduplicator(clock=lambda: 0)
        dedup.is_duplicate("a")
        dedup.is_duplicate("b")
        dedup.clear()
        assert dedup.stats()["size"] == 0

    def test_keys(self):
        dedup = RequestDeduplicator(clock=lambda: 0)
        dedup.is_duplicate("a")
        dedup.is_duplicate("b")
        assert sorted(dedup.keys()) == ["a", "b"]

    def test_ttl(self):
        dedup = RequestDeduplicator(ttl_sec=100, clock=lambda: 0)
        dedup.is_duplicate("a")
        assert dedup.ttl("a") == 100
        dedup._clock = lambda: 50
        assert dedup.ttl("a") == 50

    def test_ttl_expired(self):
        dedup = RequestDeduplicator(ttl_sec=10, clock=lambda: 0)
        dedup.is_duplicate("a")
        dedup._clock = lambda: 20
        assert dedup.ttl("a") is None

    def test_stats(self):
        dedup = RequestDeduplicator(ttl_sec=60, clock=lambda: 0)
        dedup.is_duplicate("a")  # miss
        dedup.is_duplicate("a")  # hit
        dedup.is_duplicate("b")  # miss
        stats = dedup.stats()
        assert stats["size"] == 2
        assert stats["ttl"] == 60
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.3333

    def test_mark_seen(self):
        dedup = RequestDeduplicator(clock=lambda: 0)
        dedup.mark_seen("req-1")
        assert dedup.is_duplicate("req-1") is True

    def test_repr(self):
        dedup = RequestDeduplicator()
        assert "RequestDeduplicator" in repr(dedup)
