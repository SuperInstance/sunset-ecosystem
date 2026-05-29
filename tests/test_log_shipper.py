"""Tests for log_shipper.py — Log shipping with batching and buffering.

Run: python3 -m pytest tests/test_log_shipper.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.log_shipper import LogShipper


class TestLogShipper:
    def test_create(self):
        shipper = LogShipper(batch_size=100, flush_interval_sec=5, clock=lambda: 0)
        assert shipper.stats()["buffer_size"] == 0

    def test_append(self):
        shipper = LogShipper(clock=lambda: 0)
        shipper.append({"level": "info", "message": "hello"})
        assert shipper.buffer_size() == 1

    def test_extend(self):
        shipper = LogShipper(clock=lambda: 0)
        shipper.extend([{"level": "info"}, {"level": "error"}])
        assert shipper.buffer_size() == 2

    def test_should_flush_by_size(self):
        shipper = LogShipper(batch_size=3, clock=lambda: 0)
        shipper.append({"level": "info"})
        shipper.append({"level": "info"})
        assert shipper.should_flush() is False
        shipper.append({"level": "info"})
        assert shipper.should_flush() is True

    def test_should_flush_by_interval(self):
        shipper = LogShipper(batch_size=100, flush_interval_sec=5, clock=lambda: 0)
        shipper.append({"level": "info"})
        assert shipper.should_flush() is False
        shipper._clock = lambda: 10
        assert shipper.should_flush() is True

    def test_flush(self):
        shipper = LogShipper(clock=lambda: 0)
        shipper.append({"level": "info"})
        shipper.append({"level": "error"})
        batch = shipper.flush()
        assert len(batch) == 2
        assert shipper.is_empty() is True

    def test_flush_empty(self):
        shipper = LogShipper(clock=lambda: 0)
        batch = shipper.flush()
        assert batch == []

    def test_flush_partial(self):
        shipper = LogShipper(clock=lambda: 0)
        for i in range(10):
            shipper.append({"id": i})
        batch = shipper.flush_partial(3)
        assert len(batch) == 3
        assert shipper.buffer_size() == 7

    def test_peek(self):
        shipper = LogShipper(clock=lambda: 0)
        shipper.append({"id": 1})
        shipper.append({"id": 2})
        assert shipper.peek(1) == [{"id": 1}]
        assert shipper.buffer_size() == 2

    def test_stats(self):
        shipper = LogShipper(batch_size=100, flush_interval_sec=5, clock=lambda: 0)
        for i in range(5):
            shipper.append({"id": i})
        shipper.flush()
        stats = shipper.stats()
        assert stats["buffer_size"] == 0
        assert stats["total_shipped"] == 5
        assert stats["total_batches"] == 1
        assert stats["batch_size"] == 100

    def test_repr(self):
        shipper = LogShipper()
        assert "LogShipper" in repr(shipper)
