"""Tests for stream_processor.py — Windowed stream aggregation.

Run: python3 -m pytest tests/test_stream_processor.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.stream_processor import StreamProcessor


class TestStreamProcessor:
    def test_create(self):
        sp = StreamProcessor()
        assert len(sp.keys()) == 0

    def test_push_and_count(self):
        sp = StreamProcessor()
        sp.push({"cpu": 42.0})
        sp.push({"cpu": 45.0})
        assert sp.count("cpu") == 2

    def test_sum(self):
        sp = StreamProcessor()
        sp.push({"cpu": 10.0})
        sp.push({"cpu": 20.0})
        assert sp.sum("cpu") == 30.0

    def test_avg(self):
        sp = StreamProcessor()
        sp.push({"cpu": 10.0})
        sp.push({"cpu": 20.0})
        assert sp.avg("cpu") == 15.0

    def test_min_max(self):
        sp = StreamProcessor()
        sp.push({"cpu": 30.0})
        sp.push({"cpu": 10.0})
        sp.push({"cpu": 20.0})
        assert sp.min("cpu") == 10.0
        assert sp.max("cpu") == 30.0

    def test_aggregate(self):
        sp = StreamProcessor()
        sp.push({"x": 1})
        sp.push({"x": 2})
        sp.push({"x": 3})
        result = sp.aggregate("x", lambda vals: sum(v * 2 for v in vals))
        assert result == 12

    def test_window_eviction(self):
        sp = StreamProcessor(window_sec=0.05)
        sp.push({"cpu": 42.0}, timestamp=time.time())
        time.sleep(0.06)
        sp.push({"cpu": 45.0}, timestamp=time.time())
        # First event should be evicted
        assert sp.count("cpu") == 1

    def test_multiple_keys(self):
        sp = StreamProcessor()
        sp.push({"cpu": 42.0, "mem": 80.0})
        assert sp.count("cpu") == 1
        assert sp.count("mem") == 1

    def test_max_events_eviction(self):
        sp = StreamProcessor(max_events=2)
        sp.push({"x": 1})
        sp.push({"x": 2})
        sp.push({"x": 3})
        assert sp.count("x") == 2

    def test_clear(self):
        sp = StreamProcessor()
        sp.push({"x": 1})
        sp.clear()
        assert sp.count("x") == 0

    def test_stats(self):
        sp = StreamProcessor()
        sp.push({"x": 1})
        sp.push({"x": 2})
        stats = sp.stats()
        assert stats["pushed"] == 2

    def test_repr(self):
        sp = StreamProcessor()
        sp.push({"x": 1})
        assert "StreamProcessor" in repr(sp)
