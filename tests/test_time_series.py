"""Tests for time_series.py — Circular time series buffer.

Run: python3 -m pytest tests/test_time_series.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.time_series import TimeSeries


class TestTimeSeries:
    def test_create(self):
        ts = TimeSeries(capacity=10)
        assert ts.count() == 0

    def test_push_and_count(self):
        ts = TimeSeries(capacity=10)
        ts.push(42.0)
        ts.push(45.0)
        assert ts.count() == 2

    def test_avg(self):
        ts = TimeSeries(capacity=10)
        ts.push(10.0)
        ts.push(20.0)
        assert ts.avg() == 15.0

    def test_sum(self):
        ts = TimeSeries(capacity=10)
        ts.push(1.0)
        ts.push(2.0)
        assert ts.sum() == 3.0

    def test_min_max(self):
        ts = TimeSeries(capacity=10)
        ts.push(3.0)
        ts.push(1.0)
        ts.push(2.0)
        assert ts.min() == 1.0
        assert ts.max() == 3.0

    def test_capacity_eviction(self):
        ts = TimeSeries(capacity=3)
        ts.push(1.0)
        ts.push(2.0)
        ts.push(3.0)
        ts.push(4.0)
        assert ts.count() == 3
        assert ts.min() == 2.0

    def test_window_eviction(self):
        ts = TimeSeries(capacity=100, window_sec=0.05)
        now = time.time()
        ts.push(1.0, timestamp=now)
        ts.push(2.0, timestamp=now + 0.01)
        ts.push(3.0, timestamp=now + 0.06)
        assert ts.count() == 2  # first evicted

    def test_latest_earliest(self):
        ts = TimeSeries(capacity=10)
        ts.push(1.0)
        ts.push(2.0)
        assert ts.latest().value == 2.0
        assert ts.earliest().value == 1.0

    def test_range(self):
        ts = TimeSeries(capacity=10)
        ts.push(5.0)
        ts.push(10.0)
        assert ts.range() == (5.0, 10.0)

    def test_to_list(self):
        ts = TimeSeries(capacity=10)
        ts.push(1.0)
        ts.push(2.0)
        result = ts.to_list()
        assert len(result) == 2
        assert result[0][1] == 1.0

    def test_downsample(self):
        ts = TimeSeries(capacity=10)
        now = time.time()
        ts.push(10.0, timestamp=now)
        ts.push(20.0, timestamp=now + 1)
        ts.push(30.0, timestamp=now + 2)
        buckets = ts.downsample(bucket_sec=2.0)
        assert len(buckets) >= 1

    def test_clear(self):
        ts = TimeSeries(capacity=10)
        ts.push(1.0)
        ts.clear()
        assert ts.count() == 0

    def test_empty_queries(self):
        ts = TimeSeries(capacity=10)
        assert ts.avg() is None
        assert ts.min() is None
        assert ts.max() is None
        assert ts.latest() is None
        assert ts.range() is None

    def test_repr(self):
        ts = TimeSeries(capacity=10)
        ts.push(1.0)
        assert "TimeSeries" in repr(ts)
