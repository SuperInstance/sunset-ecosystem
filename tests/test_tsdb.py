"""Tests for tsdb.py — Time-series database for fleet metrics.

Run: python3 -m pytest tests/test_tsdb.py -v --tb=short
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from fleet.tsdb import Aggregation, TimeSeriesDB


class TestMetricSeries:
    def test_append_and_last(self):
        db = TimeSeriesDB()
        db.record("cpu", 0.5, labels={"node": "a"})
        db.record("cpu", 0.6, labels={"node": "a"})
        series = db.create_series("cpu", labels={"node": "a"})
        last = series.last()
        assert last is not None
        assert last.value == pytest.approx(0.6)

    def test_range_query(self):
        db = TimeSeriesDB()
        now = time.time_ns()
        for i in range(10):
            db.record("temp", float(i), timestamp_ns=now + i * 1_000_000_000)
        results = db.query("temp", start_ns=now, end_ns=now + 5_000_000_000)
        # Should get points from t=0 to t=5
        assert len(results) >= 5

    def test_downsample_mean(self):
        db = TimeSeriesDB()
        now = time.time_ns()
        # 10 points, 1 per second, bucket=2 seconds
        for i in range(10):
            db.record("val", float(i), timestamp_ns=now + i * 1_000_000_000)
        results = db.query(
            "val",
            start_ns=now,
            end_ns=now + 10_000_000_000,
            aggregation=Aggregation.MEAN,
            bucket_ns=2_000_000_000,
        )
        assert len(results) > 0
        # Each bucket should average ~2 values
        for _, v in results:
            assert 0 <= v <= 9

    def test_downsample_count(self):
        db = TimeSeriesDB()
        now = (time.time_ns() // 2_000_000_000) * 2_000_000_000  # align to 2s boundary
        for i in range(10):
            db.record("val", 1.0, timestamp_ns=now + i * 1_000_000_000)
        results = db.query(
            "val",
            aggregation=Aggregation.COUNT,
            bucket_ns=2_000_000_000,
        )
        # With 10 points at 1s intervals and 2s buckets aligned,
        # every bucket should have exactly 2 points
        for _, v in results:
            assert v == pytest.approx(2.0)

    def test_stats(self):
        db = TimeSeriesDB()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            db.record("metric", v)
        series = db.create_series("metric")
        s = series.stats()
        assert s["count"] == 5
        assert s["min"] == pytest.approx(1.0)
        assert s["max"] == pytest.approx(5.0)
        assert s["mean"] == pytest.approx(3.0)
        assert s["last"] == pytest.approx(5.0)

    def test_retention_wraparound(self):
        db = TimeSeriesDB()
        series = db.create_series("wrap", retention=5)
        for i in range(10):
            series.append(float(i))
        assert series._count == 5  # capped at retention
        ts, vs = series._get_sorted()
        assert len(vs) == 5
        # Should contain the last 5 values
        np.testing.assert_allclose(vs, np.array([5.0, 6.0, 7.0, 8.0, 9.0]))

    def test_empty_stats(self):
        db = TimeSeriesDB()
        series = db.create_series("empty")
        s = series.stats()
        assert s["count"] == 0


class TestTimeSeriesDB:
    def test_create_and_record(self):
        db = TimeSeriesDB()
        db.record("cpu", 0.5, labels={"node": "a"})
        db.record("cpu", 0.6, labels={"node": "b"})
        assert db.series_names() == {"cpu"}
        r = db.report()
        assert r["series_count"] == 2
        assert r["total_points"] == 2

    def test_query_missing_series(self):
        db = TimeSeriesDB()
        results = db.query("missing")
        assert results == []

    def test_aggregate(self):
        db = TimeSeriesDB()
        db.record("cpu", 0.5, labels={"node": "a"})
        db.record("cpu", 0.7, labels={"node": "b"})
        val = db.aggregate("cpu", aggregation=Aggregation.MEAN)
        assert val == pytest.approx(0.6)

    def test_aggregate_missing(self):
        db = TimeSeriesDB()
        val = db.aggregate("missing")
        assert val is None

    def test_prometheus_exposition(self):
        db = TimeSeriesDB()
        db.record("cpu", 0.5, labels={"node": "a"})
        exp = db.prometheus_exposition()
        assert "cpu" in exp
        assert "node=\"a\"" in exp
        assert "0.5" in exp

    def test_series_key_sorting(self):
        db = TimeSeriesDB()
        db.record("m", 1.0, labels={"b": "2", "a": "1"})
        series = db.create_series("m", labels={"a": "1", "b": "2"})
        assert series is not None
        assert series.labels == {"b": "2", "a": "1"}

    def test_label_filtering(self):
        db = TimeSeriesDB()
        db.record("cpu", 0.5, labels={"node": "a", "core": "0"})
        db.record("cpu", 0.6, labels={"node": "a", "core": "1"})
        db.record("cpu", 0.7, labels={"node": "b"})
        results_a = db.query("cpu", labels={"node": "a", "core": "0"})
        assert len(results_a) == 1

    def test_multiple_aggregations(self):
        db = TimeSeriesDB()
        for v in [10.0, 20.0, 30.0]:
            db.record("metric", v)
        assert db.aggregate("metric", Aggregation.MAX) == pytest.approx(30.0)
        assert db.aggregate("metric", Aggregation.MIN) == pytest.approx(10.0)
        assert db.aggregate("metric", Aggregation.COUNT) == pytest.approx(3.0)

    def test_time_range_filtering(self):
        db = TimeSeriesDB()
        now = time.time_ns()
        db.record("t", 1.0, timestamp_ns=now - 10_000_000_000)
        db.record("t", 2.0, timestamp_ns=now - 5_000_000_000)
        db.record("t", 3.0, timestamp_ns=now)
        results = db.query("t", start_ns=now - 6_000_000_000)
        values = [v for _, v in results]
        assert 2.0 in values
        assert 3.0 in values
        assert 1.0 not in values

    def test_downsample_std(self):
        db = TimeSeriesDB()
        now = (time.time_ns() // 10_000_000_000) * 10_000_000_000  # align to 10s boundary
        for i in range(10):
            db.record("val", float(i), timestamp_ns=now + i * 1_000_000_000)
        results = db.query(
            "val",
            aggregation=Aggregation.STD,
            bucket_ns=10_000_000_000,
        )
        # Single bucket with 10 values 0-9
        assert len(results) == 1
        # std of 0..9 ≈ 2.87
        assert results[0][1] == pytest.approx(2.87, abs=0.1)

    def test_all_series(self):
        db = TimeSeriesDB()
        db.record("a", 1.0)
        db.record("b", 2.0)
        all_s = db.all_series()
        assert len(all_s) == 2
