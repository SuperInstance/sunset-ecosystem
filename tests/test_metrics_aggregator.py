"""Tests for metrics_aggregator.py — Metrics aggregation with rollup windows.

Run: python3 -m pytest tests/test_metrics_aggregator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.metrics_aggregator import MetricsAggregator


class TestMetricsAggregator:
    def test_create(self):
        agg = MetricsAggregator(window_sec=60)
        assert agg.stats()["window_sec"] == 60

    def test_record(self):
        agg = MetricsAggregator()
        agg.record("cpu", 45.0)
        assert agg.count("cpu") == 1

    def test_rollup(self):
        agg = MetricsAggregator(window_sec=60, clock=lambda: 100)
        agg.record("cpu", 40.0)
        agg.record("cpu", 60.0)
        rollup = agg.rollup("cpu")
        assert rollup is not None
        assert rollup["sum"] == 100.0
        assert rollup["avg"] == 50.0
        assert rollup["min"] == 40.0
        assert rollup["max"] == 60.0
        assert rollup["count"] == 2

    def test_rollup_empty(self):
        agg = MetricsAggregator()
        assert agg.rollup("missing") is None

    def test_rollup_outside_window(self):
        agg = MetricsAggregator(window_sec=10, clock=lambda: 0)
        agg.record("cpu", 40.0)
        agg._clock = lambda: 100
        assert agg.rollup("cpu") is None

    def test_prune(self):
        agg = MetricsAggregator(window_sec=10, clock=lambda: 100)
        agg.record("cpu", 40.0)
        agg.record("cpu", 60.0)
        agg._clock = lambda: 200
        pruned = agg.prune("cpu", max_age_sec=50)
        assert pruned == 2
        assert agg.count("cpu") == 0

    def test_prune_all(self):
        agg = MetricsAggregator(window_sec=10, clock=lambda: 100)
        agg.record("cpu", 40.0)
        agg.record("memory", 80.0)
        agg._clock = lambda: 200
        pruned = agg.prune_all(max_age_sec=50)
        assert pruned == 2

    def test_metrics(self):
        agg = MetricsAggregator()
        agg.record("cpu", 40.0)
        agg.record("memory", 80.0)
        assert sorted(agg.metrics()) == ["cpu", "memory"]

    def test_stats(self):
        agg = MetricsAggregator()
        agg.record("cpu", 40.0)
        agg.record("cpu", 60.0)
        stats = agg.stats()
        assert stats["metrics"] == 1
        assert stats["total_points"] == 2

    def test_repr(self):
        agg = MetricsAggregator()
        assert "MetricsAggregator" in repr(agg)
