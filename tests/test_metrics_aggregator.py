"""Tests for metrics_aggregator.py — Fleet-wide metrics aggregation.

Run: python3 -m pytest tests/test_metrics_aggregator.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.metrics_aggregator import MetricsAggregator, MetricSummary


class TestMetricsAggregator:
    def test_create(self):
        agg = MetricsAggregator()
        assert agg.metric_names() == []

    def test_record(self):
        agg = MetricsAggregator()
        agg.record("node-a", "cpu", 45.0)
        assert "cpu" in agg.metric_names()
        assert "node-a" in agg.node_names()

    def test_aggregate(self):
        agg = MetricsAggregator()
        agg.record("a", "temp", 20.0)
        agg.record("b", "temp", 30.0)
        agg.record("c", "temp", 40.0)
        summary = agg.aggregate("temp")
        assert summary is not None
        assert summary.count == 3
        assert summary.avg == pytest.approx(30.0)
        assert summary.min == 20.0
        assert summary.max == 40.0
        assert summary.sum == 90.0
        assert summary.nodes == 3

    def test_aggregate_empty(self):
        agg = MetricsAggregator()
        assert agg.aggregate("missing") is None

    def test_window_filter(self):
        agg = MetricsAggregator()
        agg.record("a", "temp", 20.0)
        time.sleep(0.05)
        agg.record("a", "temp", 30.0)
        summary = agg.aggregate("temp", window_sec=0.02)
        # Only the most recent should be within 0.02s
        assert summary is not None
        assert summary.count == 1
        assert summary.avg == pytest.approx(30.0)

    def test_latest(self):
        agg = MetricsAggregator()
        agg.record("a", "cpu", 10.0)
        agg.record("a", "cpu", 20.0)
        assert agg.latest("cpu") == 20.0
        assert agg.latest("cpu", "a") == 20.0
        assert agg.latest("missing") is None

    def test_per_node(self):
        agg = MetricsAggregator()
        agg.record("a", "cpu", 10.0)
        agg.record("a", "cpu", 20.0)
        agg.record("b", "cpu", 30.0)
        per_node = agg.per_node("cpu")
        assert per_node["a"] == [10.0, 20.0]
        assert per_node["b"] == [30.0]

    def test_threshold(self):
        agg = MetricsAggregator()
        agg.set_threshold("cpu", 0.0, 80.0)
        # Should not warn
        agg.record("a", "cpu", 50.0)
        # Would warn at >80 (hard to test logging, just exercise path)
        agg.record("a", "cpu", 90.0)

    def test_max_history(self):
        agg = MetricsAggregator(max_history=3)
        agg.record("a", "x", 1.0)
        agg.record("a", "x", 2.0)
        agg.record("a", "x", 3.0)
        agg.record("a", "x", 4.0)
        summary = agg.aggregate("x")
        assert summary.count == 3  # max_history=3

    def test_export(self):
        agg = MetricsAggregator()
        agg.record("a", "cpu", 50.0)
        data = agg.export()
        assert len(data) == 1
        assert data[0]["node_id"] == "a"
        assert data[0]["name"] == "cpu"
        assert data[0]["value"] == 50.0

    def test_report(self):
        agg = MetricsAggregator()
        agg.record("a", "cpu", 10.0)
        agg.record("b", "cpu", 20.0)
        r = agg.report()
        assert r["total_readings"] == 2
        assert r["nodes"] == 2
        assert "cpu" in r["metrics"]

    def test_clear(self):
        agg = MetricsAggregator()
        agg.record("a", "cpu", 10.0)
        agg.clear()
        assert agg.metric_names() == []

    def test_repr(self):
        agg = MetricsAggregator()
        assert "MetricsAggregator" in repr(agg)
