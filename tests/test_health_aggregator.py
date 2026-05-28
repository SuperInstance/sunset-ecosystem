"""Tests for health_aggregator.py — Aggregate health across fleet nodes.

Run: python3 -m pytest tests/test_health_aggregator.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.health_aggregator import HealthAggregator, FleetHealthSummary


class TestHealthAggregator:
    def test_create(self):
        ha = HealthAggregator()
        assert ha.report_count() == 0

    def test_report(self):
        ha = HealthAggregator()
        ha.report("node-1", {"cpu": 0.5, "memory": 0.6}, status="healthy")
        assert ha.report_count() == 1

    def test_get(self):
        ha = HealthAggregator()
        ha.report("node-1", {"cpu": 0.5}, status="healthy")
        report = ha.get("node-1")
        assert report is not None
        assert report.node_id == "node-1"
        assert report.status == "healthy"

    def test_summary_healthy(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.5}, status="healthy")
        ha.report("b", {"cpu": 0.4}, status="healthy")
        summary = ha.summary()
        assert summary.status == "healthy"
        assert summary.total_nodes == 2
        assert summary.healthy_count == 2
        assert summary.critical_count == 0

    def test_summary_degraded(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.5}, status="healthy")
        ha.report("b", {"cpu": 0.9}, status="degraded")
        summary = ha.summary()
        assert summary.status == "degraded"  # 1/2 degraded = 50% > 20%
        assert summary.degraded_count == 1

    def test_summary_critical(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.5}, status="healthy")
        ha.report("b", {"cpu": 0.9}, status="critical")
        summary = ha.summary()
        assert summary.status == "critical"
        assert summary.critical_count == 1

    def test_summary_degraded_minority(self):
        ha = HealthAggregator()
        for i in range(9):
            ha.report(f"node-{i}", {"cpu": 0.5}, status="healthy")
        ha.report("node-9", {"cpu": 0.9}, status="degraded")
        # 1/10 degraded = 10% <= 20%, so overall is healthy
        summary = ha.summary()
        assert summary.status == "healthy"
        assert summary.degraded_count == 1

    def test_summary_many_degraded(self):
        ha = HealthAggregator()
        for i in range(5):
            ha.report(f"node-{i}", {"cpu": 0.9}, status="degraded")
        ha.report("node-5", {"cpu": 0.5}, status="healthy")
        summary = ha.summary()
        # 5/6 degraded = 83% > 20% -> degraded
        assert summary.status == "degraded"

    def test_avg_metrics(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.2, "memory": 0.4}, status="healthy")
        ha.report("b", {"cpu": 0.4, "memory": 0.6}, status="healthy")
        summary = ha.summary()
        assert summary.avg_metrics["cpu"] == pytest.approx(0.3)
        assert summary.avg_metrics["memory"] == pytest.approx(0.5)

    def test_worst_nodes(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.2}, status="healthy")
        ha.report("b", {"cpu": 0.9}, status="critical")
        ha.report("c", {"cpu": 0.5}, status="degraded")
        summary = ha.summary()
        assert summary.worst_nodes[0] == "b"

    def test_nodes_by_status(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.5}, status="healthy")
        ha.report("b", {"cpu": 0.9}, status="critical")
        assert ha.nodes_by_status("healthy") == ["a"]
        assert ha.nodes_by_status("critical") == ["b"]

    def test_stale_nodes(self):
        ha = HealthAggregator(max_age=0.1)
        ha.report("a", {"cpu": 0.5}, status="healthy")
        time.sleep(0.15)
        assert ha.stale_nodes() == ["a"]

    def test_stale_excluded_from_summary(self):
        ha = HealthAggregator(max_age=0.1)
        ha.report("a", {"cpu": 0.5}, status="healthy")
        time.sleep(0.15)
        summary = ha.summary()
        assert summary.total_nodes == 0

    def test_all_nodes(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.5}, status="healthy")
        ha.report("b", {"cpu": 0.5}, status="healthy")
        assert sorted(ha.all_nodes()) == ["a", "b"]

    def test_clear(self):
        ha = HealthAggregator()
        ha.report("a", {"cpu": 0.5}, status="healthy")
        ha.clear()
        assert ha.report_count() == 0

    def test_repr(self):
        ha = HealthAggregator()
        assert "HealthAggregator" in repr(ha)
