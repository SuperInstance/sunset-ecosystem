"""Tests for health_aggregator.py — Health status aggregation.

Run: python3 -m pytest tests/test_health_aggregator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.health_aggregator import HealthAggregator


class TestHealthAggregator:
    def test_create(self):
        ha = HealthAggregator()
        assert ha.status() == "unknown"

    def test_report(self):
        ha = HealthAggregator()
        ha.report("svc-a", "healthy")
        assert ha.get("svc-a") == "healthy"
        assert ha.sources() == ["svc-a"]

    def test_remove(self):
        ha = HealthAggregator()
        ha.report("svc-a", "healthy")
        assert ha.remove("svc-a") is True
        assert ha.status() == "unknown"
        assert ha.remove("missing") is False

    def test_clear(self):
        ha = HealthAggregator()
        ha.report("svc-a", "healthy")
        ha.report("svc-b", "healthy")
        ha.clear()
        assert ha.status() == "unknown"

    def test_unhealthiest_strategy(self):
        ha = HealthAggregator(strategy="unhealthiest")
        ha.report("svc-a", "healthy")
        ha.report("svc-b", "degraded")
        ha.report("svc-c", "healthy")
        assert ha.status() == "degraded"

    def test_healthiest_strategy(self):
        ha = HealthAggregator(strategy="healthiest")
        ha.report("svc-a", "healthy")
        ha.report("svc-b", "degraded")
        ha.report("svc-c", "critical")
        assert ha.status() == "healthy"

    def test_average_strategy(self):
        ha = HealthAggregator(strategy="average")
        # critical=0, unhealthy=1, degraded=2, healthy=3, excellent=4
        ha.report("a", "degraded")  # 2
        ha.report("b", "healthy")   # 3
        assert ha.status() == "degraded"  # avg = 2.5 -> int -> 2

    def test_threshold_strategy(self):
        ha = HealthAggregator(strategy="threshold", threshold=0.5)
        ha.report("a", "healthy")
        ha.report("b", "healthy")
        ha.report("c", "unhealthy")
        assert ha.status() == "healthy"

    def test_threshold_strategy_fail(self):
        ha = HealthAggregator(strategy="threshold", threshold=0.6)
        ha.report("a", "healthy")
        ha.report("b", "unhealthy")
        ha.report("c", "unhealthy")
        assert ha.status() == "unhealthy"

    def test_quorum_strategy(self):
        ha = HealthAggregator(strategy="quorum", threshold=0.5)
        ha.report("a", "healthy")
        ha.report("b", "healthy")
        ha.report("c", "degraded")
        assert ha.status() == "healthy"

    def test_quorum_strategy_fail(self):
        ha = HealthAggregator(strategy="quorum", threshold=0.7)
        ha.report("a", "healthy")
        ha.report("b", "degraded")
        ha.report("c", "unhealthy")
        assert ha.status() == "unhealthy"

    def test_custom_strategy(self):
        ha = HealthAggregator(custom_strategy=lambda values: "custom")
        ha.report("a", "healthy")
        assert ha.status() == "custom"

    def test_unknown_health_level(self):
        ha = HealthAggregator(strategy="healthiest")
        ha.report("a", "unknown-level")
        assert ha.status() == "critical"  # Unknown maps to 0

    def test_counts(self):
        ha = HealthAggregator()
        ha.report("a", "healthy")
        ha.report("b", "degraded")
        ha.report("c", "healthy")
        assert ha.counts() == {"healthy": 2, "degraded": 1}

    def test_stats(self):
        ha = HealthAggregator(strategy="quorum", threshold=0.5)
        ha.report("a", "healthy")
        ha.report("b", "healthy")
        stats = ha.stats()
        assert stats["sources"] == 2
        assert stats["strategy"] == "quorum"
        assert stats["aggregate_status"] == "healthy"

    def test_repr(self):
        ha = HealthAggregator()
        assert "HealthAggregator" in repr(ha)
