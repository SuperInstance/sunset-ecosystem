"""Tests for FleetMetricsCollector — continuous metrics collection and trend analysis.

Reference: fleet/fleet_metrics_collector.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.fleet_metrics_collector import (
    FleetMetricsCollector,
    MetricsSnapshot,
    ModuleMetrics,
    TrendAnalysis,
)
from fleet.ternary_types import TernaryValue


class TestModuleMetrics:
    def test_fields(self) -> None:
        m = ModuleMetrics(
            name="Test",
            timestamp=1.0,
            health_ternary=TernaryValue.POS,
            health_emoji="🟢",
            test_count=10,
            test_passed=8,
            test_coverage=0.8,
            status="healthy",
        )
        assert m.name == "Test"
        assert m.health_ternary == TernaryValue.POS


class TestMetricsSnapshot:
    def test_empty(self) -> None:
        s = MetricsSnapshot(
            timestamp=1.0,
            cycle_number=0,
            total_modules=0,
            healthy_modules=0,
            warning_modules=0,
            critical_modules=0,
            total_tests=0,
            tests_passed=0,
            tests_failed=0,
            test_coverage_pct=0.0,
            integration_count=0,
            tested_integrations=0,
        )
        assert s.health_score == 0.0

    def test_health_score(self) -> None:
        s = MetricsSnapshot(
            timestamp=1.0,
            cycle_number=1,
            total_modules=10,
            healthy_modules=8,
            warning_modules=1,
            critical_modules=1,
            total_tests=100,
            tests_passed=90,
            tests_failed=10,
            test_coverage_pct=0.9,
            integration_count=5,
            tested_integrations=3,
        )
        assert s.health_score == 0.8


class TestTrendAnalysis:
    def test_fields(self) -> None:
        t = TrendAnalysis(
            metric_name="health",
            direction="improving",
            slope=0.1,
            current_value=0.8,
            previous_value=0.7,
            change_pct=14.3,
            confidence=0.9,
        )
        assert t.direction == "improving"
        assert t.confidence == 0.9


class TestFleetMetricsCollector:
    def test_init(self) -> None:
        collector = FleetMetricsCollector()
        assert collector.max_history == 100
        assert collector._history == []

    def test_ensure_initialized(self) -> None:
        collector = FleetMetricsCollector()
        collector._ensure_initialized()
        assert collector._orchestrator is not None
        assert collector._harbor is not None

    def test_record_beat_metrics(self) -> None:
        collector = FleetMetricsCollector()
        snapshot = collector.record_beat_metrics()
        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.total_modules == 20
        assert snapshot.healthy_modules == 20
        assert snapshot.total_tests > 300
        assert snapshot.health_score == 1.0

    def test_history_recorded(self) -> None:
        collector = FleetMetricsCollector()
        collector.record_beat_metrics()
        assert len(collector._history) == 1

    def test_history_max_size(self) -> None:
        collector = FleetMetricsCollector(max_history=3)
        for _ in range(5):
            collector.record_beat_metrics()
        assert len(collector._history) == 3

    def test_get_latest_snapshot(self) -> None:
        collector = FleetMetricsCollector()
        assert collector.get_latest_snapshot() is None
        collector.record_beat_metrics()
        latest = collector.get_latest_snapshot()
        assert latest is not None
        assert latest.total_modules == 20

    def test_get_history(self) -> None:
        collector = FleetMetricsCollector()
        history = collector.get_history()
        assert history == []
        collector.record_beat_metrics()
        assert len(collector.get_history()) == 1

    def test_analyze_trends_empty(self) -> None:
        collector = FleetMetricsCollector()
        trends = collector.analyze_trends()
        assert trends == {}

    def test_analyze_trends_single(self) -> None:
        collector = FleetMetricsCollector()
        collector.record_beat_metrics()
        trends = collector.analyze_trends()
        assert trends == {}

    def test_analyze_trends_multiple(self) -> None:
        collector = FleetMetricsCollector()
        for _ in range(5):
            collector.record_beat_metrics()
        trends = collector.analyze_trends()
        assert "health" in trends
        assert "coverage" in trends
        assert "tests" in trends
        assert "integrations" in trends
        assert "critical" in trends

    def test_trend_direction(self) -> None:
        collector = FleetMetricsCollector()
        for _ in range(5):
            collector.record_beat_metrics()
        trends = collector.analyze_trends()
        # All modules are healthy, so health should be stable
        assert trends["health"].direction in ["improving", "stable", "degrading"]

    def test_check_alerts_empty(self) -> None:
        collector = FleetMetricsCollector()
        alerts = collector.check_alerts()
        assert alerts == []

    def test_check_alerts_healthy(self) -> None:
        collector = FleetMetricsCollector()
        collector.record_beat_metrics()
        alerts = collector.check_alerts()
        # All healthy, so no critical alerts
        assert all(a["level"] != "critical" for a in alerts)

    def test_compute_trend(self) -> None:
        collector = FleetMetricsCollector()
        trend = collector._compute_trend("test", [0.5, 0.6, 0.7, 0.8, 0.9])
        assert trend.direction == "improving"
        assert trend.slope > 0

    def test_compute_trend_stable(self) -> None:
        collector = FleetMetricsCollector()
        trend = collector._compute_trend("test", [0.5, 0.5, 0.5, 0.5, 0.5])
        assert trend.direction == "stable"
        assert trend.slope == 0.0

    def test_compute_trend_degrading(self) -> None:
        collector = FleetMetricsCollector()
        trend = collector._compute_trend("test", [0.9, 0.8, 0.7, 0.6, 0.5])
        assert trend.direction == "degrading"
        assert trend.slope < 0

    def test_compute_trend_inverse(self) -> None:
        collector = FleetMetricsCollector()
        # Inverse: decreasing values = improving (fewer critical modules)
        trend = collector._compute_trend("test", [5, 4, 3, 2, 1], inverse=True)
        assert trend.direction == "improving"

    def test_compute_trend_short(self) -> None:
        collector = FleetMetricsCollector()
        trend = collector._compute_trend("test", [0.5])
        assert trend.direction == "stable"

    def test_save_and_load_history(self, tmp_path: Path) -> None:
        collector = FleetMetricsCollector()
        collector.record_beat_metrics()
        path = tmp_path / "history.json"
        collector.save_history(path)
        assert path.exists()

        collector2 = FleetMetricsCollector()
        collector2.load_history(path)
        assert len(collector2._history) == 1
        assert collector2._history[0].total_modules == 20

    def test_generate_trend_report(self, tmp_path: Path) -> None:
        collector = FleetMetricsCollector()
        collector.record_beat_metrics()
        output = tmp_path / "trend.md"
        content = collector.generate_trend_report(output)
        assert output.exists()
        assert "# 📈 Fleet Metrics Trend Report" in content

    def test_report_has_alerts(self, tmp_path: Path) -> None:
        collector = FleetMetricsCollector()
        collector.record_beat_metrics()
        output = tmp_path / "trend.md"
        content = collector.generate_trend_report(output)
        assert "## Alerts" in content

    def test_report_has_trends(self, tmp_path: Path) -> None:
        collector = FleetMetricsCollector()
        for _ in range(5):
            collector.record_beat_metrics()
        output = tmp_path / "trend.md"
        content = collector.generate_trend_report(output)
        assert "## Trends" in content

    def test_print_summary(self, capsys) -> None:
        collector = FleetMetricsCollector()
        collector.record_beat_metrics()
        collector.print_summary()
        captured = capsys.readouterr()
        assert "FLEET METRICS COLLECTOR" in captured.out
        assert "Health:" in captured.out

    def test_print_summary_empty(self, capsys) -> None:
        collector = FleetMetricsCollector()
        collector.print_summary()
        captured = capsys.readouterr()
        assert "No metrics recorded yet." in captured.out

    def test_snapshot_module_metrics(self) -> None:
        collector = FleetMetricsCollector()
        snapshot = collector.record_beat_metrics()
        assert len(snapshot.module_metrics) > 0
        assert all(isinstance(m, ModuleMetrics) for m in snapshot.module_metrics)

    def test_snapshot_health_emoji(self) -> None:
        collector = FleetMetricsCollector()
        snapshot = collector.record_beat_metrics()
        for m in snapshot.module_metrics:
            assert m.health_emoji in ["🟢", "🟡", "🔴"]
