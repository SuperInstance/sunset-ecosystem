"""Tests for FleetMetricsExporter — metrics export to Prometheus and InfluxDB.

Reference: fleet/fleet_metrics_exporter.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fleet.fleet_metrics_exporter import (
    FleetMetricsExporter,
    InfluxDBPoint,
    PrometheusMetric,
)


class TestPrometheusMetric:
    def test_to_prometheus_gauge(self) -> None:
        metric = PrometheusMetric(
            name="fleet_modules_total",
            value=20,
            metric_type="gauge",
            help_text="Total modules",
        )
        text = metric.to_prometheus()
        assert "# HELP fleet_modules_total Total modules" in text
        assert "# TYPE fleet_modules_total gauge" in text
        assert "fleet_modules_total 20" in text

    def test_to_prometheus_with_labels(self) -> None:
        metric = PrometheusMetric(
            name="fleet_health",
            value=0.95,
            metric_type="gauge",
            labels={"workspace": "test"},
        )
        text = metric.to_prometheus()
        assert 'fleet_health{workspace="test"} 0.95' in text

    def test_to_prometheus_counter(self) -> None:
        metric = PrometheusMetric(
            name="fleet_cycles",
            value=100,
            metric_type="counter",
            help_text="Cycle count",
        )
        text = metric.to_prometheus()
        assert "# TYPE fleet_cycles counter" in text


class TestInfluxDBPoint:
    def test_to_line_basic(self) -> None:
        point = InfluxDBPoint(
            measurement="fleet_modules",
            fields={"total": 20, "healthy": 18},
        )
        line = point.to_line()
        assert "fleet_modules total=20,healthy=18" in line

    def test_to_line_with_tags(self) -> None:
        point = InfluxDBPoint(
            measurement="fleet_tests",
            fields={"passed": 668},
            tags={"workspace": "sunset"},
        )
        line = point.to_line()
        assert "fleet_tests,workspace=sunset passed=668" in line

    def test_to_line_with_timestamp(self) -> None:
        point = InfluxDBPoint(
            measurement="fleet_health",
            fields={"score": 0.95},
            timestamp_ns=1234567890000000000,
        )
        line = point.to_line()
        assert "fleet_health score=0.95 1234567890000000000" in line


class TestFleetMetricsExporter:
    def test_init(self) -> None:
        exporter = FleetMetricsExporter()
        assert exporter.workspace.exists()

    def test_collect_fleet_metrics(self) -> None:
        exporter = FleetMetricsExporter()
        metrics = exporter.collect_fleet_metrics()
        assert isinstance(metrics, dict)
        assert "modules_total" in metrics
        assert "modules_healthy" in metrics
        assert "modules_degraded" in metrics
        assert "modules_critical" in metrics
        assert "test_coverage" in metrics
        assert "tests_passed" in metrics
        assert "tests_failed" in metrics
        assert "health_score" in metrics
        assert "cycle_number" in metrics
        assert "timestamp" in metrics

    def test_to_prometheus(self) -> None:
        exporter = FleetMetricsExporter()
        text = exporter.to_prometheus()
        assert "# HELP fleet_modules_total" in text
        assert "# TYPE fleet_modules_total gauge" in text
        assert "fleet_modules_total" in text
        assert "fleet_modules_healthy" in text
        assert "fleet_modules_degraded" in text
        assert "fleet_modules_critical" in text
        assert "fleet_test_coverage" in text
        assert "fleet_tests_passed" in text
        assert "fleet_tests_failed" in text
        assert "fleet_health_score" in text
        assert "fleet_cycle_number" in text

    def test_to_prometheus_values(self) -> None:
        exporter = FleetMetricsExporter()
        text = exporter.to_prometheus()
        # Should contain actual numeric values
        lines = [
            l
            for l in text.split("\n")
            if l.startswith("fleet_") and not l.startswith("#")
        ]
        assert len(lines) > 0
        for line in lines:
            parts = line.split()
            assert len(parts) >= 2
            assert float(parts[-1]) >= 0

    def test_to_influxdb(self) -> None:
        exporter = FleetMetricsExporter()
        text = exporter.to_influxdb()
        assert "fleet_modules" in text
        assert "fleet_tests" in text
        assert "fleet_health" in text
        assert "total=" in text
        assert "healthy=" in text
        assert "passed=" in text
        assert "score=" in text

    def test_to_influxdb_structure(self) -> None:
        exporter = FleetMetricsExporter()
        text = exporter.to_influxdb()
        lines = [l for l in text.strip().split("\n") if l]
        assert len(lines) == 3
        for line in lines:
            assert " " in line

    def test_to_json(self) -> None:
        exporter = FleetMetricsExporter()
        text = exporter.to_json()
        data = json.loads(text)
        assert isinstance(data, dict)
        assert "modules_total" in data

    def test_write_prometheus_file(self, tmp_path) -> None:
        exporter = FleetMetricsExporter()
        path = str(tmp_path / "metrics.prom")
        content = exporter.write_prometheus_file(path)
        assert Path(path).exists()
        assert Path(path).read_text() == content
        assert "fleet_modules_total" in content

    def test_write_influxdb_file(self, tmp_path) -> None:
        exporter = FleetMetricsExporter()
        path = str(tmp_path / "metrics.txt")
        content = exporter.write_influxdb_file(path)
        assert Path(path).exists()
        assert Path(path).read_text() == content
        assert "fleet_modules" in content

    def test_benchmark_to_prometheus(self) -> None:
        exporter = FleetMetricsExporter()
        text = exporter.benchmark_to_prometheus()
        assert "fleet_benchmark_" in text
        assert "_mean_ms" in text
        assert "_ops_per_sec" in text
        assert "_memory_peak_mb" in text
        assert "# TYPE" in text

    def test_benchmark_to_prometheus_values(self) -> None:
        exporter = FleetMetricsExporter()
        text = exporter.benchmark_to_prometheus()
        lines = [
            l
            for l in text.split("\n")
            if l.startswith("fleet_benchmark_") and not l.startswith("#")
        ]
        assert len(lines) > 0
        for line in lines:
            parts = line.split()
            assert len(parts) >= 2
            # Value should be a number
            assert float(parts[-1]) >= 0 or float(parts[-1]) == 0

    def test_metrics_consistency(self) -> None:
        exporter = FleetMetricsExporter()
        prom = exporter.to_prometheus()
        influx = exporter.to_influxdb()
        json_data = exporter.to_json()
        data = json.loads(json_data)

        # Prometheus should have all the metric names
        assert str(data["modules_total"]) in prom
        assert str(data["modules_healthy"]) in prom
        assert str(data["health_score"]) in prom

    def test_timestamp_present(self) -> None:
        exporter = FleetMetricsExporter()
        metrics = exporter.collect_fleet_metrics()
        assert metrics["timestamp"] > 0
        assert metrics["timestamp"] <= time.time() + 1

    def test_lazy_init_orchestrator(self) -> None:
        exporter = FleetMetricsExporter()
        assert exporter._orchestrator is None
        exporter._ensure_orchestrator()
        assert exporter._orchestrator is not None

    def test_lazy_init_metrics(self) -> None:
        exporter = FleetMetricsExporter()
        assert exporter._metrics is None
        exporter._ensure_metrics()
        assert exporter._metrics is not None

    def test_lazy_init_benchmark(self) -> None:
        exporter = FleetMetricsExporter()
        assert exporter._benchmark is None
        exporter._ensure_benchmark()
        assert exporter._benchmark is not None
