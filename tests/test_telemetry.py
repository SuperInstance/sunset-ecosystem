"""Tests for telemetry.py — Metrics collection and Prometheus/StatsD format.

Run: python3 -m pytest tests/test_telemetry.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.telemetry import TelemetryRegistry, Counter, Gauge, Histogram


class TestTelemetryRegistry:
    def test_create(self):
        reg = TelemetryRegistry()
        assert "fleet" in repr(reg)

    def test_counter(self):
        reg = TelemetryRegistry()
        c = reg.counter("breeds_total")
        c.inc()
        c.inc(2)
        assert c.value == 3

    def test_gauge(self):
        reg = TelemetryRegistry()
        g = reg.gauge("cpu_percent")
        g.set(42.0)
        assert g.value == 42.0
        g.inc(8)
        assert g.value == 50.0
        g.dec(10)
        assert g.value == 40.0

    def test_histogram(self):
        reg = TelemetryRegistry()
        h = reg.histogram("latency_ms")
        h.observe(50)
        h.observe(150)
        h.observe(250)
        assert h._total == 3
        assert h._sum == 450

    def test_histogram_percentile(self):
        reg = TelemetryRegistry()
        h = reg.histogram("latency_ms")
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            h.observe(v)
        assert h.percentile(0.5) >= 50

    def test_prometheus_format(self):
        reg = TelemetryRegistry()
        reg.counter("breeds_total").inc(5)
        reg.gauge("cpu_percent").set(42.0)
        text = reg.prometheus_format()
        assert "fleet_breeds_total" in text
        assert "fleet_cpu_percent" in text
        assert "42.0" in text

    def test_statsd_format(self):
        reg = TelemetryRegistry()
        reg.counter("breeds_total").inc(5)
        lines = reg.statsd_format()
        assert any("fleet.breeds_total:5" in line and "|c" in line for line in lines)

    def test_all_metrics(self):
        reg = TelemetryRegistry()
        reg.counter("c").inc()
        reg.gauge("g").set(1.0)
        reg.histogram("h").observe(100)
        m = reg.all_metrics()
        assert len(m["counters"]) == 1
        assert len(m["gauges"]) == 1
        assert len(m["histograms"]) == 1

    def test_labels(self):
        reg = TelemetryRegistry()
        c = reg.counter("requests", status="200")
        c.inc()
        key = reg._key("requests", {"status": "200"})
        assert key == 'requests{status="200"}'

    def test_histogram_custom_buckets(self):
        reg = TelemetryRegistry()
        h = reg.histogram("custom", buckets=[1, 2, 3])
        h.observe(1.5)
        assert h._counts[2] == 1

    def test_repr(self):
        reg = TelemetryRegistry()
        reg.counter("a")
        reg.gauge("b")
        assert "metrics=2" in repr(reg)
