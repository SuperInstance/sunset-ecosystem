"""Tests for metric_reporter.py — Metric reporting with aggregation and flush.

Run: python3 -m pytest tests/test_metric_reporter.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.metric_reporter import MetricReporter


class TestMetricReporter:
    def test_create(self):
        reporter = MetricReporter(flush_interval_sec=60, clock=lambda: 0)
        assert reporter.stats()["counters"] == 0

    def test_counter(self):
        reporter = MetricReporter(clock=lambda: 0)
        reporter.counter("requests", 1)
        reporter.counter("requests", 2)
        assert reporter.get_counter("requests") == 3

    def test_gauge(self):
        reporter = MetricReporter(clock=lambda: 0)
        reporter.gauge("cpu", 0.75)
        assert reporter.get_gauge("cpu") == 0.75
        reporter.gauge("cpu", 0.80)
        assert reporter.get_gauge("cpu") == 0.80

    def test_timer(self):
        reporter = MetricReporter(clock=lambda: 0)
        reporter.timer("latency", 0.023)
        reporter.timer("latency", 0.045)
        stats = reporter.get_timer_stats("latency")
        assert stats["count"] == 2
        assert stats["sum"] == 0.068
        assert stats["avg"] == 0.034
        assert stats["min"] == 0.023
        assert stats["max"] == 0.045

    def test_should_flush(self):
        reporter = MetricReporter(flush_interval_sec=10, clock=lambda: 0)
        reporter.counter("a", 1)
        assert reporter.should_flush() is False
        reporter._clock = lambda: 15
        assert reporter.should_flush() is True

    def test_flush(self):
        reporter = MetricReporter(clock=lambda: 0)
        reporter.counter("requests", 5)
        reporter.gauge("cpu", 0.75)
        reporter.timer("latency", 0.1)
        batch = reporter.flush()
        assert batch["counters"] == {"requests": 5}
        assert batch["gauges"] == {"cpu": 0.75}
        assert batch["timers"]["latency"]["count"] == 1
        assert reporter.stats()["counters"] == 0

    def test_flush_empty(self):
        reporter = MetricReporter(clock=lambda: 0)
        batch = reporter.flush()
        assert batch == {"counters": {}, "gauges": {}, "timers": {}}

    def test_metric_names(self):
        reporter = MetricReporter(clock=lambda: 0)
        reporter.counter("a", 1)
        reporter.gauge("b", 2)
        reporter.timer("c", 3)
        assert sorted(reporter.metric_names()) == ["a", "b", "c"]

    def test_timer_stats_empty(self):
        reporter = MetricReporter(clock=lambda: 0)
        assert reporter.get_timer_stats("missing") is None

    def test_stats(self):
        reporter = MetricReporter(flush_interval_sec=60, clock=lambda: 0)
        reporter.counter("a", 1)
        reporter.gauge("b", 2)
        stats = reporter.stats()
        assert stats["counters"] == 1
        assert stats["gauges"] == 1
        assert stats["timers"] == 0
        assert stats["total_flushes"] == 0

    def test_repr(self):
        reporter = MetricReporter()
        assert "MetricReporter" in repr(reporter)
