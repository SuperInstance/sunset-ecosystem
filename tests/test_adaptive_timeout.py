"""Tests for adaptive_timeout.py — Adaptive timeout based on historical latency.

Run: python3 -m pytest tests/test_adaptive_timeout.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.adaptive_timeout import AdaptiveTimeout


class TestAdaptiveTimeout:
    def test_create(self):
        timer = AdaptiveTimeout(initial_sec=1.0, min_sec=0.5, max_sec=10.0)
        assert timer.current_timeout() == 1.0

    def test_record_latency(self):
        timer = AdaptiveTimeout(initial_sec=1.0, alpha=0.5)
        timer.record_latency(0.5)
        assert timer.stats()["count"] == 1
        assert timer.stats()["ema"] == 0.75  # (0.5 * 0.5) + (1.0 * 0.5)

    def test_current_timeout_basic(self):
        timer = AdaptiveTimeout(initial_sec=1.0, min_sec=0.5, max_sec=10.0)
        timer.record_latency(0.8)
        timer.record_latency(1.2)
        timeout = timer.current_timeout()
        assert 0.5 <= timeout <= 10.0

    def test_current_timeout_empty(self):
        timer = AdaptiveTimeout(initial_sec=2.0)
        assert timer.current_timeout() == 2.0

    def test_timeout_grows_with_latency(self):
        timer = AdaptiveTimeout(initial_sec=0.5, max_sec=10.0)
        for _ in range(10):
            timer.record_latency(5.0)
        assert timer.current_timeout() > 0.5

    def test_timeout_respects_max(self):
        timer = AdaptiveTimeout(initial_sec=1.0, max_sec=2.0)
        for _ in range(20):
            timer.record_latency(10.0)
        assert timer.current_timeout() <= 2.0

    def test_timeout_respects_min(self):
        timer = AdaptiveTimeout(initial_sec=5.0, min_sec=1.0)
        for _ in range(20):
            timer.record_latency(0.1)
        assert timer.current_timeout() >= 1.0

    def test_record_timeout_event(self):
        timer = AdaptiveTimeout(initial_sec=1.0, max_sec=5.0)
        timer.record_timeout()
        assert timer.current_timeout() >= 1.0

    def test_average_latency(self):
        timer = AdaptiveTimeout()
        assert timer.average_latency() is None
        timer.record_latency(1.0)
        timer.record_latency(2.0)
        assert timer.average_latency() == 1.5

    def test_percentile_latency(self):
        timer = AdaptiveTimeout()
        for i in range(10):
            timer.record_latency(float(i))
        assert timer.percentile_latency(0.0) == 0.0
        assert timer.percentile_latency(0.5) == 5.0
        assert timer.percentile_latency(1.0) == 9.0

    def test_sample_size_limit(self):
        timer = AdaptiveTimeout(sample_size=5)
        for i in range(10):
            timer.record_latency(float(i))
        assert len(timer.latency_history()) == 5

    def test_reset(self):
        timer = AdaptiveTimeout(initial_sec=1.0)
        timer.record_latency(5.0)
        timer.reset()
        assert timer.current_timeout() == 1.0
        assert timer.stats()["count"] == 0

    def test_latency_history(self):
        timer = AdaptiveTimeout()
        timer.record_latency(1.0)
        timer.record_latency(2.0)
        assert timer.latency_history() == [1.0, 2.0]

    def test_stats(self):
        timer = AdaptiveTimeout(initial_sec=1.0, min_sec=0.5, max_sec=10.0, percentile=0.95, alpha=0.3)
        timer.record_latency(0.8)
        stats = timer.stats()
        assert stats["initial"] == 1.0
        assert stats["min"] == 0.5
        assert stats["max"] == 10.0
        assert stats["percentile"] == 0.95
        assert stats["alpha"] == 0.3
        assert stats["count"] == 1
        assert "ema" in stats
        assert "current_timeout" in stats
        assert "average_latency" in stats

    def test_repr(self):
        timer = AdaptiveTimeout()
        assert "AdaptiveTimeout" in repr(timer)
