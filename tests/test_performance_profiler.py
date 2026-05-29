"""Tests for performance_profiler.py — Simple profiling and timing.

Run: python3 -m pytest tests/test_performance_profiler.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.performance_profiler import PerformanceProfiler


class TestPerformanceProfiler:
    def test_create(self):
        prof = PerformanceProfiler()
        assert prof.summary()["measurements"] == 0

    def test_time_context(self):
        prof = PerformanceProfiler()
        with prof.time("test"):
            pass
        assert prof.count("test") == 1

    def test_timing_stats(self):
        prof = PerformanceProfiler()
        with prof.time("test"):
            time.sleep(0.01)
        stats = prof.stats("test")
        assert stats["count"] == 1
        assert stats["total"] > 0
        assert stats["min"] > 0
        assert stats["max"] > 0

    def test_avg(self):
        prof = PerformanceProfiler()
        with prof.time("test"):
            pass
        with prof.time("test"):
            pass
        avg = prof.avg("test")
        assert avg > 0

    def test_reset(self):
        prof = PerformanceProfiler()
        with prof.time("test"):
            pass
        prof.reset("test")
        assert prof.count("test") == 0

    def test_reset_all(self):
        prof = PerformanceProfiler()
        with prof.time("a"):
            pass
        with prof.time("b"):
            pass
        prof.reset()
        assert prof.summary()["measurements"] == 0

    def test_summary(self):
        prof = PerformanceProfiler()
        with prof.time("a"):
            pass
        with prof.time("b"):
            pass
        summary = prof.summary()
        assert summary["measurements"] == 2
        assert summary["total_calls"] == 2

    def test_repr(self):
        prof = PerformanceProfiler()
        assert "PerformanceProfiler" in repr(prof)
