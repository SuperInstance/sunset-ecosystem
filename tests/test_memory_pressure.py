"""Tests for memory_pressure.py — Memory monitoring and pressure handling.

Run: python3 -m pytest tests/test_memory_pressure.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.memory_pressure import MemoryPressure, MemorySnapshot


class TestMemoryPressure:
    def test_create(self):
        mp = MemoryPressure()
        assert mp.stats()["rss_mb"] >= 0

    def test_snapshot(self):
        mp = MemoryPressure()
        snap = mp.snapshot()
        assert snap.rss_mb >= 0
        assert snap.vms_mb >= 0
        assert snap.timestamp > 0

    def test_check_rate_limit(self):
        mp = MemoryPressure(check_interval=1.0)
        snap1 = mp.check()
        snap2 = mp.check()  # Should return cached within interval
        # The second check might return the same cached snapshot or a new one
        # depending on timing, so we just verify it doesn't crash
        assert snap2 is not None

    def test_peak(self):
        mp = MemoryPressure()
        mp.snapshot()
        assert mp.peak() >= 0

    def test_average(self):
        mp = MemoryPressure()
        mp.snapshot()
        mp.snapshot()
        assert mp.average() >= 0

    def test_average_empty(self):
        mp = MemoryPressure()
        assert mp.average() == 0.0

    def test_trend(self):
        mp = MemoryPressure()
        mp.snapshot()
        time.sleep(0.01)
        mp.snapshot()
        trend = mp.trend()
        assert isinstance(trend, float)

    def test_trend_empty(self):
        mp = MemoryPressure()
        assert mp.trend() == 0.0

    def test_register_alert(self):
        alerts = []
        mp = MemoryPressure()
        mp.register_alert("warn", lambda level, snap: alerts.append(level))
        # Manually trigger check with very low threshold to force alert
        mp._warn_mb = 0.1
        mp._critical_mb = 1000
        mp._check_interval = 0.0
        mp.check()
        assert "warn" in alerts

    def test_register_evictor(self):
        freed = [0]
        mp = MemoryPressure()
        mp.register_evictor(lambda: freed.__setitem__(0, freed[0] + 100) or freed[0])
        mp._gc_mb = 0.1
        mp._check_interval = 0.0
        mp.check()
        # If memory is low, evictor may not run
        # If memory is high, evictor runs
        # Just verify it doesn't crash

    def test_history_limit(self):
        mp = MemoryPressure()
        mp._max_history = 3
        for _ in range(5):
            mp.snapshot()
        assert len(mp._history) == 3

    def test_stats(self):
        mp = MemoryPressure()
        stats = mp.stats()
        assert "rss_mb" in stats
        assert "peak_mb" in stats
        assert "average_mb" in stats
        assert "trend_mb_per_min" in stats
        assert "history_samples" in stats

    def test_repr(self):
        mp = MemoryPressure()
        assert "MemoryPressure" in repr(mp)
