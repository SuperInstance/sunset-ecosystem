"""Tests for distill.delta_tracker."""

from __future__ import annotations

import pytest

from distill.delta_tracker import DeltaSnapshot, DeltaTracker


class TestDeltaSnapshot:
    def test_creation(self):
        snap = DeltaSnapshot(generation=1, avg_quality=0.85, hint_level=5)
        assert snap.generation == 1
        assert snap.avg_quality == 0.85
        assert snap.hint_level == 5
        assert snap.timestamp > 0

    def test_repr(self):
        snap = DeltaSnapshot(generation=3, avg_quality=0.912, hint_level=2)
        r = repr(snap)
        assert "gen=3" in r
        assert "quality=0.912" in r
        assert "hints=2" in r


class TestDeltaTracker:
    def test_empty_tracker(self):
        t = DeltaTracker()
        assert t.latest is None
        assert t.snapshots == []
        assert t.delta() == 0.0
        assert not t.is_regression()
        assert not t.should_revert()
        assert t.best_generation() is None
        assert t.trend() == 0.0
        assert "empty" in repr(t)

    def test_record_and_latest(self):
        t = DeltaTracker()
        snap = t.record(generation=1, avg_quality=0.5, hint_level=10)
        assert snap.generation == 1
        assert t.latest is snap
        assert len(t.snapshots) == 1

    def test_delta_positive_improvement(self):
        t = DeltaTracker()
        t.record(generation=1, avg_quality=0.5, hint_level=10)
        t.record(generation=2, avg_quality=0.7, hint_level=8)
        t.record(generation=3, avg_quality=0.9, hint_level=6)
        assert t.delta() > 0.0

    def test_delta_negative_regression(self):
        t = DeltaTracker()
        t.record(generation=1, avg_quality=0.9, hint_level=6)
        t.record(generation=2, avg_quality=0.5, hint_level=8)
        assert t.delta() < 0.0

    def test_delta_not_enough_data(self):
        t = DeltaTracker()
        t.record(generation=1, avg_quality=0.5, hint_level=10)
        assert t.delta() == 0.0

    def test_is_regression(self):
        t = DeltaTracker()
        # 4 consecutive drops needed (negative_streak=3 means 3+1 snapshots)
        t.record(generation=1, avg_quality=0.9, hint_level=6)
        t.record(generation=2, avg_quality=0.8, hint_level=6)
        t.record(generation=3, avg_quality=0.7, hint_level=6)
        t.record(generation=4, avg_quality=0.6, hint_level=6)
        assert t.is_regression(negative_streak=3)

    def test_no_regression_with_recovery(self):
        t = DeltaTracker()
        t.record(generation=1, avg_quality=0.9, hint_level=6)
        t.record(generation=2, avg_quality=0.8, hint_level=6)
        t.record(generation=3, avg_quality=0.85, hint_level=6)  # recovery
        t.record(generation=4, avg_quality=0.7, hint_level=6)
        assert not t.is_regression(negative_streak=3)

    def test_should_revert(self):
        t = DeltaTracker()
        for i in range(5):
            t.record(generation=i, avg_quality=0.9 - i * 0.1, hint_level=6)
        assert t.should_revert()

    def test_best_generation(self):
        t = DeltaTracker()
        t.record(generation=1, avg_quality=0.5, hint_level=10)
        t.record(generation=2, avg_quality=0.95, hint_level=8)
        t.record(generation=3, avg_quality=0.7, hint_level=6)
        best = t.best_generation()
        assert best is not None
        assert best.generation == 2
        assert best.avg_quality == 0.95

    def test_trend_improving(self):
        t = DeltaTracker()
        for i in range(6):
            t.record(generation=i, avg_quality=0.5 + i * 0.05, hint_level=10 - i)
        assert t.trend() > 0.0

    def test_trend_declining(self):
        t = DeltaTracker()
        for i in range(6):
            t.record(generation=i, avg_quality=0.9 - i * 0.05, hint_level=10 - i)
        assert t.trend() < 0.0

    def test_repr_with_data(self):
        t = DeltaTracker()
        t.record(generation=1, avg_quality=0.75, hint_level=5)
        r = repr(t)
        assert "snapshots=1" in r
        assert "latest_gen=1" in r
