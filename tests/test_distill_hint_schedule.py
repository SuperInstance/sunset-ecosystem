"""Tests for distill.hint_schedule."""

from __future__ import annotations

import pytest

from distill.hint_schedule import ExponentialBackoffSchedule


class TestExponentialBackoffSchedule:
    def test_initial_level(self):
        s = ExponentialBackoffSchedule(max_hints=10)
        assert s.current_level() == 10
        assert not s.is_autonomous()

    def test_reduce(self):
        s = ExponentialBackoffSchedule(max_hints=10, reduction_rate=0.5)
        new_level = s.reduce()
        assert new_level < 10
        assert s.current_level() == new_level

    def test_reduce_clamps_to_min(self):
        s = ExponentialBackoffSchedule(max_hints=2, min_level=0, reduction_rate=0.5)
        s.reduce()
        s.reduce()
        assert s.current_level() == 0
        assert s.is_autonomous()

    def test_should_reduce_tracks_wins(self):
        s = ExponentialBackoffSchedule(max_hints=10, reduction_rate=0.5)
        # threshold = 1 + (10 - 10) = 1, so 1 win should be enough
        assert s.should_reduce(True)
        assert not s.should_reduce(False)  # resets streak

    def test_should_reduce_requires_more_wins_near_autonomy(self):
        s = ExponentialBackoffSchedule(max_hints=10, reduction_rate=0.5)
        # Reduce down to level 5 manually
        s._level = 5
        # threshold = 1 + (10 - 5) = 6
        for _ in range(5):
            assert not s.should_reduce(True)
        assert s.should_reduce(True)  # 6th consecutive win

    def test_consecutive_wins_reset_on_loss(self):
        s = ExponentialBackoffSchedule(max_hints=10, reduction_rate=0.5)
        s._level = 5
        for _ in range(3):
            s.should_reduce(True)
        s.should_reduce(False)  # reset
        # Now need 6 more wins
        for _ in range(5):
            assert not s.should_reduce(True)
        assert s.should_reduce(True)

    def test_is_autonomous_at_zero(self):
        s = ExponentialBackoffSchedule(max_hints=0)
        assert s.is_autonomous()

    def test_repr(self):
        s = ExponentialBackoffSchedule(max_hints=10)
        r = repr(s)
        assert "level=10/10" in r
        assert "progress=0%" in r

    def test_repr_after_reduction(self):
        s = ExponentialBackoffSchedule(max_hints=10)
        s.reduce()
        r = repr(s)
        assert "progress=" in r
