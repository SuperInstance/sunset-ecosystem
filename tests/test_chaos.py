"""Tests for chaos injection and delta tracker."""

import pytest
import time

from swarm.chaos import ChaosProbability, ChaosEvent, inject_chaos
from distill.delta_tracker import DeltaSnapshot, DeltaTracker


class TestChaosProbability:
    def test_initial_state(self):
        cp = ChaosProbability(initial=0.3, decay=0.95, minimum=0.01)
        assert cp.current == 0.3
        assert cp.initial == 0.3
        assert cp.decay == 0.95
        assert cp.minimum == 0.01

    def test_update_decays(self):
        cp = ChaosProbability(initial=0.3, decay=0.95, minimum=0.01)
        original = cp.current
        cp.update(adaptation_score=0.0)
        assert cp.current < original

    def test_update_with_high_adaptation_decays_faster(self):
        cp_low = ChaosProbability(initial=0.3, decay=0.95, minimum=0.01)
        cp_high = ChaosProbability(initial=0.3, decay=0.95, minimum=0.01)
        cp_low.update(adaptation_score=0.0)
        cp_high.update(adaptation_score=1.0)
        assert cp_high.current < cp_low.current

    def test_never_goes_below_minimum(self):
        cp = ChaosProbability(initial=0.3, decay=0.5, minimum=0.05)
        for _ in range(100):
            cp.update(adaptation_score=1.0)
        assert cp.current >= cp.minimum

    def test_reset(self):
        cp = ChaosProbability(initial=0.3, decay=0.95, minimum=0.01)
        cp.update(0.5)
        assert cp.current < 0.3
        cp.reset()
        assert cp.current == 0.3

    def test_repr(self):
        cp = ChaosProbability()
        assert "ChaosProbability" in repr(cp)


class TestChaosEvent:
    def test_creation(self):
        event = ChaosEvent(
            original_route="A→B",
            new_route="A→C",
            reason="swap",
        )
        assert event.original_route == "A→B"
        assert event.new_route == "A→C"
        assert event.reason == "swap"
        assert event.timestamp > 0

    def test_repr(self):
        event = ChaosEvent("A→B", "A→C", "swap")
        assert "swap" in repr(event)


class TestInjectChaos:
    def test_no_chaos_when_prob_zero(self):
        """With minimum chaos, routes should mostly stay the same."""
        cp = ChaosProbability(initial=0.0, minimum=0.0)
        cp.current = 0.0
        routes = {"src1": ["dst1", "dst2"], "src2": ["dst3"]}
        new_routes, events = inject_chaos(routes, cp, adaptation_score=0.0)
        assert len(events) == 0
        assert new_routes["src1"] == ["dst1", "dst2"]
        assert new_routes["src2"] == ["dst3"]

    def test_does_not_mutate_input(self):
        cp = ChaosProbability(initial=0.3, minimum=0.0)
        cp.current = 0.3
        routes = {"src1": ["dst1", "dst2"]}
        original_routes = {"src1": ["dst1", "dst2"]}
        inject_chaos(routes, cp, adaptation_score=0.0)
        assert routes == original_routes

    def test_empty_routes(self):
        cp = ChaosProbability()
        new_routes, events = inject_chaos({}, cp)
        assert new_routes == {}
        assert events == []

    def test_single_source_no_swap_only_reroute(self):
        """With one source, only reroute events can happen (no swaps)."""
        cp = ChaosProbability(initial=1.0, minimum=0.0)
        cp.current = 1.0
        routes = {"src1": ["A", "B", "C", "D"]}
        new_routes, events = inject_chaos(routes, cp)
        # With prob=1.0, all 4 destinations should be touched
        assert len(events) > 0
        # All events should be reroute (can't swap with single source)
        for e in events:
            assert e.reason == "reroute"

    def test_returns_chaos_events(self):
        cp = ChaosProbability(initial=1.0, minimum=0.0)
        cp.current = 1.0
        routes = {"src1": ["A", "B"], "src2": ["C", "D"]}
        new_routes, events = inject_chaos(routes, cp)
        assert len(events) > 0
        for e in events:
            assert isinstance(e, ChaosEvent)
            assert e.reason in ("swap", "reroute")

    def test_adaptation_reduces_chaos(self):
        """Higher adaptation score should result in fewer chaos events on average."""
        import random
        random.seed(42)

        total_events_low_adapt = 0
        total_events_high_adapt = 0

        for _ in range(50):
            routes = {"s1": ["a", "b", "c"], "s2": ["d", "e", "f"]}

            cp_low = ChaosProbability(initial=0.3, minimum=0.01)
            _, events_low = inject_chaos(routes, cp_low, adaptation_score=0.0)

            cp_high = ChaosProbability(initial=0.3, minimum=0.01)
            _, events_high = inject_chaos(routes, cp_high, adaptation_score=0.9)

            total_events_low_adapt += len(events_low)
            total_events_high_adapt += len(events_high)

        # Higher adaptation should result in fewer events (chaos decays more)
        assert total_events_high_adapt <= total_events_low_adapt


class TestDeltaSnapshot:
    def test_creation(self):
        s = DeltaSnapshot(generation=5, avg_quality=0.85, hint_level=3)
        assert s.generation == 5
        assert s.avg_quality == 0.85
        assert s.hint_level == 3
        assert s.timestamp > 0

    def test_repr(self):
        s = DeltaSnapshot(generation=1, avg_quality=0.5, hint_level=10)
        assert "gen=1" in repr(s)


class TestDeltaTracker:
    def test_empty(self):
        dt = DeltaTracker()
        assert dt.latest is None
        assert dt.delta() == 0.0
        assert not dt.is_regression()
        assert not dt.should_revert()
        assert dt.best_generation() is None

    def test_record_and_latest(self):
        dt = DeltaTracker()
        s = dt.record(generation=1, avg_quality=0.6, hint_level=10)
        assert dt.latest == s
        assert dt.latest.generation == 1

    def test_delta_improvement(self):
        dt = DeltaTracker()
        dt.record(1, 0.5, 10)
        dt.record(2, 0.7, 9)
        dt.record(3, 0.8, 8)
        assert dt.delta(n_generations=2) == pytest.approx(0.3)
        assert dt.delta(n_generations=1) == pytest.approx(0.1)

    def test_delta_regression(self):
        dt = DeltaTracker()
        dt.record(1, 0.8, 10)
        dt.record(2, 0.6, 9)
        dt.record(3, 0.4, 8)
        assert dt.delta(n_generations=2) == pytest.approx(-0.4)

    def test_delta_single_snapshot(self):
        dt = DeltaTracker()
        dt.record(1, 0.5, 10)
        assert dt.delta() == 0.0

    def test_is_regression_true(self):
        dt = DeltaTracker()
        dt.record(1, 0.8, 10)
        dt.record(2, 0.7, 9)
        dt.record(3, 0.6, 8)
        dt.record(4, 0.5, 7)
        assert dt.is_regression(negative_streak=3)

    def test_is_regression_false_with_recovery(self):
        dt = DeltaTracker()
        dt.record(1, 0.8, 10)
        dt.record(2, 0.7, 9)
        dt.record(3, 0.6, 8)
        dt.record(4, 0.7, 7)  # Recovery breaks streak
        assert not dt.is_regression(negative_streak=3)

    def test_is_regression_not_enough_data(self):
        dt = DeltaTracker()
        dt.record(1, 0.8, 10)
        dt.record(2, 0.7, 9)
        assert not dt.is_regression(negative_streak=3)

    def test_should_revert(self):
        dt = DeltaTracker()
        dt.record(1, 0.9, 10)
        dt.record(2, 0.8, 9)
        dt.record(3, 0.7, 8)
        dt.record(4, 0.6, 7)
        assert dt.should_revert()

    def test_best_generation(self):
        dt = DeltaTracker()
        dt.record(1, 0.5, 10)
        dt.record(2, 0.9, 8)
        dt.record(3, 0.7, 6)
        best = dt.best_generation()
        assert best is not None
        assert best.generation == 2
        assert best.avg_quality == 0.9

    def test_trend_improving(self):
        dt = DeltaTracker()
        dt.record(1, 0.3, 10)
        dt.record(2, 0.5, 9)
        dt.record(3, 0.7, 8)
        dt.record(4, 0.9, 7)
        assert dt.trend() > 0

    def test_trend_declining(self):
        dt = DeltaTracker()
        dt.record(1, 0.9, 10)
        dt.record(2, 0.7, 9)
        dt.record(3, 0.5, 8)
        assert dt.trend() < 0

    def test_repr(self):
        dt = DeltaTracker()
        assert "empty" in repr(dt)
        dt.record(1, 0.5, 10)
        assert "latest_gen=1" in repr(dt)
