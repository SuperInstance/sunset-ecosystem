"""Tests for AdaptationEngine and ShoeTracker — the shoe metaphor lifecycle.

Tracks notice_level decay across COMPILED, ADAPTING, and NOVELTY_ALERT states.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from nerve.adaptation import AdaptationEngine, ShoeTracker, ShoeState
from nerve.fiber import NerveFiber, FiberState


# ---------------------------------------------------------------------------
# ShoeState
# ---------------------------------------------------------------------------

class TestShoeState:
    def test_creation(self):
        s = ShoeState(pattern_id="abc")
        assert s.pattern_id == "abc"
        assert s.notice_level == 1.0
        assert s.steps == 0
        assert s.last_state == FiberState.PERCEIVING

    def test_repr(self):
        s = ShoeState(pattern_id="abc", notice_level=0.5, steps=5, last_state=FiberState.COMPILED)
        r = repr(s)
        assert "ShoeState" in r
        assert "notice=0.50" in r
        assert "state=compiled" in r


# ---------------------------------------------------------------------------
# ShoeTracker
# ---------------------------------------------------------------------------

class TestShoeTracker:
    def test_empty_adaptation(self):
        tracker = ShoeTracker()
        assert tracker.adaptation_score == 0.0
        assert tracker.compiled_count == 0
        assert tracker.total_patterns == 0

    def test_step_new_pattern(self):
        tracker = ShoeTracker()
        shoe = tracker.step("p1", FiberState.PERCEIVING)
        assert shoe.pattern_id == "p1"
        assert shoe.steps == 1
        assert shoe.notice_level == 1.0
        assert shoe.last_state == FiberState.PERCEIVING
        assert tracker.total_patterns == 1

    def test_step_compiled(self):
        tracker = ShoeTracker()
        tracker.step("p1", FiberState.COMPILED)
        assert tracker.compiled_count == 1
        shoe = tracker._shoes["p1"]
        assert shoe.notice_level == 0.8  # 1.0 - 0.2

    def test_step_adapting(self):
        tracker = ShoeTracker()
        tracker.step("p1", FiberState.ADAPTING)
        shoe = tracker._shoes["p1"]
        assert shoe.notice_level == 0.95  # 1.0 - 0.05

    def test_step_novelty(self):
        tracker = ShoeTracker()
        tracker.step("p1", FiberState.NOVELTY_ALERT)
        shoe = tracker._shoes["p1"]
        assert shoe.notice_level == 1.0  # 1.0 + 0.5 capped at 1.0

    def test_step_novelty_floor(self):
        tracker = ShoeTracker()
        # Set notice low, then novelty alert should bump it up
        tracker._shoes["p1"] = ShoeState(pattern_id="p1", notice_level=0.2)
        tracker.step("p1", FiberState.NOVELTY_ALERT)
        shoe = tracker._shoes["p1"]
        assert shoe.notice_level == 0.7  # 0.2 + 0.5

    def test_adaptation_score(self):
        tracker = ShoeTracker()
        tracker.step("p1", FiberState.COMPILED)
        tracker.step("p2", FiberState.PERCEIVING)
        # p1: 0.8, p2: 1.0  → average = 0.9  → score = 1 - 0.9 = 0.1
        assert tracker.adaptation_score == pytest.approx(0.1, abs=0.01)

    def test_repr(self):
        tracker = ShoeTracker()
        tracker.step("p1", FiberState.COMPILED)
        r = repr(tracker)
        assert "ShoeTracker" in r
        assert "total=1" in r
        assert "compiled=1" in r

    def test_thread_safety(self):
        tracker = ShoeTracker()
        import threading
        results = []
        def worker(n):
            for i in range(10):
                shoe = tracker.step(f"pattern_{n}", FiberState.COMPILED)
            results.append(shoe)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert tracker.total_patterns == 5


# ---------------------------------------------------------------------------
# AdaptationEngine
# ---------------------------------------------------------------------------

class TestAdaptationEngine:
    def _mock_fiber(self, state=FiberState.PERCEIVING, confidence=0.5, pattern_id="mock"):
        fiber = MagicMock(spec=NerveFiber)
        fiber.fiber_id = "f1"
        fiber.state = state
        fiber.confidence = confidence
        tile = MagicMock()
        tile.pattern_id = pattern_id
        tile.state = state
        fiber.perceive.return_value = tile
        fiber.stats = {"perceive_calls": 1}
        return fiber

    def test_init_empty(self):
        engine = AdaptationEngine()
        assert engine.adaptation_score == 0.0
        assert engine.shoe_tracker.total_patterns == 0

    def test_register(self):
        engine = AdaptationEngine()
        fiber = self._mock_fiber()
        engine.register(fiber)
        assert "f1" in engine._fibers

    def test_process_signal(self):
        fiber = self._mock_fiber(state=FiberState.COMPILED)
        engine = AdaptationEngine({"f1": fiber})
        result = engine.process_signal("f1", "signal")
        assert "tile" in result
        assert "shoe" in result
        assert result["fiber_state"] == FiberState.COMPILED
        assert "adaptation_score" in result

    def test_process_signal_unknown_fiber(self):
        engine = AdaptationEngine()
        result = engine.process_signal("f1", "signal")
        assert "error" in result

    def test_system_status(self):
        fiber = self._mock_fiber(state=FiberState.COMPILED)
        engine = AdaptationEngine({"f1": fiber})
        engine.process_signal("f1", "signal")
        status = engine.system_status()
        assert status["total_fibers"] == 1
        assert status["compiled"] == 1
        assert status["perceiving"] == 0
        assert status["adaptation_score"] == pytest.approx(0.2, abs=0.01)

    def test_repr(self):
        engine = AdaptationEngine()
        r = repr(engine)
        assert "AdaptationEngine" in r
