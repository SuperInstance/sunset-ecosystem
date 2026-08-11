"""Tests for BaselineConfidenceGate — the overknowledge prevention gate."""

import pytest
import logging
import io

from distill.baseline_gate import (
    BaselineConfidenceGate,
    GateDecision,
    GateAction,
)


# ── Basic threshold tests ──────────────────────────────────────

class TestSkipThreshold:
    def test_skip_when_above_085(self):
        """Scores > 0.85 must SKIP teaching entirely."""
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.90)
        assert decision.action is GateAction.SKIP
        assert decision.max_passes == 0
        assert decision.hint_level_pct == 0.0
        assert decision.allow_repetition is False

    def test_skip_at_086(self):
        """0.86 is above the default skip threshold."""
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.86)
        assert decision.action is GateAction.SKIP

    def test_skip_reason_contains_baseline_confident(self):
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.99)
        assert "baseline_confident: skip" in decision.reason


class TestFullThreshold:
    def test_full_when_below_060(self):
        """Scores < 0.60 get FULL intensity."""
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.45)
        assert decision.action is GateAction.FULL
        assert decision.max_passes == gate.full_max_passes
        assert decision.hint_level_pct == 100.0
        assert decision.allow_repetition is True

    def test_full_at_030(self):
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.30)
        assert decision.action is GateAction.FULL

    def test_full_reason_contains_baseline_low(self):
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.10)
        assert "baseline_low: full intensity" in decision.reason


class TestReducedZone:
    def test_reduced_between_060_and_085(self):
        """Scores in [0.60, 0.85] get REDUCED intensity."""
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.72)
        assert decision.action is GateAction.REDUCED
        assert decision.max_passes == gate.reduced_max_passes
        assert decision.hint_level_pct == gate.reduced_hint_pct
        assert decision.allow_repetition is False

    def test_reduced_reason_contains_baseline_moderate(self):
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.70)
        assert "baseline_moderate: reduced" in decision.reason


# ── Edge cases at exact boundaries ─────────────────────────────

class TestExactBoundaries:
    def test_exactly_060_is_reduced(self):
        """Score == 0.60 falls in the reduced zone (not full)."""
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.60)
        assert decision.action is GateAction.REDUCED

    def test_exactly_085_is_reduced(self):
        """Score == 0.85 falls in the reduced zone (not skip).

        The skip threshold uses strict > so 0.85 itself is NOT skipped.
        """
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.85)
        assert decision.action is GateAction.REDUCED

    def test_just_above_085_is_skip(self):
        """0.8501 should skip."""
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.8501)
        assert decision.action is GateAction.SKIP

    def test_just_below_060_is_full(self):
        """0.5999 should be full intensity."""
        gate = BaselineConfidenceGate()
        decision = gate.evaluate(0.5999)
        assert decision.action is GateAction.FULL


# ── Logging behavior ───────────────────────────────────────────

class TestLogging:
    def test_skip_logs_baseline_confident(self, caplog):
        """SKIP decisions must log 'baseline_confident: skip'."""
        with caplog.at_level(logging.INFO, logger="distill.baseline_gate"):
            gate = BaselineConfidenceGate()
            gate.evaluate(0.95, topic="photosynthesis")
        assert any("baseline_confident: skip" in r.message for r in caplog.records)

    def test_full_logs_baseline_low(self, caplog):
        with caplog.at_level(logging.INFO, logger="distill.baseline_gate"):
            gate = BaselineConfidenceGate()
            gate.evaluate(0.20, topic="fractions")
        assert any("baseline_low: full intensity" in r.message for r in caplog.records)

    def test_reduced_logs_baseline_moderate(self, caplog):
        with caplog.at_level(logging.INFO, logger="distill.baseline_gate"):
            gate = BaselineConfidenceGate()
            gate.evaluate(0.75, topic="verbs")
        assert any("baseline_moderate: reduced" in r.message for r in caplog.records)

    def test_topic_appears_in_log(self, caplog):
        with caplog.at_level(logging.INFO, logger="distill.baseline_gate"):
            gate = BaselineConfidenceGate()
            gate.evaluate(0.90, topic="kernels")
        assert any("kernels" in r.message for r in caplog.records)


# ── History and statistics ─────────────────────────────────────

class TestHistory:
    def test_history_records_decisions(self):
        gate = BaselineConfidenceGate()
        gate.evaluate(0.90)
        gate.evaluate(0.30)
        gate.evaluate(0.70)
        assert len(gate.history) == 3
        assert gate.history[0].action is GateAction.SKIP
        assert gate.history[1].action is GateAction.FULL
        assert gate.history[2].action is GateAction.REDUCED

    def test_skip_rate(self):
        gate = BaselineConfidenceGate()
        gate.evaluate(0.90)  # skip
        gate.evaluate(0.90)  # skip
        gate.evaluate(0.30)  # full
        gate.evaluate(0.70)  # reduced
        assert gate.skip_rate == 0.5

    def test_reset_clears_history(self):
        gate = BaselineConfidenceGate()
        gate.evaluate(0.90)
        gate.evaluate(0.30)
        assert len(gate.history) == 2
        gate.reset()
        assert len(gate.history) == 0
        assert gate.skip_rate == 0.0

    def test_empty_history_skip_rate(self):
        gate = BaselineConfidenceGate()
        assert gate.skip_rate == 0.0


# ── Custom thresholds ──────────────────────────────────────────

class TestCustomThresholds:
    def test_custom_thresholds(self):
        gate = BaselineConfidenceGate(
            skip_threshold=0.95,
            full_threshold=0.50,
        )
        # 0.80 is between 0.50 and 0.95 → reduced
        d = gate.evaluate(0.80)
        assert d.action is GateAction.REDUCED

        # 0.96 > 0.95 → skip
        d = gate.evaluate(0.96)
        assert d.action is GateAction.SKIP

        # 0.40 < 0.50 → full
        d = gate.evaluate(0.40)
        assert d.action is GateAction.FULL

    def test_invalid_thresholds_raise(self):
        """skip_threshold must be > full_threshold."""
        with pytest.raises(ValueError, match="must be greater than"):
            BaselineConfidenceGate(skip_threshold=0.50, full_threshold=0.60)

    def test_custom_pass_counts(self):
        gate = BaselineConfidenceGate(
            full_max_passes=5,
            reduced_max_passes=2,
        )
        d_full = gate.evaluate(0.30)
        assert d_full.max_passes == 5

        d_reduced = gate.evaluate(0.70)
        assert d_reduced.max_passes == 2

    def test_custom_hint_pct(self):
        gate = BaselineConfidenceGate(reduced_hint_pct=25.0)
        d = gate.evaluate(0.70)
        assert d.hint_level_pct == 25.0


# ── GateDecision dataclass ─────────────────────────────────────

class TestGateDecision:
    def test_repr(self):
        d = GateDecision(
            action=GateAction.SKIP,
            baseline_score=0.92,
            max_passes=0,
            hint_level_pct=0.0,
            allow_repetition=False,
            reason="test",
        )
        r = repr(d)
        assert "skip" in r
        assert "0.920" in r
        assert "passes=0" in r
