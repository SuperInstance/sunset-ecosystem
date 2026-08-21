"""Tests for distill.backtest_runner."""

from __future__ import annotations

import pytest

from distill.backtest_runner import BacktestResult, BacktestRunner
from distill.prompt_history import PromptHistory, PromptRecord


class TestBacktestResult:
    def test_creation(self):
        r = BacktestResult(
            prompt="test",
            reference_response="ref",
            distilled_response="dist",
            similarity=0.85,
        )
        assert r.similarity == 0.85
        assert not r.improved

    def test_repr(self):
        r = BacktestResult(
            prompt="test",
            reference_response="ref",
            distilled_response="dist",
            similarity=0.75,
            hint_level=3,
        )
        assert "sim=0.75" in repr(r)


class TestBacktestRunner:
    def _make_history(self, n=5):
        h = PromptHistory()
        for i in range(n):
            h.add(
                PromptRecord(
                    prompt=f"prompt {i} with words",
                    response=f"response {i} with words",
                    quality_score=0.5,
                )
            )
        return h

    def test_no_capacity_skips(self):
        h = self._make_history()
        runner = BacktestRunner(h, spare_threshold=0.5)
        result = runner.run_cycle(hint_level=3, spare_capacity=0.2)
        assert result is None

    def test_run_cycle_success(self):
        h = self._make_history()
        runner = BacktestRunner(h, spare_threshold=0.1)
        result = runner.run_cycle(hint_level=3, spare_capacity=0.8)
        assert result is not None
        assert runner.backtests_run == 1

    def test_empty_history_skips(self):
        h = PromptHistory()
        runner = BacktestRunner(h, spare_threshold=0.1)
        result = runner.run_cycle(hint_level=3, spare_capacity=0.8)
        assert result is None

    def test_similarity_computation(self):
        sim = BacktestRunner._compute_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0)

    def test_similarity_partial(self):
        sim = BacktestRunner._compute_similarity("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_similarity_empty(self):
        sim = BacktestRunner._compute_similarity("", "hello")
        assert sim == 0.0

    def test_improvement_trend(self):
        h = self._make_history(20)
        runner = BacktestRunner(h, spare_threshold=0.0)
        # Run several cycles
        for _ in range(5):
            runner.run_cycle(hint_level=3, spare_capacity=1.0)
        trend = runner.improvement_trend()
        assert isinstance(trend, float)

    def test_improvement_trend_insufficient(self):
        h = self._make_history()
        runner = BacktestRunner(h, spare_threshold=0.0)
        runner.run_cycle(hint_level=3, spare_capacity=1.0)
        assert runner.improvement_trend() == 0.0

    def test_repr(self):
        h = self._make_history()
        runner = BacktestRunner(h)
        r = repr(runner)
        assert "run=0" in r
