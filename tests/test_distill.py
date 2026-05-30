"""Tests for distill modules: prompt_history, hint_schedule, backtest_runner, distillation_signal."""

import pytest

from distill.prompt_history import PromptHistory, PromptRecord
from distill.hint_schedule import ExponentialBackoffSchedule
from distill.backtest_runner import BacktestRunner
from distill.distillation_signal import DistillationSignal, DistillationGuidance
from ranking.ranked_response import RankedResponse
from ranking.user_ranking import UserRanking


class TestPromptHistory:
    def test_add_and_count(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="hello", response="world"))
        assert h.count() == 1

    def test_query_by_model(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", model="gpt-4"))
        h.add(PromptRecord(prompt="c", response="d", model="claude"))
        results = h.query(model="gpt-4")
        assert len(results) == 1

    def test_query_by_hint_level(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", hint_level=5))
        h.add(PromptRecord(prompt="c", response="d", hint_level=0))
        results = h.query(hint_level=0)
        assert len(results) == 1
        assert results[0].hint_level == 0

    def test_random(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b"))
        h.add(PromptRecord(prompt="c", response="d"))
        r = h.random()
        assert r is not None
        assert r.prompt in ("a", "c")

    def test_max_records(self):
        h = PromptHistory(max_records=2)
        h.add(PromptRecord(prompt="1", response="1"))
        h.add(PromptRecord(prompt="2", response="2"))
        h.add(PromptRecord(prompt="3", response="3"))
        assert h.count() == 2

    def test_average_quality(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="a", response="b", quality_score=0.8))
        h.add(PromptRecord(prompt="c", response="d", quality_score=0.6))
        assert h.average_quality() == pytest.approx(0.7)


class TestHintSchedule:
    def test_initial_level(self):
        s = ExponentialBackoffSchedule(max_hints=10)
        assert s.current_level() == 10

    def test_reduce(self):
        s = ExponentialBackoffSchedule(max_hints=10, reduction_rate=0.5)
        new = s.reduce()
        assert new < 10

    def test_progressive_reduction(self):
        s = ExponentialBackoffSchedule(max_hints=10, reduction_rate=0.5)
        levels = [s.current_level()]
        while not s.is_autonomous():
            s.reduce()
            levels.append(s.current_level())
        # Should reduce over time
        assert levels[-1] <= levels[0]

    def test_should_reduce(self):
        s = ExponentialBackoffSchedule(max_hints=10)
        # First reduction: need 1 consecutive win
        assert s.should_reduce(True) is True

    def test_autonomous(self):
        s = ExponentialBackoffSchedule(max_hints=0)
        assert s.is_autonomous() is True


class TestBacktestRunner:
    def test_run_cycle(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="test prompt", response="test response", quality_score=0.5))
        runner = BacktestRunner(h)
        result = runner.run_cycle(hint_level=5, spare_capacity=0.8)
        assert result is not None
        assert runner.backtests_run == 1

    def test_skip_low_capacity(self):
        h = PromptHistory()
        runner = BacktestRunner(h, spare_threshold=0.3)
        result = runner.run_cycle(spare_capacity=0.1)
        assert result is None

    def test_improvement_trend(self):
        h = PromptHistory()
        h.add(PromptRecord(prompt="test", response="test", quality_score=0.3))
        runner = BacktestRunner(h)
        for _ in range(5):
            runner.run_cycle(spare_capacity=0.8)
        trend = runner.improvement_trend()
        assert isinstance(trend, float)


class TestDistillationSignal:
    def test_distilled_wins(self):
        signal = DistillationSignal()
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="d", source="distilled_v1", rank=1),
                RankedResponse(response="b", source="gpt-4", rank=2),
            ],
            user_notes="concise",
        )
        guidance = signal.process_ranking(ranking)
        assert guidance.reduce_hints is True
        assert guidance.distilled_rank == 1

    def test_big_model_wins(self):
        signal = DistillationSignal()
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="d", source="distilled_v1", rank=2),
                RankedResponse(response="b", source="gpt-4", rank=1),
            ],
        )
        guidance = signal.process_ranking(ranking)
        assert guidance.reduce_hints is False

    def test_reduction_rate(self):
        signal = DistillationSignal()
        for win in [True, True, False, True]:
            ranking = UserRanking(
                prompt="test",
                responses=[
                    RankedResponse(response="d", source="distilled", rank=1 if win else 2),
                    RankedResponse(response="b", source="gpt-4", rank=2 if win else 1),
                ],
            )
            signal.process_ranking(ranking)
        assert signal.reduction_rate == 0.75
