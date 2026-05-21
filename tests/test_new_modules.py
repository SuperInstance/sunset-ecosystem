"""Tests for distill, ranking, and swarm modules."""

import pytest

from ranking.ranked_response import RankedResponse
from ranking.user_ranking import UserRanking
from ranking.personalization import PersonalizationStore
from distill.prompt_history import PromptHistory, PromptRecord
from distill.hint_schedule import ExponentialBackoffSchedule
from distill.backtest_runner import BacktestRunner, BacktestResult
from distill.distillation_signal import DistillationSignal, DistillationGuidance
from swarm.penrose import PenrosePosition, assign_positions, compute_overlap, minimum_overlap
from swarm.broadcast import BroadcastMessage, BroadcastingChannel
from swarm.swarm_runner import SwarmRunner, SwarmStatus
from nerve.fiber import NerveFiber


# ── Ranking Tests ────────────────────────────────────────────


class TestRankedResponse:
    def test_create(self):
        r = RankedResponse(response="hello", source="gpt-4", rank=1)
        assert r.response == "hello"
        assert r.rank == 1

    def test_is_distilled(self):
        r1 = RankedResponse(response="x", source="distilled_v3")
        assert r1.is_distilled is True
        r2 = RankedResponse(response="x", source="gpt-4")
        assert r2.is_distilled is False

    def test_repr(self):
        r = RankedResponse(response="test", source="model", rank=2, hint_level=5)
        assert "model" in repr(r)


class TestUserRanking:
    def test_best_response(self):
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="a", rank=2),
                RankedResponse(response="b", rank=1),
            ],
        )
        assert ranking.best_response().response == "b"

    def test_worst_response(self):
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="a", rank=1),
                RankedResponse(response="b", rank=3),
            ],
        )
        assert ranking.worst_response().response == "b"

    def test_distilled_beats_big(self):
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="dist", source="distilled_v3", rank=1),
                RankedResponse(response="big", source="gpt-4", rank=2),
            ],
        )
        assert ranking.distilled_beats_big_model() is True

    def test_big_beats_distilled(self):
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="dist", source="distilled_v3", rank=2),
                RankedResponse(response="big", source="gpt-4", rank=1),
            ],
        )
        assert ranking.distilled_beats_big_model() is False

    def test_preference_tags(self):
        ranking = UserRanking(
            prompt="test",
            responses=[],
            user_notes="I prefer concise and correct answers",
        )
        tags = ranking.preference_tags
        assert "concise" in tags
        assert "correct" in tags

    def test_empty_responses(self):
        ranking = UserRanking(prompt="test")
        assert ranking.best_response() is None
        assert ranking.worst_response() is None


class TestPersonalizationStore:
    def test_ingest_and_get(self):
        store = PersonalizationStore()
        ranking = UserRanking(
            prompt="test",
            responses=[],
            user_notes="concise and helpful",
        )
        store.ingest(ranking)
        truths = store.get_ground_truths()
        assert "concise" in truths
        assert "helpful" in truths

    def test_score_response(self):
        store = PersonalizationStore()
        ranking = UserRanking(prompt="test", responses=[], user_notes="concise")
        store.ingest(ranking)
        score = store.score_response(["concise"])
        assert score.score > 0

    def test_decay(self):
        store = PersonalizationStore(decay_factor=0.5)
        ranking = UserRanking(prompt="test", responses=[], user_notes="concise")
        store.ingest(ranking)
        # Ingest another that doesn't have concise
        ranking2 = UserRanking(prompt="test2", responses=[], user_notes="thorough")
        store.ingest(ranking2)
        truths = store.get_ground_truths()
        # Both should exist but weights differ
        assert "concise" in truths or "thorough" in truths

    def test_ranking_count(self):
        store = PersonalizationStore()
        assert store.ranking_count == 0
        store.ingest(UserRanking(prompt="t", responses=[]))
        assert store.ranking_count == 1


# ── Distill Tests ────────────────────────────────────────────


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


# ── Swarm Tests ──────────────────────────────────────────────


class TestPenrose:
    def test_assign_positions(self):
        agents = [f"agent-{i}" for i in range(12)]
        positions = assign_positions(agents)
        assert len(positions) == 12
        assert all(isinstance(p, PenrosePosition) for p in positions)

    def test_unique_positions(self):
        agents = [f"a{i}" for i in range(20)]
        positions = assign_positions(agents)
        coords = [(p.x, p.y) for p in positions]
        assert len(set(coords)) == 20  # All unique

    def test_compute_overlap(self):
        p1 = PenrosePosition("a", 0.0, 0.0, 0, 0.0)
        p2 = PenrosePosition("b", 0.5, 0.0, 0, 0.0)
        overlap = compute_overlap(p1, p2, radius=1.0)
        assert 0.0 < overlap < 1.0

    def test_no_overlap(self):
        p1 = PenrosePosition("a", 0.0, 0.0, 0, 0.0)
        p2 = PenrosePosition("b", 10.0, 10.0, 0, 0.0)
        assert compute_overlap(p1, p2, radius=1.0) == 0.0

    def test_full_overlap(self):
        p1 = PenrosePosition("a", 0.0, 0.0, 0, 0.0)
        assert compute_overlap(p1, p1, radius=1.0) == 1.0

    def test_minimum_overlap(self):
        agents = [f"a{i}" for i in range(12)]
        positions = assign_positions(agents)
        min_ov = minimum_overlap(positions, radius=2.0)
        assert 0.0 <= min_ov <= 1.0


class TestBroadcast:
    def test_subscribe_and_broadcast(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent-1", "room-1")
        msg = BroadcastMessage(content="hello", source_agent="src", target_room="room-1")
        recipients = ch.broadcast(msg)
        assert "agent-1" in recipients

    def test_no_match(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent-1", "room-1")
        msg = BroadcastMessage(content="hello", source_agent="src", target_room="room-2")
        recipients = ch.broadcast(msg)
        assert "agent-1" not in recipients

    def test_receive(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent-1", "room-1")
        ch.broadcast(BroadcastMessage(content="test", target_room="room-1"))
        msgs = ch.receive("agent-1")
        assert len(msgs) == 1

    def test_feedback(self):
        ch = BroadcastingChannel()
        ch.subscribe("a", "r")
        ch.broadcast(BroadcastMessage(content="x", source_agent="src", target_room="r"))
        w1 = ch.get_channel_weight("src", "a")
        ch.feedback("src", "a", useful=True)
        w2 = ch.get_channel_weight("src", "a")
        assert w2 > w1


class TestSwarmRunner:
    def test_create(self):
        runner = SwarmRunner()
        assert runner.status().total_agents == 0

    def test_add_fiber_and_tick(self):
        runner = SwarmRunner()
        runner.add_fiber(NerveFiber("f1", epsilon=0.2))
        result = runner.tick("test signal")
        assert "tiles" in result
        assert "f1" in result["tiles"]

    def test_distribute(self):
        runner = SwarmRunner()
        agents = [f"agent-{i}" for i in range(12)]
        positions = runner.distribute(agents)
        assert len(positions) == 12

    def test_status(self):
        runner = SwarmRunner()
        runner.add_fiber(NerveFiber("f1"))
        status = runner.status()
        assert isinstance(status, SwarmStatus)

    def test_spare_capacity(self):
        runner = SwarmRunner()
        runner.add_fiber(NerveFiber("f1"))
        cap = runner.spare_capacity()
        assert 0.0 <= cap <= 1.0

    def test_backtest_cycle(self):
        runner = SwarmRunner()
        # No spare capacity initially (adaptation=0)
        assert runner.run_backtest_cycle() is False
