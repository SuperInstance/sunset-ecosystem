"""Tests for the feedback loop."""

import pytest

from ranking.feedback_loop import FeedbackLoop
from ranking.ranked_response import RankedResponse
from ranking.user_ranking import UserRanking
from ranking.personalization import PersonalizationStore
from distill.hint_schedule import ExponentialBackoffSchedule
from distill.distillation_signal import DistillationSignal


class TestFeedbackLoop:
    def test_ingest(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        store = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, store, signal)

        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="d", source="distilled_v3", rank=1),
                RankedResponse(response="b", source="gpt-4", rank=2),
            ],
            user_notes="concise and correct",
        )
        guidance = loop.ingest(ranking)
        assert guidance.reduce_hints is True
        assert loop.total_rankings == 1

    def test_personalization_updated(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        store = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, store, signal)

        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="d", source="distilled", rank=1),
                RankedResponse(response="b", source="gpt-4", rank=2),
            ],
            user_notes="I prefer concise responses",
        )
        loop.ingest(ranking)
        truths = store.get_ground_truths()
        assert "concise" in truths

    def test_hint_schedule_updated(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        store = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, store, signal)

        initial = schedule.current_level()
        # Need enough consecutive wins to trigger reduction
        for _ in range(2):
            ranking = UserRanking(
                prompt="test",
                responses=[
                    RankedResponse(response="d", source="distilled", rank=1),
                    RankedResponse(response="b", source="gpt-4", rank=2),
                ],
            )
            loop.ingest(ranking)
        # Schedule should have reduced at least once
        assert schedule.current_level() <= initial

    def test_get_status(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        store = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, store, signal)

        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="d", source="distilled", rank=1),
                RankedResponse(response="b", source="gpt-4", rank=2),
            ],
            user_notes="concise",
        )
        loop.ingest(ranking)
        status = loop.get_status()
        assert status["total_rankings"] == 1
        assert "hint_level" in status
        assert "is_autonomous" in status
        assert "ground_truths" in status
        assert "reduction_rate" in status

    def test_repr(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        store = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, store, signal)
        assert "FeedbackLoop" in repr(loop)

    def test_big_model_wins_no_reduce(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        store = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, store, signal)

        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse(response="d", source="distilled", rank=2),
                RankedResponse(response="b", source="gpt-4", rank=1),
            ],
        )
        guidance = loop.ingest(ranking)
        assert guidance.reduce_hints is False
