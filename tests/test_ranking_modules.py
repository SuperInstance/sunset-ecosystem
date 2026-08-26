"""Tests for ranking modules: ranked_response, user_ranking, personalization, feedback_loop."""

from __future__ import annotations

import pytest

from ranking.ranked_response import RankedResponse
from ranking.user_ranking import UserRanking
from ranking.personalization import PersonalizationStore
from ranking.feedback_loop import FeedbackLoop
from distill.hint_schedule import ExponentialBackoffSchedule
from distill.distillation_signal import DistillationSignal


class TestRankedResponse:
    def test_distilled_detection(self):
        r = RankedResponse(response="test", source="distilled_v3")
        assert r.is_distilled
        assert not r.is_big_model

    def test_nerve_is_distilled(self):
        r = RankedResponse(response="test", source="nerve_compiled")
        assert r.is_distilled

    def test_big_model_detection(self):
        r = RankedResponse(response="test", source="gpt-4")
        assert not r.is_distilled
        assert r.is_big_model

    def test_repr(self):
        r = RankedResponse(response="test", source="gpt-4", rank=1)
        assert "gpt-4" in repr(r)
        assert "rank=1" in repr(r)


class TestUserRanking:
    def _make_responses(self):
        return [
            RankedResponse(response="distilled", source="distilled_v3", rank=1),
            RankedResponse(response="big", source="gpt-4", rank=2),
        ]

    def test_best_response(self):
        ranking = UserRanking(prompt="test", responses=self._make_responses())
        best = ranking.best_response()
        assert best is not None
        assert best.rank == 1

    def test_worst_response(self):
        ranking = UserRanking(prompt="test", responses=self._make_responses())
        worst = ranking.worst_response()
        assert worst is not None
        assert worst.rank == 2

    def test_best_response_none(self):
        ranking = UserRanking(prompt="test", responses=[])
        assert ranking.best_response() is None

    def test_distilled_beats_big_model(self):
        ranking = UserRanking(prompt="test", responses=self._make_responses())
        assert ranking.distilled_beats_big_model()

    def test_distilled_does_not_beat(self):
        responses = [
            RankedResponse(response="distilled", source="distilled_v3", rank=2),
            RankedResponse(response="big", source="gpt-4", rank=1),
        ]
        ranking = UserRanking(prompt="test", responses=responses)
        assert not ranking.distilled_beats_big_model()

    def test_preference_tags(self):
        ranking = UserRanking(
            prompt="test",
            responses=[],
            user_notes="This was concise and helpful but too abstract",
        )
        tags = ranking.preference_tags
        assert "concise" in tags
        assert "helpful" in tags
        assert "too_abstract" in tags

    def test_preference_tags_empty_notes(self):
        ranking = UserRanking(prompt="test", responses=[], user_notes="")
        assert ranking.preference_tags == []

    def test_repr(self):
        ranking = UserRanking(prompt="a" * 50, responses=[])
        r = repr(ranking)
        assert "responses=0" in r


class TestPersonalizationStore:
    def _make_ranking(self, notes=""):
        return UserRanking(
            prompt="test",
            responses=[],
            user_notes=notes,
        )

    def test_ingest_builds_weights(self):
        store = PersonalizationStore()
        store.ingest(self._make_ranking("I like concise helpful answers"))
        gt = store.get_ground_truths()
        assert "concise" in gt
        assert "helpful" in gt

    def test_decay_factor(self):
        store = PersonalizationStore(decay_factor=0.5)
        store.ingest(self._make_ranking("concise"))
        store.ingest(self._make_ranking("thorough"))
        top = store.get_top_preferences(5)
        # "thorough" was just reinforced, "concise" was decayed
        tags = [t for t, _ in top]
        assert "thorough" in tags

    def test_score_response(self):
        store = PersonalizationStore()
        store.ingest(self._make_ranking("concise helpful"))
        score = store.score_response(["concise"])
        assert score.score > 0.0
        assert "concise" in score.matched_tags

    def test_score_response_empty_store(self):
        store = PersonalizationStore()
        score = store.score_response(["concise"])
        assert score.score == 0.5  # neutral

    def test_get_top_preferences(self):
        store = PersonalizationStore()
        store.ingest(self._make_ranking("concise"))
        store.ingest(self._make_ranking("concise"))
        store.ingest(self._make_ranking("thorough"))
        top = store.get_top_preferences(1)
        assert top[0][0] == "concise"

    def test_ranking_count(self):
        store = PersonalizationStore()
        assert store.ranking_count == 0
        store.ingest(self._make_ranking())
        assert store.ranking_count == 1

    def test_repr(self):
        store = PersonalizationStore()
        r = repr(store)
        assert "rankings=0" in r


class TestFeedbackLoop:
    def _make_ranking(self, distilled_rank=1, big_rank=2, notes=""):
        return UserRanking(
            prompt="test",
            responses=[
                RankedResponse(
                    response="d", source="distilled_v3", rank=distilled_rank
                ),
                RankedResponse(response="b", source="gpt-4", rank=big_rank),
            ],
            user_notes=notes,
        )

    def test_ingest_distilled_wins(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        personalization = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, personalization, signal)

        # Need enough consecutive wins to trigger reduction
        for _ in range(10):
            guidance = loop.ingest(self._make_ranking(distilled_rank=1, big_rank=2))
        assert loop.total_rankings == 10

    def test_ingest_returns_guidance(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        personalization = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, personalization, signal)

        guidance = loop.ingest(self._make_ranking(distilled_rank=1, big_rank=2))
        assert guidance.reduce_hints

    def test_get_status(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        personalization = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, personalization, signal)

        loop.ingest(self._make_ranking(distilled_rank=1, big_rank=2))
        status = loop.get_status()
        assert "total_rankings" in status
        assert "hint_level" in status
        assert status["total_rankings"] == 1

    def test_repr(self):
        schedule = ExponentialBackoffSchedule(max_hints=10)
        personalization = PersonalizationStore()
        signal = DistillationSignal()
        loop = FeedbackLoop(schedule, personalization, signal)
        r = repr(loop)
        assert "rankings=0" in r
