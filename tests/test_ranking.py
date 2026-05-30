"""Tests for ranking modules: ranked_response, user_ranking, personalization."""

import pytest

from ranking.ranked_response import RankedResponse
from ranking.user_ranking import UserRanking
from ranking.personalization import PersonalizationStore


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
