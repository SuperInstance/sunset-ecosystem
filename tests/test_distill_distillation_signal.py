"""Tests for distill.distillation_signal."""

from __future__ import annotations

import pytest

from distill.distillation_signal import DistillationGuidance, DistillationSignal
from ranking.ranked_response import RankedResponse
from ranking.user_ranking import UserRanking


class TestDistillationGuidance:
    def test_defaults(self):
        g = DistillationGuidance()
        assert not g.reduce_hints
        assert g.personalization_tags == []
        assert g.confidence_delta == 0.0
        assert g.distilled_rank == 999

    def test_repr(self):
        g = DistillationGuidance(reduce_hints=True, confidence_delta=0.1)
        r = repr(g)
        assert "reduce=True" in r
        assert "Δconf=+0.10" in r


class TestDistillationSignal:
    def _make_ranking(self, distilled_rank=1, big_rank=2, notes=""):
        return UserRanking(
            prompt="test",
            responses=[
                RankedResponse(
                    response="distilled", source="distilled_v3", rank=distilled_rank
                ),
                RankedResponse(response="big", source="gpt-4", rank=big_rank),
            ],
            user_notes=notes,
        )

    def test_distilled_wins(self):
        sig = DistillationSignal()
        g = sig.process_ranking(self._make_ranking(distilled_rank=1, big_rank=2))
        assert g.reduce_hints
        assert g.confidence_delta > 0.0

    def test_big_model_wins(self):
        sig = DistillationSignal()
        g = sig.process_ranking(self._make_ranking(distilled_rank=2, big_rank=1))
        assert not g.reduce_hints
        assert g.confidence_delta < 0.0

    def test_preference_tags_extracted(self):
        sig = DistillationSignal()
        g = sig.process_ranking(self._make_ranking(notes="I like concise answers"))
        assert "concise" in g.personalization_tags

    def test_guidance_history(self):
        sig = DistillationSignal()
        sig.process_ranking(self._make_ranking())
        sig.process_ranking(self._make_ranking())
        assert len(sig.guidance_history) == 2

    def test_reduction_rate(self):
        sig = DistillationSignal()
        sig.process_ranking(self._make_ranking(distilled_rank=1, big_rank=2))
        sig.process_ranking(self._make_ranking(distilled_rank=2, big_rank=1))
        assert sig.reduction_rate == pytest.approx(0.5)

    def test_reduction_rate_empty(self):
        sig = DistillationSignal()
        assert sig.reduction_rate == 0.0

    def test_repr(self):
        sig = DistillationSignal()
        assert "rankings_processed=0" in repr(sig)
