"""Tests for distillation_signal.py — JEPA-as-signal integration."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from ranking.user_ranking import UserRanking
from ranking.ranked_response import RankedResponse
from distill.distillation_signal import (
    DistillationSignal,
    DistillationGuidance,
    JEPASignalSource,
)


class TestDistillationGuidance:
    def test_repr(self):
        g = DistillationGuidance(reduce_hints=True, confidence_delta=0.1)
        assert "reduce=True" in repr(g)
        assert "Δconf=+0.10" in repr(g)


class TestJEPASignalSource:
    def test_score_response(self):
        encoder = lambda text: np.ones(64, dtype=np.float32) * 0.5
        source = JEPASignalSource(encoder=encoder)
        room = np.ones(64, dtype=np.float32) * 0.5
        score = source.score_response("hello", room)
        assert score == pytest.approx(1.0, 0.01)

    def test_score_response_opposite(self):
        encoder = lambda text: np.ones(64, dtype=np.float32)
        source = JEPASignalSource(encoder=encoder)
        room = -np.ones(64, dtype=np.float32)
        score = source.score_response("hello", room)
        assert score == pytest.approx(-1.0, 0.01)

    def test_score_response_zero_norm(self):
        encoder = lambda text: np.zeros(64, dtype=np.float32)
        source = JEPASignalSource(encoder=encoder)
        room = np.zeros(64, dtype=np.float32)
        score = source.score_response("hello", room)
        assert score == 0.0

    def test_score_response_shape_mismatch(self):
        encoder = lambda text: np.ones(32, dtype=np.float32)
        source = JEPASignalSource(encoder=encoder)
        room = np.ones(64, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            source.score_response("hello", room)

    def test_no_encoder(self):
        source = JEPASignalSource()
        with pytest.raises(RuntimeError, match="no encoder"):
            source._encode("hello")

    def test_encoder_wrong_type(self):
        source = JEPASignalSource(encoder=lambda t: "not an array")
        with pytest.raises(TypeError, match="np.ndarray"):
            source._encode("hello")

    def test_rank_responses(self):
        encoder = lambda text: np.ones(64, dtype=np.float32) * hash(text) % 100 / 100
        source = JEPASignalSource(encoder=encoder)
        room = np.ones(64, dtype=np.float32) * 0.5
        texts = ["a", "b", "c"]
        ranked = source.rank_responses(texts, room)
        assert len(ranked) == 3
        # Sorted by descending score
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


class TestDistillationSignalUserRanking:
    def test_process_ranking_distilled_wins(self):
        ds = DistillationSignal()
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse("distilled", source="distilled_v3", rank=1),
                RankedResponse("big", source="gpt-4", rank=2),
            ],
        )
        g = ds.process_ranking(ranking)
        assert g.reduce_hints is True
        assert g.distilled_rank == 1
        assert g.big_model_rank == 2
        assert g.confidence_delta == 0.1

    def test_process_ranking_big_wins(self):
        ds = DistillationSignal()
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse("distilled", source="distilled_v3", rank=2),
                RankedResponse("big", source="gpt-4", rank=1),
            ],
        )
        g = ds.process_ranking(ranking)
        assert g.reduce_hints is False
        assert g.distilled_rank == 2
        assert g.big_model_rank == 1
        assert g.confidence_delta == -0.05

    def test_process_ranking_tie(self):
        ds = DistillationSignal()
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse("distilled", source="distilled_v3", rank=1),
                RankedResponse("big", source="gpt-4", rank=1),
            ],
        )
        g = ds.process_ranking(ranking)
        assert g.confidence_delta == 0.0

    def test_process_ranking_with_backtester(self):
        backtester = MagicMock()
        backtester.improvement_trend.return_value = 0.25
        ds = DistillationSignal(backtest_runner=backtester)
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse("distilled", source="distilled_v3", rank=1),
            ],
        )
        g = ds.process_ranking(ranking)
        assert g.trend == 0.25

    def test_reduction_rate(self):
        ds = DistillationSignal()
        assert ds.reduction_rate == 0.0
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse("distilled", source="distilled_v3", rank=1),
                RankedResponse("big", source="gpt-4", rank=2),
            ],
        )
        ds.process_ranking(ranking)
        assert ds.reduction_rate == 1.0

    def test_guidance_history(self):
        ds = DistillationSignal()
        assert ds.guidance_history == []
        ranking = UserRanking(
            prompt="test",
            responses=[
                RankedResponse("distilled", source="distilled_v3", rank=1),
            ],
        )
        ds.process_ranking(ranking)
        assert len(ds.guidance_history) == 1

    def test_repr(self):
        ds = DistillationSignal()
        assert "rankings_processed=0" in repr(ds)


class TestDistillationSignalJEPA:
    def test_process_jepa_embeddings(self):
        # Use orthogonal basis vectors so scores differ meaningfully
        def encoder(text):
            emb = np.zeros(64, dtype=np.float32)
            if text.endswith("A"):
                emb[0] = 1.0
            else:
                emb[1] = 1.0
            return emb

        jepa = JEPASignalSource(encoder=encoder)
        ds = DistillationSignal(jepa_source=jepa)
        room = np.array([0.9, 0.1] + [0.0] * 62, dtype=np.float32)
        texts = ["response A", "response B"]
        g = ds.process_jepa_embeddings(texts, room, source_tags=["jepa"])
        assert g.reduce_hints is True  # best_score > 0.8
        assert g.confidence_delta == 0.1  # spread > 0.3
        assert g.distilled_rank == 1
        assert g.big_model_rank == 2
        assert g.personalization_tags == ["jepa"]

    def test_process_jepa_embeddings_low_score(self):
        # Orthogonal vectors → score = 0.0, so reduce_hints stays False
        encoder = lambda text: np.array([1.0] + [0.0] * 63, dtype=np.float32)
        jepa = JEPASignalSource(encoder=encoder)
        ds = DistillationSignal(jepa_source=jepa)
        room = np.array([0.0, 1.0] + [0.0] * 62, dtype=np.float32)
        texts = ["a"]
        g = ds.process_jepa_embeddings(texts, room)
        assert g.reduce_hints is False  # best_score = 0.0 <= 0.8
        assert g.confidence_delta == -0.05  # spread = 0.0 < 0.05

    def test_process_jepa_embeddings_no_jepa_source(self):
        ds = DistillationSignal()
        with pytest.raises(RuntimeError, match="JEPASignalSource not configured"):
            ds.process_jepa_embeddings(["a"], np.zeros(64))

    def test_process_jepa_embeddings_empty(self):
        jepa = JEPASignalSource(encoder=lambda t: np.zeros(64))
        ds = DistillationSignal(jepa_source=jepa)
        g = ds.process_jepa_embeddings([], np.zeros(64))
        assert g.distilled_rank == 999
        assert g.big_model_rank == 999

    def test_process_jepa_embeddings_history(self):
        jepa = JEPASignalSource(encoder=lambda t: np.ones(64) * 0.9)
        ds = DistillationSignal(jepa_source=jepa)
        ds.process_jepa_embeddings(["a"], np.ones(64) * 0.9)
        assert len(ds.guidance_history) == 1
        assert ds.reduction_rate == 1.0
