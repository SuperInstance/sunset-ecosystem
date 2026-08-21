"""DistillationSignal — Converts user rankings into guidance."""

from __future__ import annotations

__all__ = ["DistillationSignal", "DistillationGuidance", "JEPASignalSource"]

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ranking.user_ranking import UserRanking
from .backtest_runner import BacktestRunner


@dataclass
class DistillationGuidance:
    """Guidance derived from a user ranking.

    Attributes:
        reduce_hints: Whether to reduce the hint level.
        personalization_tags: Preference tags extracted from user notes.
        confidence_delta: Change in distillation confidence (+/-).
        distilled_rank: Best rank of distilled responses.
        big_model_rank: Best rank of big model responses.
        trend: Recent improvement trend from backtesting.
    """

    reduce_hints: bool = False
    personalization_tags: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    distilled_rank: int = 999
    big_model_rank: int = 999
    trend: float = 0.0

    def __repr__(self) -> str:
        return (
            f"DistillationGuidance(reduce={self.reduce_hints}, "
            f"tags={self.personalization_tags}, "
            f"Δconf={self.confidence_delta:+.2f})"
        )


class JEPASignalSource:
    """Computes JEPA latent embeddings for ranking responses.

    Accepts any encoder callable (text → np.ndarray).  The cosine
    similarity between a response embedding and the room's predicted
    latent becomes the ranking score.
    """

    def __init__(self, encoder: Callable[[str], np.ndarray] | None = None) -> None:
        self.encoder = encoder

    def _encode(self, text: str) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("JEPASignalSource has no encoder")
        emb = self.encoder(text)
        if not isinstance(emb, np.ndarray):
            raise TypeError(f"encoder must return np.ndarray, got {type(emb)}")
        return emb.astype(np.float32)

    def score_response(self, response_text: str, room_latent: np.ndarray) -> float:
        """Cosine similarity between response embedding and room latent.

        Returns a float in [-1, 1] where higher = better alignment.
        """
        emb = self._encode(response_text)
        room = np.asarray(room_latent, dtype=np.float32)
        if emb.shape != room.shape:
            raise ValueError(
                f"embedding shape {emb.shape} does not match room latent shape {room.shape}"
            )
        dot = float(np.dot(emb, room))
        norm_emb = float(np.linalg.norm(emb))
        norm_room = float(np.linalg.norm(room))
        if norm_emb == 0 or norm_room == 0:
            return 0.0
        return dot / (norm_emb * norm_room)

    def rank_responses(
        self,
        response_texts: list[str],
        room_latent: np.ndarray,
    ) -> list[tuple[int, float]]:
        """Rank responses by JEPA cosine similarity.

        Returns list of (index, score) sorted by descending score.
        """
        scores = [
            (i, self.score_response(text, room_latent))
            for i, text in enumerate(response_texts)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class DistillationSignal:
    """Converts user rankings into distillation guidance.

    Tracks whether distilled responses are improving relative to
    the big model over time.

    Args:
        backtest_runner: Optional backtest runner for trend data.
        jepa_source: Optional JEPA signal source for latent-based ranking.
    """

    def __init__(
        self,
        backtest_runner: BacktestRunner | None = None,
        jepa_source: JEPASignalSource | None = None,
    ) -> None:
        self._backtester = backtest_runner
        self._jepa_source = jepa_source
        self._history: list[DistillationGuidance] = []

    def __repr__(self) -> str:
        return f"DistillationSignal(rankings_processed={len(self._history)})"

    def process_ranking(self, ranking: UserRanking) -> DistillationGuidance:
        """Process a user ranking into distillation guidance.

        Args:
            ranking: The user's ranking of multiple responses.

        Returns:
            DistillationGuidance with actionable decisions.
        """
        # Get best ranks
        distilled_ranks = [
            r.rank for r in ranking.responses if r.is_distilled and r.rank > 0
        ]
        big_ranks = [r.rank for r in ranking.responses if r.is_big_model and r.rank > 0]

        best_distilled = min(distilled_ranks) if distilled_ranks else 999
        best_big = min(big_ranks) if big_ranks else 999

        # Should we reduce hints?
        reduce = best_distilled < best_big

        # Confidence delta
        if reduce:
            delta = 0.1
        elif best_distilled > best_big:
            delta = -0.05
        else:
            delta = 0.0

        # Get backtest trend
        trend = 0.0
        if self._backtester:
            trend = self._backtester.improvement_trend()

        guidance = DistillationGuidance(
            reduce_hints=reduce,
            personalization_tags=ranking.preference_tags,
            confidence_delta=delta,
            distilled_rank=best_distilled,
            big_model_rank=best_big,
            trend=trend,
        )

        self._history.append(guidance)
        return guidance

    def process_jepa_embeddings(
        self,
        response_texts: list[str],
        room_latent: np.ndarray,
        *,
        source_tags: list[str] | None = None,
    ) -> DistillationGuidance:
        """Rank responses by JEPA latent similarity and produce guidance.

        Args:
            response_texts: List of response strings to rank.
            room_latent: The room's predicted JEPA latent vector.
            source_tags: Optional metadata tags for provenance.

        Returns:
            DistillationGuidance where ``distilled_rank`` and
            ``big_model_rank`` are replaced by JEPA similarity ranks.
        """
        if self._jepa_source is None:
            raise RuntimeError("JEPASignalSource not configured")

        ranked = self._jepa_source.rank_responses(response_texts, room_latent)
        if not ranked:
            return DistillationGuidance(
                reduce_hints=False,
                personalization_tags=list(source_tags or []),
                confidence_delta=0.0,
                distilled_rank=999,
                big_model_rank=999,
                trend=0.0,
            )

        best_score = ranked[0][1]
        worst_score = ranked[-1][1]

        # Confidence delta based on score spread
        spread = best_score - worst_score
        if spread > 0.3:
            delta = 0.1
        elif spread < 0.05:
            delta = -0.05
        else:
            delta = 0.0

        # Reduce hints when best score is high (> 0.8)
        reduce = best_score > 0.8

        # Backtest trend
        trend = 0.0
        if self._backtester:
            trend = self._backtester.improvement_trend()

        guidance = DistillationGuidance(
            reduce_hints=reduce,
            personalization_tags=list(source_tags or []),
            confidence_delta=delta,
            distilled_rank=1,  # JEPA best
            big_model_rank=2,  # JEPA second best (or 999 if only 1)
            trend=trend,
        )

        self._history.append(guidance)
        return guidance

    @property
    def guidance_history(self) -> list[DistillationGuidance]:
        """All guidance decisions made."""
        return list(self._history)

    @property
    def reduction_rate(self) -> float:
        """Fraction of rankings that resulted in hint reduction."""
        if not self._history:
            return 0.0
        return sum(1 for g in self._history if g.reduce_hints) / len(self._history)
