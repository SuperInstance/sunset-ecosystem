"""DistillationSignal — Converts user rankings into guidance."""

from __future__ import annotations

__all__ = ["DistillationSignal", "DistillationGuidance"]

from dataclasses import dataclass, field
from typing import Any

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


class DistillationSignal:
    """Converts user rankings into distillation guidance.

    Tracks whether distilled responses are improving relative to
    the big model over time.

    Args:
        backtest_runner: Optional backtest runner for trend data.
    """

    def __init__(self, backtest_runner: BacktestRunner | None = None) -> None:
        self._backtester = backtest_runner
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
            r.rank for r in ranking.responses
            if r.is_distilled and r.rank > 0
        ]
        big_ranks = [
            r.rank for r in ranking.responses
            if r.is_big_model and r.rank > 0
        ]

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
