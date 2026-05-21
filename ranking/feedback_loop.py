"""FeedbackLoop — Connects user rankings back to all subsystems.

When a user ranks responses, this module:
1. Feeds preference tags into PersonalizationStore
2. Checks if distilled beat big model → feeds into HintSchedule
3. Generates DistillationGuidance via DistillationSignal
"""

from __future__ import annotations

__all__ = ["FeedbackLoop"]

from dataclasses import dataclass
from typing import Any

from ranking.user_ranking import UserRanking
from ranking.personalization import PersonalizationStore
from distill.hint_schedule import HintSchedule
from distill.distillation_signal import DistillationSignal, DistillationGuidance


class FeedbackLoop:
    """Connects user rankings to all subsystems.

    Args:
        hint_schedule: The hint removal schedule to update.
        personalization: The preference store to update.
        signal: The distillation signal processor.
    """

    def __init__(
        self,
        hint_schedule: HintSchedule,
        personalization: PersonalizationStore,
        signal: DistillationSignal,
    ) -> None:
        self._schedule = hint_schedule
        self._personalization = personalization
        self._signal = signal
        self._total_rankings: int = 0

    def __repr__(self) -> str:
        return (
            f"FeedbackLoop(rankings={self._total_rankings}, "
            f"hint_level={self._schedule.current_level()})"
        )

    def ingest(self, ranking: UserRanking) -> DistillationGuidance:
        """Process a user ranking through all subsystems.

        1. Update personalization preferences
        2. Generate distillation signal
        3. If signal says reduce → update hint schedule

        Args:
            ranking: The user's ranking.

        Returns:
            The generated distillation guidance.
        """
        self._total_rankings += 1

        # 1. Update personalization
        self._personalization.ingest(ranking)

        # 2. Generate signal
        guidance = self._signal.process_ranking(ranking)

        # 3. Update hint schedule if appropriate
        if guidance.reduce_hints and self._schedule.should_reduce(True):
            self._schedule.reduce()
        elif not guidance.reduce_hints:
            self._schedule.should_reduce(False)

        return guidance

    def get_status(self) -> dict[str, Any]:
        """Get current status of all subsystems."""
        return {
            "total_rankings": self._total_rankings,
            "hint_level": self._schedule.current_level(),
            "is_autonomous": self._schedule.is_autonomous(),
            "ground_truths": self._personalization.get_ground_truths(),
            "reduction_rate": self._signal.reduction_rate,
        }

    @property
    def total_rankings(self) -> int:
        """Number of rankings processed."""
        return self._total_rankings
