"""BaselineConfidenceGate — Prevents the overknowledge problem.

When Wesley's baseline score is already high, teaching HURTS.
This gate decides whether to teach, skip, or teach with reduced
intensity based on the baseline confidence score.

Thresholds:
    > 0.85  → SKIP teaching entirely (baseline confident)
    < 0.60  → Teach normally (full intensity)
    0.6–0.85 → Teach with reduced intensity (single pass, 50% hints)
"""

from __future__ import annotations

__all__ = ["BaselineConfidenceGate", "GateDecision"]

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("distill.baseline_gate")


class GateAction(Enum):
    """What the gate decided to do."""
    SKIP = "skip"
    FULL = "full"
    REDUCED = "reduced"


@dataclass
class GateDecision:
    """The result of a baseline confidence gate check.

    Attributes:
        action: What to do (SKIP, FULL, or REDUCED).
        baseline_score: The score that was checked.
        max_passes: How many teaching passes to run (0 for skip).
        hint_level_pct: Hint level as percentage of max (0–100).
        allow_repetition: Whether repetition passes are allowed.
        reason: Human-readable explanation.
    """
    action: GateAction
    baseline_score: float
    max_passes: int
    hint_level_pct: float
    allow_repetition: bool
    reason: str

    def __repr__(self) -> str:
        return (
            f"GateDecision(action={self.action.value}, "
            f"score={self.baseline_score:.3f}, "
            f"passes={self.max_passes}, "
            f"hints={self.hint_level_pct:.0f}%)"
        )


class BaselineConfidenceGate:
    """Decides teaching intensity based on baseline confidence.

    The overknowledge problem: when a student model already performs
    well on a topic, continued teaching on that topic degrades
    performance. This gate prevents that by checking the baseline
    score before any teaching begins.

    Args:
        skip_threshold: Scores above this → SKIP entirely. Default 0.85.
        full_threshold: Scores below this → FULL intensity. Default 0.60.
        reduced_hint_pct: Hint percentage for reduced mode. Default 50.0.
        reduced_max_passes: Passes in reduced mode. Default 1.
        full_max_passes: Passes in full mode. Default 3.
    """

    def __init__(
        self,
        skip_threshold: float = 0.85,
        full_threshold: float = 0.60,
        reduced_hint_pct: float = 50.0,
        reduced_max_passes: int = 1,
        full_max_passes: int = 3,
    ) -> None:
        if skip_threshold <= full_threshold:
            raise ValueError(
                f"skip_threshold ({skip_threshold}) must be greater than "
                f"full_threshold ({full_threshold})"
            )
        self.skip_threshold = skip_threshold
        self.full_threshold = full_threshold
        self.reduced_hint_pct = reduced_hint_pct
        self.reduced_max_passes = reduced_max_passes
        self.full_max_passes = full_max_passes
        self._history: list[GateDecision] = []

    def __repr__(self) -> str:
        return (
            f"BaselineConfidenceGate("
            f"skip>{self.skip_threshold}, "
            f"full<{self.full_threshold})"
        )

    def evaluate(self, baseline_score: float, topic: Optional[str] = None) -> GateDecision:
        """Evaluate whether teaching should proceed.

        Args:
            baseline_score: The baseline confidence score (0.0–1.0).
            topic: Optional topic label for logging.

        Returns:
            GateDecision with the action and parameters.
        """
        topic_str = topic or "unnamed"

        if baseline_score > self.skip_threshold:
            decision = GateDecision(
                action=GateAction.SKIP,
                baseline_score=baseline_score,
                max_passes=0,
                hint_level_pct=0.0,
                allow_repetition=False,
                reason=(
                    f"baseline_confident: skip (score={baseline_score:.3f} "
                    f"> {self.skip_threshold})"
                ),
            )
            logger.info(
                "baseline_confident: skip | topic=%s score=%.3f threshold=%.2f",
                topic_str, baseline_score, self.skip_threshold,
            )

        elif baseline_score < self.full_threshold:
            decision = GateDecision(
                action=GateAction.FULL,
                baseline_score=baseline_score,
                max_passes=self.full_max_passes,
                hint_level_pct=100.0,
                allow_repetition=True,
                reason=(
                    f"baseline_low: full intensity (score={baseline_score:.3f} "
                    f"< {self.full_threshold})"
                ),
            )
            logger.info(
                "baseline_low: full intensity | topic=%s score=%.3f passes=%d",
                topic_str, baseline_score, self.full_max_passes,
            )

        else:
            decision = GateDecision(
                action=GateAction.REDUCED,
                baseline_score=baseline_score,
                max_passes=self.reduced_max_passes,
                hint_level_pct=self.reduced_hint_pct,
                allow_repetition=False,
                reason=(
                    f"baseline_moderate: reduced intensity (score={baseline_score:.3f} "
                    f"in [{self.full_threshold}, {self.skip_threshold}])"
                ),
            )
            logger.info(
                "baseline_moderate: reduced | topic=%s score=%.3f passes=%d hints=%.0f%%",
                topic_str, baseline_score, self.reduced_max_passes, self.reduced_hint_pct,
            )

        self._history.append(decision)
        return decision

    @property
    def history(self) -> list[GateDecision]:
        """All decisions made by this gate."""
        return list(self._history)

    @property
    def skip_rate(self) -> float:
        """Fraction of evaluations that resulted in SKIP."""
        if not self._history:
            return 0.0
        return sum(1 for d in self._history if d.action is GateAction.SKIP) / len(self._history)

    def reset(self) -> None:
        """Clear decision history."""
        self._history.clear()
