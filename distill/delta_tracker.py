"""Delta Tracker — Quality delta between distillation generations.

Tracks quality scores across generations to detect improvements
or regressions. Enables the system to detect when distillation
is going wrong and revert to a known-good state.
"""

from __future__ import annotations

__all__ = ["DeltaSnapshot", "DeltaTracker"]

import time
from dataclasses import dataclass, field


@dataclass
class DeltaSnapshot:
    """A single quality measurement for a distillation generation.

    Attributes:
        generation: The generation number.
        avg_quality: Average quality score for this generation (0.0-1.0).
        hint_level: Hint level used during this generation.
        timestamp: When this snapshot was recorded.
    """

    generation: int
    avg_quality: float
    hint_level: int
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"DeltaSnapshot(gen={self.generation}, "
            f"quality={self.avg_quality:.3f}, hints={self.hint_level})"
        )


class DeltaTracker:
    """Tracks quality deltas across distillation generations.

    Records quality snapshots and computes trends. Detects regressions
    (consecutive negative deltas) and triggers revert recommendations.

    Thread safety: not thread-safe by design. Caller should synchronize
    if sharing across threads.
    """

    def __init__(self) -> None:
        self._snapshots: list[DeltaSnapshot] = []

    def __repr__(self) -> str:
        n = len(self._snapshots)
        if n == 0:
            return "DeltaTracker(empty)"
        latest = self._snapshots[-1]
        return (
            f"DeltaTracker(snapshots={n}, "
            f"latest_gen={latest.generation}, "
            f"latest_quality={latest.avg_quality:.3f})"
        )

    def record(
        self, generation: int, avg_quality: float, hint_level: int
    ) -> DeltaSnapshot:
        """Record a quality snapshot for a generation.

        Args:
            generation: The generation number.
            avg_quality: Average quality score (0.0-1.0).
            hint_level: Current hint level used.

        Returns:
            The recorded DeltaSnapshot.
        """
        snapshot = DeltaSnapshot(
            generation=generation,
            avg_quality=avg_quality,
            hint_level=hint_level,
        )
        self._snapshots.append(snapshot)
        return snapshot

    @property
    def snapshots(self) -> list[DeltaSnapshot]:
        """All recorded snapshots."""
        return list(self._snapshots)

    @property
    def latest(self) -> DeltaSnapshot | None:
        """Most recent snapshot, or None if empty."""
        return self._snapshots[-1] if self._snapshots else None

    def delta(self, n_generations: int = 3) -> float:
        """Compute quality change over the last N generations.

        Positive = improvement, negative = regression.

        Args:
            n_generations: How many generations back to compare.

        Returns:
            Quality delta (current - N-generations-ago).
            Returns 0.0 if not enough data.
        """
        if len(self._snapshots) < 2:
            return 0.0

        end_idx = len(self._snapshots) - 1
        start_idx = max(0, end_idx - n_generations)

        current = self._snapshots[end_idx].avg_quality
        baseline = self._snapshots[start_idx].avg_quality

        return current - baseline

    def is_regression(self, negative_streak: int = 3) -> bool:
        """Check for N consecutive negative deltas (quality dropping).

        Args:
            negative_streak: Number of consecutive negative deltas required.

        Returns:
            True if quality has been dropping for N consecutive steps.
        """
        if len(self._snapshots) < negative_streak + 1:
            return False

        streak = 0
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1].avg_quality
            curr = self._snapshots[i].avg_quality
            if curr < prev:
                streak += 1
            else:
                streak = 0

        return streak >= negative_streak

    def should_revert(self) -> bool:
        """Recommend reverting to a previous generation.

        Revert is recommended when:
        - There's a sustained regression (3+ consecutive drops)

        Returns:
            True if the system should revert.
        """
        return self.is_regression(negative_streak=3)

    def best_generation(self) -> DeltaSnapshot | None:
        """Find the generation with the highest quality score.

        Returns:
            The best snapshot, or None if no data.
        """
        if not self._snapshots:
            return None
        return max(self._snapshots, key=lambda s: s.avg_quality)

    def trend(self, window: int = 5) -> float:
        """Compute the average per-step quality change over a window.

        Args:
            window: Number of recent steps to consider.

        Returns:
            Average delta per step. Positive = improving, negative = declining.
        """
        if len(self._snapshots) < 2:
            return 0.0

        start = max(0, len(self._snapshots) - window)
        recent = self._snapshots[start:]

        deltas = []
        for i in range(1, len(recent)):
            deltas.append(recent[i].avg_quality - recent[i - 1].avg_quality)

        return sum(deltas) / len(deltas) if deltas else 0.0
