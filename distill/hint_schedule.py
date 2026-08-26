"""HintSchedule — Controls progressive hint removal over generations."""

from __future__ import annotations

__all__ = ["HintSchedule", "ExponentialBackoffSchedule"]

from abc import ABC, abstractmethod
from dataclasses import dataclass


class HintSchedule(ABC):
    """Base class for hint removal schedules.

    Starts at max_hints and reduces over generations based on
    user ranking feedback.
    """

    @abstractmethod
    def current_level(self) -> int:
        """Get the current hint level."""
        ...

    @abstractmethod
    def reduce(self) -> int:
        """Reduce hint level by one step. Returns new level."""
        ...

    @abstractmethod
    def should_reduce(self, distilled_won: bool) -> bool:
        """Whether to reduce hints based on latest ranking."""
        ...

    @abstractmethod
    def is_autonomous(self) -> bool:
        """Whether the system has reached full autonomy (hint_level=0)."""
        ...


class ExponentialBackoffSchedule(HintSchedule):
    """Fast initial reduction, slow near zero.

    First reductions remove many hints at once. Later reductions
    are smaller, giving the system time to stabilize.

    Args:
        max_hints: Starting hint level (default 10).
        min_level: Minimum level before declaring autonomous (default 0).
        reduction_rate: Base for exponential decay (default 0.5).
            Each reduction: level -= max(1, int(level * reduction_rate))
    """

    def __init__(
        self,
        max_hints: int = 10,
        min_level: int = 0,
        reduction_rate: float = 0.5,
    ) -> None:
        self._level = max_hints
        self._max_hints = max_hints
        self._min_level = min_level
        self._reduction_rate = reduction_rate
        self._consecutive_wins = 0

    def __repr__(self) -> str:
        pct = (
            (1.0 - self._level / self._max_hints) * 100 if self._max_hints > 0 else 0.0
        )
        return (
            f"ExponentialBackoffSchedule(level={self._level}/{self._max_hints}, "
            f"progress={pct:.0f}%)"
        )

    def current_level(self) -> int:
        return self._level

    def reduce(self) -> int:
        """Reduce hint level exponentially."""
        reduction = max(1, int(self._level * self._reduction_rate))
        self._level = max(self._min_level, self._level - reduction)
        return self._level

    def should_reduce(self, distilled_won: bool) -> bool:
        """Reduce if distilled beat big model.

        Tracks consecutive wins — require fewer wins for early reductions,
        more wins for later (closer to autonomy) reductions.
        """
        if distilled_won:
            self._consecutive_wins += 1
        else:
            self._consecutive_wins = 0

        # Threshold increases as we get closer to autonomy
        threshold = 1 + (self._max_hints - self._level)
        return self._consecutive_wins >= threshold

    def is_autonomous(self) -> bool:
        return self._level <= self._min_level
