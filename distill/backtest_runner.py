"""BacktestRunner — Runs old prompts on spare capacity."""

from __future__ import annotations

__all__ = ["BacktestRunner", "BacktestResult"]

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .prompt_history import PromptHistory, PromptRecord


@dataclass
class BacktestResult:
    """Result of a single backtest run.

    Attributes:
        prompt: The prompt that was tested.
        reference_response: The original big-model response.
        distilled_response: The swarm's response.
        similarity: How similar distilled is to reference (0-1).
        hint_level: Hint level used for this backtest.
        routes_used: Which routes fired during processing.
        latency_ms: Time taken.
        improved: Whether this was better than the previous best.
    """
    prompt: str
    reference_response: str
    distilled_response: str
    similarity: float = 0.0
    hint_level: int = 0
    routes_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    improved: bool = False

    def __repr__(self) -> str:
        return (
            f"BacktestResult(sim={self.similarity:.2f}, "
            f"hints={self.hint_level}, improved={self.improved})"
        )


class BacktestRunner:
    """Runs old prompts through the swarm on spare capacity.

    Picks random prompts from history, re-runs them at current hint level,
    and compares with the reference response.

    Args:
        history: The prompt history to backtest against.
        spare_threshold: Minimum spare capacity (0-1) to run a backtest.
    """

    def __init__(
        self,
        history: PromptHistory,
        spare_threshold: float = 0.3,
    ) -> None:
        self._history = history
        self._spare_threshold = spare_threshold
        self._results: list[BacktestResult] = []
        self._backtests_run: int = 0

    def __repr__(self) -> str:
        return (
            f"BacktestRunner(run={self._backtests_run}, "
            f"results={len(self._results)}, "
            f"threshold={self._spare_threshold})"
        )

    def run_cycle(
        self,
        hint_level: int = 0,
        spare_capacity: float = 0.5,
        application: Optional[str] = None,
    ) -> Optional[BacktestResult]:
        """Run one backtest cycle if there's spare capacity.

        Args:
            hint_level: Current hint level to test at.
            spare_capacity: Current system spare capacity (0-1).
            application: Optional app filter.

        Returns:
            BacktestResult if a test was run, None if skipped.
        """
        if spare_capacity < self._spare_threshold:
            return None

        record = self._history.random(application=application)
        if record is None:
            return None

        start = time.time()
        # Simulate swarm processing at current hint level
        distilled = self._simulate_process(record.prompt, hint_level)
        latency = (time.time() - start) * 1000

        similarity = self._compute_similarity(distilled, record.response)

        result = BacktestResult(
            prompt=record.prompt,
            reference_response=record.response,
            distilled_response=distilled,
            similarity=similarity,
            hint_level=hint_level,
            latency_ms=latency,
            improved=similarity > record.quality_score if record.quality_score >= 0 else True,
        )

        self._results.append(result)
        self._backtests_run += 1

        # Update the record's quality score
        if result.improved:
            record.quality_score = similarity

        return result

    def _simulate_process(self, prompt: str, hint_level: int) -> str:
        """Simulate swarm processing. In production, this would run the actual swarm."""
        # Simple simulation: higher hints = more verbose (closer to reference)
        words = prompt.split()
        length = len(words) + hint_level * 2
        return " ".join(words[:length]) if length <= len(words) else prompt + " processed"

    @staticmethod
    def _compute_similarity(a: str, b: str) -> float:
        """Simple word-overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @property
    def results(self) -> list[BacktestResult]:
        """All backtest results."""
        return list(self._results)

    @property
    def backtests_run(self) -> int:
        """Total backtests run."""
        return self._backtests_run

    def improvement_trend(self, last_n: int = 10) -> float:
        """Trend of recent results: positive = improving, negative = degrading."""
        recent = self._results[-last_n:]
        if len(recent) < 2:
            return 0.0
        first_half = sum(r.similarity for r in recent[: len(recent) // 2])
        second_half = sum(r.similarity for r in recent[len(recent) // 2 :])
        n1 = max(1, len(recent) // 2)
        n2 = max(1, len(recent) - len(recent) // 2)
        return (second_half / n2) - (first_half / n1)
