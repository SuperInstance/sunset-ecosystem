"""PromptHistory — Stores prompt→response pairs with metadata."""

from __future__ import annotations

__all__ = ["PromptHistory", "PromptRecord"]

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PromptRecord:
    """A single prompt→response pair with generation metadata.

    Attributes:
        prompt: The input prompt text.
        response: The generated response text.
        seed: Random seed used.
        temperature: Temperature setting.
        model: Model name that generated this response.
        hint_level: How many hints from the big model (0 = autonomous).
        quality_score: Backtest quality score (0-1, -1 if not tested).
        application: Which application this prompt belongs to.
        timestamp: When this record was created.
    """
    prompt: str
    response: str
    seed: int = 42
    temperature: float = 0.7
    model: str = "unknown"
    hint_level: int = 10
    quality_score: float = -1.0
    application: str = "default"
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"PromptRecord(model={self.model!r}, hints={self.hint_level}, "
            f"quality={self.quality_score:.2f}, app={self.application!r})"
        )


class PromptHistory:
    """Thread-safe store of prompt→response pairs.

    Supports querying by model, hint level, application, and date range.

    Args:
        max_records: Maximum records to keep (oldest evicted first).
    """

    def __init__(self, max_records: int = 10000) -> None:
        self._max_records = max_records
        self._records: list[PromptRecord] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"PromptHistory(records={len(self._records)}, max={self._max_records})"

    def add(self, record: PromptRecord) -> None:
        """Add a prompt record."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

    def query(
        self,
        model: Optional[str] = None,
        hint_level: Optional[int] = None,
        application: Optional[str] = None,
        min_quality: float = -1.0,
        limit: int = 100,
    ) -> list[PromptRecord]:
        """Query records by filters."""
        with self._lock:
            results = self._records

        if model is not None:
            results = [r for r in results if r.model == model]
        if hint_level is not None:
            results = [r for r in results if r.hint_level == hint_level]
        if application is not None:
            results = [r for r in results if r.application == application]
        if min_quality >= 0:
            results = [r for r in results if r.quality_score >= min_quality]

        return results[-limit:]

    def random(self, application: Optional[str] = None) -> Optional[PromptRecord]:
        """Get a random record (for backtesting)."""
        import random

        with self._lock:
            pool = self._records

        if application is not None:
            pool = [r for r in pool if r.application == application]

        if not pool:
            return None
        return random.choice(pool)

    def count(self) -> int:
        """Total number of records."""
        return len(self._records)

    def average_quality(self, application: Optional[str] = None) -> float:
        """Average quality score across tested records."""
        with self._lock:
            pool = self._records

        if application is not None:
            pool = [r for r in pool if r.application == application]

        scored = [r for r in pool if r.quality_score >= 0]
        if not scored:
            return 0.0
        return sum(r.quality_score for r in scored) / len(scored)
