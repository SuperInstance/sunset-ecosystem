"""Tracks human needs in real-time.

Monitors current tasks, frustration levels, wait times, and satisfaction
signals to produce a NeedState snapshot that other agents can act on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Urgency(Enum):
    """How urgently the human needs help."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Satisfaction(Enum):
    """How satisfied the human appears to be."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    UNKNOWN = "unknown"


@dataclass
class NeedState:
    """Snapshot of the human's current need state.

    Attributes:
        task: What the human is currently working on (if known).
        urgency: How urgently they need help.
        frustration: Frustration level 0.0 (calm) to 1.0 (enraged).
        satisfaction: Detected satisfaction signal.
        wait_time_seconds: How long since the human got a useful response.
        last_updated: Unix timestamp of last state change.
    """

    task: Optional[str] = None
    urgency: Urgency = Urgency.LOW
    frustration: float = 0.0
    satisfaction: Satisfaction = Satisfaction.UNKNOWN
    wait_time_seconds: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"NeedState(task={self.task!r}, urgency={self.urgency.value}, "
            f"frustration={self.frustration:.2f}, satisfaction={self.satisfaction.value}, "
            f"wait={self.wait_time_seconds:.1f}s)"
        )

    def __all__(self) -> list[str]:
        return [
            "task", "urgency", "frustration",
            "satisfaction", "wait_time_seconds", "last_updated",
        ]


# Frustration heuristics
_SATISFACTION_WORDS_POSITIVE = frozenset({
    "thanks", "thank you", "great", "perfect", "works", "awesome",
    "done", "got it", "nice", "cool", "sweet", "exactly", "that's it",
    "fixed", "resolved", "no worries", "appreciate",
})

_SATISFACTION_WORDS_NEGATIVE = frozenset({
    "still not working", "wrong", "nope", "try again", "doesn't work",
    "broken", "error", "fail", "ugh", "come on", "seriously",
    "again?", "still broken", "not helpful", "useless",
})

_FRUSTRATION_ESCALATION_PHRASES = frozenset({
    "again", "still", "repeatedly", "over and over", "keep getting",
    "why does", "still not", "same error",
})


class NeedTracker:
    """Tracks human needs in real-time.

    Observes queries, responses, and behavioral signals to maintain a
    current NeedState. Call update() with each interaction to keep state fresh.

    Example::

        tracker = NeedTracker()
        tracker.record_query("deploy the staging server")
        tracker.record_response("deployed to staging.example.com")
        state = tracker.snapshot()
    """

    def __init__(self) -> None:
        self._task: Optional[str] = None
        self._frustration: float = 0.0
        self._satisfaction: Satisfaction = Satisfaction.UNKNOWN
        self._last_useful_response: float = time.time()
        self._query_count: int = 0
        self._repeat_query_count: int = 0
        self._last_query: Optional[str] = None
        self._abandoned_attempts: int = 0

    def record_query(self, query: str) -> None:
        """Record a new query from the human.

        Detects repeated queries (frustration signal) and extracts
        satisfaction/frustration cues from the text.

        Args:
            query: The raw query text from the human.
        """
        self._query_count += 1
        query_lower = query.lower().strip()

        # Detect repeated queries
        if self._last_query and self._is_similar(query_lower, self._last_query):
            self._repeat_query_count += 1
            self._frustration = min(1.0, self._frustration + 0.15)
        self._last_query = query_lower

        # Detect frustration escalation
        for phrase in _FRUSTRATION_ESCALATION_PHRASES:
            if phrase in query_lower:
                self._frustration = min(1.0, self._frustration + 0.1)
                break

        # Detect satisfaction signals
        for word in _SATISFACTION_WORDS_NEGATIVE:
            if word in query_lower:
                self._frustration = min(1.0, self._frustration + 0.2)
                self._satisfaction = Satisfaction.NEGATIVE
                break
        else:
            for word in _SATISFACTION_WORDS_POSITIVE:
                if word in query_lower:
                    self._satisfaction = Satisfaction.POSITIVE
                    self._frustration = max(0.0, self._frustration - 0.2)
                    break

        # Extract task from query (simple heuristic: use the query itself)
        if not self._task or self._frustration < 0.3:
            self._task = query.strip()

        # Track as potential abandoned attempt if satisfaction is negative
        if self._satisfaction == Satisfaction.NEGATIVE:
            self._abandoned_attempts += 1

    def record_response(self, response: str) -> None:
        """Record a response given to the human.

        Resets wait time and may adjust satisfaction based on content.

        Args:
            response: The response text delivered to the human.
        """
        self._last_useful_response = time.time()
        response_lower = response.lower().strip()

        # Check if response signals completion
        completion_words = {"done", "complete", "finished", "deployed", "fixed", "resolved"}
        for word in completion_words:
            if word in response_lower:
                self._satisfaction = Satisfaction.POSITIVE
                self._frustration = max(0.0, self._frustration - 0.1)
                break

    def record_abandonment(self) -> None:
        """Record that the human abandoned the current attempt."""
        self._abandoned_attempts += 1
        self._frustration = min(1.0, self._frustration + 0.2)
        self._satisfaction = Satisfaction.NEGATIVE

    def record_satisfaction(self, signal: str) -> None:
        """Record an explicit satisfaction signal from the human.

        Args:
            signal: Text that indicates satisfaction or dissatisfaction.
        """
        signal_lower = signal.lower().strip()
        for word in _SATISFACTION_WORDS_POSITIVE:
            if word in signal_lower:
                self._satisfaction = Satisfaction.POSITIVE
                self._frustration = max(0.0, self._frustration - 0.3)
                return
        for word in _SATISFACTION_WORDS_NEGATIVE:
            if word in signal_lower:
                self._satisfaction = Satisfaction.NEGATIVE
                self._frustration = min(1.0, self._frustration + 0.2)
                return

    def snapshot(self) -> NeedState:
        """Return a snapshot of the current need state.

        Returns:
            A NeedState dataclass capturing the human's current situation.
        """
        wait = time.time() - self._last_useful_response
        urgency = self._compute_urgency()

        return NeedState(
            task=self._task,
            urgency=urgency,
            frustration=round(self._frustration, 3),
            satisfaction=self._satisfaction,
            wait_time_seconds=round(wait, 1),
            last_updated=time.time(),
        )

    def _compute_urgency(self) -> Urgency:
        """Derive urgency from frustration, wait time, and repeats."""
        wait = time.time() - self._last_useful_response

        if self._frustration >= 0.8 or wait > 300:
            return Urgency.CRITICAL
        if self._frustration >= 0.5 or wait > 120 or self._repeat_query_count >= 3:
            return Urgency.HIGH
        if self._frustration >= 0.2 or wait > 60 or self._repeat_query_count >= 1:
            return Urgency.MEDIUM
        return Urgency.LOW

    @staticmethod
    def _is_similar(a: str, b: str, threshold: float = 0.6) -> bool:
        """Simple word-overlap similarity check (no numpy needed)."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        return (overlap / union) >= threshold if union > 0 else False

    def __repr__(self) -> str:
        return (
            f"NeedTracker(queries={self._query_count}, "
            f"repeats={self._repeat_query_count}, "
            f"frustration={self._frustration:.2f})"
        )
