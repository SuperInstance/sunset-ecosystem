"""Logs all human interactions for pattern detection and scoring.

Records queries, responses, follow-ups, and time-to-solution to surface
recurring needs, chronic frustrations, and agent effectiveness.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InteractionRecord:
    """A single human interaction.

    Attributes:
        query: What the human asked.
        response: What response was given (if any).
        timestamp: Unix timestamp of the query.
        response_timestamp: When the response was delivered.
        time_to_solution: Seconds from query to satisfactory resolution.
        needed_follow_up: Whether the human had to ask again.
        resolved: Whether the interaction reached a satisfying conclusion.
        tags: Arbitrary tags for categorization.
    """

    query: str
    response: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    response_timestamp: Optional[float] = None
    time_to_solution: Optional[float] = None
    needed_follow_up: bool = False
    resolved: bool = False
    tags: list[str] = field(default_factory=list)

    @property
    def latency(self) -> Optional[float]:
        """Time between query and response in seconds."""
        if self.response_timestamp and self.timestamp:
            return self.response_timestamp - self.timestamp
        return None

    def __repr__(self) -> str:
        truncated = self.query[:40] + "..." if len(self.query) > 40 else self.query
        return (
            f"InteractionRecord(query={truncated!r}, resolved={self.resolved}, "
            f"follow_up={self.needed_follow_up})"
        )


@dataclass
class InteractionSummary:
    """Aggregate statistics over interactions.

    Attributes:
        total_interactions: Total number of recorded interactions.
        resolved_count: How many were resolved satisfactorily.
        avg_time_to_solution: Mean time-to-solution in seconds.
        follow_up_rate: Fraction of interactions that needed follow-up.
        recurring_needs: Topics that appear 2+ times.
        chronic_frustrations: Topics that repeatedly go unresolved.
        top_tags: Most common tags and their counts.
    """

    total_interactions: int = 0
    resolved_count: int = 0
    avg_time_to_solution: float = 0.0
    follow_up_rate: float = 0.0
    recurring_needs: list[str] = field(default_factory=list)
    chronic_frustrations: list[str] = field(default_factory=list)
    top_tags: list[tuple[str, int]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"InteractionSummary(total={self.total_interactions}, "
            f"resolved={self.resolved_count}, "
            f"follow_up_rate={self.follow_up_rate:.1%})"
        )


class InteractionLog:
    """Logs all human interactions and detects patterns.

    Maintains a chronological record of every query/response pair and
    surfaces recurring needs, chronic frustrations, and effectiveness metrics.

    Example::

        log = InteractionLog()
        rec = log.start_interaction("how do I deploy?")
        log.record_response(rec, "use the deploy.sh script")
        log.resolve(rec)
        summary = log.summarize()
    """

    def __init__(self) -> None:
        self._records: list[InteractionRecord] = []
        self._open_records: dict[int, InteractionRecord] = {}
        self._keyword_index: Counter[str] = Counter()

    def start_interaction(self, query: str, tags: Optional[list[str]] = None) -> InteractionRecord:
        """Start recording a new interaction.

        Args:
            query: The human's query.
            tags: Optional categorization tags.

        Returns:
            The new InteractionRecord (not yet resolved).
        """
        record = InteractionRecord(
            query=query,
            tags=tags or [],
        )
        self._records.append(record)
        self._open_records[id(record)] = record

        # Index keywords for pattern detection
        for word in query.lower().split():
            if len(word) > 3:  # skip short words
                self._keyword_index[word] += 1

        return record

    def record_response(self, record: InteractionRecord, response: str) -> None:
        """Record the response given for an interaction.

        Args:
            record: The interaction to update.
            response: The response text.
        """
        record.response = response
        record.response_timestamp = time.time()

    def resolve(self, record: InteractionRecord, time_to_solution: Optional[float] = None) -> None:
        """Mark an interaction as resolved.

        Args:
            record: The interaction to resolve.
            time_to_solution: Explicit time-to-solution in seconds.
        """
        record.resolved = True
        if time_to_solution is not None:
            record.time_to_solution = time_to_solution
        elif record.response_timestamp and record.timestamp:
            record.time_to_solution = record.response_timestamp - record.timestamp
        self._open_records.pop(id(record), None)

    def mark_follow_up(self, record: InteractionRecord) -> None:
        """Mark that this interaction required a follow-up.

        Args:
            record: The interaction that needed follow-up.
        """
        record.needed_follow_up = True

    @property
    def records(self) -> list[InteractionRecord]:
        """All recorded interactions (read-only copy)."""
        return list(self._records)

    def summarize(self) -> InteractionSummary:
        """Compute aggregate statistics over all interactions.

        Returns:
            An InteractionSummary with patterns and metrics.
        """
        if not self._records:
            return InteractionSummary()

        resolved = [r for r in self._records if r.resolved]
        follow_ups = [r for r in self._records if r.needed_follow_up]
        unresolved = [r for r in self._records if not r.resolved]

        # Average time to solution
        solution_times = [
            r.time_to_solution for r in resolved
            if r.time_to_solution is not None
        ]
        avg_tts = (
            sum(solution_times) / len(solution_times)
            if solution_times else 0.0
        )

        # Recurring needs: keywords appearing 2+ times
        recurring = [
            word for word, count in self._keyword_index.most_common(20)
            if count >= 2
        ]

        # Chronic frustrations: keywords common in unresolved interactions
        unresolved_counter: Counter[str] = Counter()
        for r in unresolved:
            for word in r.query.lower().split():
                if len(word) > 3:
                    unresolved_counter[word] += 1
        chronic = [
            word for word, count in unresolved_counter.most_common(10)
            if count >= 2
        ]

        # Top tags
        tag_counter: Counter[str] = Counter()
        for r in self._records:
            for tag in r.tags:
                tag_counter[tag] += 1

        return InteractionSummary(
            total_interactions=len(self._records),
            resolved_count=len(resolved),
            avg_time_to_solution=round(avg_tts, 2),
            follow_up_rate=round(len(follow_ups) / len(self._records), 3),
            recurring_needs=recurring,
            chronic_frustrations=chronic,
            top_tags=tag_counter.most_common(10),
        )

    def detect_patterns(self) -> dict[str, list[str]]:
        """Detect recurring interaction patterns.

        Returns:
            A dict with keys 'recurring_needs', 'chronic_frustrations',
            'slow_resolutions', and 'values' as lists of descriptions.
        """
        summary = self.summarize()
        slow = [
            f"{r.query[:50]} ({r.time_to_solution:.0f}s)"
            for r in self._records
            if r.time_to_solution is not None and r.time_to_solution > 120
        ]
        return {
            "recurring_needs": summary.recurring_needs,
            "chronic_frustrations": summary.chronic_frustrations,
            "slow_resolutions": slow,
        }

    def __repr__(self) -> str:
        return (
            f"InteractionLog(records={len(self._records)}, "
            f"open={len(self._open_records)})"
        )

    def __len__(self) -> int:
        return len(self._records)
