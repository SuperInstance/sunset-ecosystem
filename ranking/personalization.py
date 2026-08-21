"""Personalization — Extracts user preferences over time.

Every ranking teaches the system about THIS user. Preferences accumulate
as ground-truths-for-now in the pathos room.
"""

from __future__ import annotations

__all__ = ["PersonalizationStore", "PreferenceScore"]

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .user_ranking import UserRanking


@dataclass
class PreferenceScore:
    """How well a response matches learned user preferences.

    Attributes:
        score: 0.0-1.0 match against learned preferences.
        matched_tags: Which preference tags this response satisfies.
        missed_tags: Which preference tags this response misses.
    """

    score: float = 0.0
    matched_tags: list[str] = field(default_factory=list)
    missed_tags: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"PreferenceScore(score={self.score:.2f}, matched={self.matched_tags})"


class PersonalizationStore:
    """Accumulates user preferences across all rankings.

    Thread-safe. Tracks tag frequencies and trends.
    Provides ground-truths-for-now for the pathos room.

    Args:
        decay_factor: How much older preferences decay per new ranking (0-1).
    """

    def __init__(self, decay_factor: float = 0.95) -> None:
        self._decay_factor = decay_factor
        self._tag_weights: dict[str, float] = {}
        self._ranking_count: int = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        top = self.get_top_preferences(3)
        tags = ", ".join(f"{t[0]}={t[1]:.2f}" for t in top)
        return f"PersonalizationStore(rankings={self._ranking_count}, top=[{tags}])"

    def ingest(self, ranking: UserRanking) -> None:
        """Ingest a user ranking, updating preference weights.

        Tags from the ranking get reinforced. Others decay slightly.
        """
        with self._lock:
            self._ranking_count += 1
            tags = ranking.preference_tags

            # Decay all existing tags
            for tag in self._tag_weights:
                self._tag_weights[tag] *= self._decay_factor

            # Reinforce tags from this ranking
            for tag in tags:
                self._tag_weights[tag] = self._tag_weights.get(tag, 0.0) + 1.0

    def get_ground_truths(self) -> dict[str, Any]:
        """Get current user preferences as ground-truths-for-now.

        Returns dict of tag → weight for tags above threshold.
        """
        with self._lock:
            return {
                tag: round(weight, 3)
                for tag, weight in sorted(
                    self._tag_weights.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                if weight > 0.5
            }

    def get_top_preferences(self, n: int = 5) -> list[tuple[str, float]]:
        """Get the top-N preferences by weight."""
        with self._lock:
            sorted_tags = sorted(
                self._tag_weights.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_tags[:n]

    def score_response(self, tags: list[str]) -> PreferenceScore:
        """Score how well a response's tags match learned preferences.

        Args:
            tags: The response's characteristic tags.

        Returns:
            PreferenceScore with match analysis.
        """
        with self._lock:
            if not self._tag_weights:
                return PreferenceScore(score=0.5, matched_tags=[], missed_tags=[])

            total_weight = sum(self._tag_weights.values())
            if total_weight == 0:
                return PreferenceScore(score=0.5)

            matched_weight = sum(self._tag_weights.get(t, 0.0) for t in tags)
            score = matched_weight / total_weight if total_weight > 0 else 0.5

            top_tags = set(
                t
                for t, w in sorted(
                    self._tag_weights.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            )
            matched = [t for t in tags if t in top_tags]
            missed = [t for t in top_tags if t not in tags]

            return PreferenceScore(
                score=min(1.0, score),
                matched_tags=matched,
                missed_tags=missed,
            )

    @property
    def ranking_count(self) -> int:
        """Number of rankings ingested."""
        return self._ranking_count
