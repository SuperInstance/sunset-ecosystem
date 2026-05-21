"""UserRanking — Collects rankings for a single prompt."""

from __future__ import annotations

__all__ = ["UserRanking"]

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .ranked_response import RankedResponse


# Common preference tags extracted from user notes
_PREFERENCE_PATTERNS: dict[str, list[str]] = {
    "concise": ["concise", "brief", "short", "terse", "to the point"],
    "thorough": ["thorough", "detailed", "comprehensive", "complete", "exhaustive"],
    "code_examples": ["code example", "show me code", "code snippet", "implementation"],
    "too_verbose": ["too long", "verbose", "wordy", "rambling", "too much"],
    "correct": ["correct", "accurate", "right", "spot on", "exactly"],
    "incorrect": ["wrong", "incorrect", "mistake", "error", "bad"],
    "helpful": ["helpful", "useful", "practical", "actionable"],
    "too_abstract": ["abstract", "theoretical", "vague", "unclear"],
    "fast": ["fast", "quick", "responsive", "immediate"],
    "slow": ["slow", "took too long", "laggy"],
}


@dataclass
class UserRanking:
    """A user's ranking of multiple responses to a single prompt.

    The user sees 2-4 options, ranks them, and explains why.
    The "why" is the distillation signal.

    Attributes:
        prompt: The original prompt.
        responses: The ranked responses.
        user_notes: Free text explaining the ranking.
        timestamp: When this ranking was collected.
    """
    prompt: str
    responses: list[RankedResponse] = field(default_factory=list)
    user_notes: str = ""
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"UserRanking(prompt={self.prompt[:30]!r}..., "
            f"responses={len(self.responses)}, notes={len(self.user_notes)} chars)"
        )

    def best_response(self) -> Optional[RankedResponse]:
        """Get the best-ranked response (rank=1)."""
        ranked = [r for r in self.responses if r.rank > 0]
        if not ranked:
            return None
        return min(ranked, key=lambda r: r.rank)

    def worst_response(self) -> Optional[RankedResponse]:
        """Get the worst-ranked response."""
        ranked = [r for r in self.responses if r.rank > 0]
        if not ranked:
            return None
        return max(ranked, key=lambda r: r.rank)

    def distilled_beats_big_model(self) -> bool:
        """Whether the best distilled response beat the best big model response."""
        best_distilled = min(
            (r.rank for r in self.responses if r.is_distilled and r.rank > 0),
            default=999,
        )
        best_big = min(
            (r.rank for r in self.responses if r.is_big_model and r.rank > 0),
            default=999,
        )
        return best_distilled < best_big

    @property
    def preference_tags(self) -> list[str]:
        """Extract preference tags from user notes."""
        notes_lower = self.user_notes.lower()
        tags: list[str] = []
        for tag, patterns in _PREFERENCE_PATTERNS.items():
            if any(p in notes_lower for p in patterns):
                tags.append(tag)
        return tags
