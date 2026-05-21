"""The three documents every sunset agent writes before going dark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Epilogue:
    """Final reflection — why this agent's journey ended."""

    agent_id: str
    what_i_tried: str = ""
    what_i_found: str = ""
    why_not_relevant: str = ""
    peak_trinity_score: float = 0.0
    generation: int = 0

    def __repr__(self) -> str:
        return (
            f"Epilogue(agent={self.agent_id!r}, gen={self.generation}, "
            f"peak={self.peak_trinity_score:.4f})"
        )


@dataclass
class Summary:
    """Subjective work log — not objective, personal."""

    agent_id: str
    work_from_my_perspective: str = ""
    key_insights: List[str] = field(default_factory=list)
    failed_approaches: List[str] = field(default_factory=list)
    connections_made: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Summary(agent={self.agent_id!r}, "
            f"insights={len(self.key_insights)}, "
            f"failed={len(self.failed_approaches)})"
        )


@dataclass
class Onboarding:
    """Letter to the next generation — written knowing it's being put away."""

    agent_id: str
    letter_to_children: str = ""
    what_works: str = ""
    what_doesnt: str = ""
    where_to_look: str = ""
    variant: str = "continuation"  # continuation | cross-pollination | mutation
    parent_id: Optional[str] = None
    generation: int = 0

    def __repr__(self) -> str:
        return (
            f"Onboarding(agent={self.agent_id!r}, variant={self.variant!r}, "
            f"gen={self.generation})"
        )


__all__ = ["Epilogue", "Summary", "Onboarding"]
