"""Agent lifecycle — birth, competition, breeding, sunset."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class AgentPhase(Enum):
    """Lifecycle phases every agent passes through."""

    INCUBATING = "incubating"  # Just born, reading trinity rooms
    COMPETING = "competing"  # Finding relevance
    BREEDING = "breeding"  # Scored high, spawning children
    SUNSETTING = "sunsetting"  # Writing epilogue
    ASLEEP = "asleep"  # Archived, searchable


@dataclass
class ResourceBudget:
    """Compute/time budget allocated to an agent."""

    max_tokens: int = 4096
    max_time_seconds: float = 60.0
    parallel_slots: int = 1

    def __repr__(self) -> str:
        return (
            f"ResourceBudget(max_tokens={self.max_tokens}, "
            f"max_time_seconds={self.max_time_seconds}, "
            f"parallel_slots={self.parallel_slots})"
        )


@dataclass
class Agent:
    """A single-generation agent in the SUNSET ecosystem."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    parent_id: Optional[str] = None
    room: str = ""
    phase: AgentPhase = AgentPhase.INCUBATING
    trinity_score: float = 0.0
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.room:
            self.room = f"agent-{self.id}"

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id!r}, gen={self.generation}, "
            f"phase={self.phase.value!r}, trinity={self.trinity_score:.4f})"
        )

    def advance(self, new_phase: AgentPhase) -> None:
        """Transition to a new lifecycle phase."""
        self.phase = new_phase


__all__ = ["Agent", "AgentPhase", "ResourceBudget"]
