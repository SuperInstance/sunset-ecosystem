"""Track agent generations across the sunset lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


__all__ = ["AgentGeneration", "GenerationHistory", "GenerationMemory"]


@dataclass
class AgentGeneration:
    """A record of a single agent generation."""

    agent_id: str
    name: str
    generation: int
    created_at: datetime
    sunset_at: Optional[datetime] = None
    purpose: str = ""
    achievements: List[str] = field(default_factory=list)
    sunset_reason: Optional[str] = None
    onboarding_docs: List[str] = field(default_factory=list)
    children_spawned: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    patterns_preserved: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AgentGeneration(id={self.agent_id!r}, name={self.name!r}, "
            f"gen={self.generation})"
        )

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "generation": self.generation,
            "created_at": self.created_at.isoformat(),
            "sunset_at": self.sunset_at.isoformat() if self.sunset_at else None,
            "purpose": self.purpose,
            "achievements": self.achievements,
            "sunset_reason": self.sunset_reason,
            "onboarding_docs": self.onboarding_docs,
            "children_spawned": self.children_spawned,
            "parent_id": self.parent_id,
            "patterns_preserved": self.patterns_preserved,
            "lessons_learned": self.lessons_learned,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentGeneration":
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            generation=data["generation"],
            created_at=datetime.fromisoformat(data["created_at"]),
            sunset_at=datetime.fromisoformat(data["sunset_at"])
            if data.get("sunset_at")
            else None,
            purpose=data.get("purpose", ""),
            achievements=data.get("achievements", []),
            sunset_reason=data.get("sunset_reason"),
            onboarding_docs=data.get("onboarding_docs", []),
            children_spawned=data.get("children_spawned", []),
            parent_id=data.get("parent_id"),
            patterns_preserved=data.get("patterns_preserved", []),
            lessons_learned=data.get("lessons_learned", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GenerationHistory:
    """A view of agent generation history."""

    generations: List[AgentGeneration] = field(default_factory=list)
    total_generations: int = 0
    active_agents: int = 0
    surviving_patterns: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"GenerationHistory(gens={self.total_generations}, "
            f"active={self.active_agents})"
        )


class GenerationMemory:
    """Tracks agent generations: births, sunsets, lineage, and surviving patterns."""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._agents: Dict[str, AgentGeneration] = {}
        self._store_path = Path(store_path) if store_path else None
        if self._store_path and self._store_path.exists():
            self._load()

    def __repr__(self) -> str:
        return f"GenerationMemory(agents={len(self._agents)}, store={self._store_path})"

    def register(
        self,
        agent_id: str,
        name: str,
        generation: int,
        purpose: str = "",
        parent_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        onboarding_docs: Optional[List[str]] = None,
    ) -> AgentGeneration:
        """Register a new agent generation."""
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        gen = AgentGeneration(
            agent_id=agent_id,
            name=name,
            generation=generation,
            created_at=created_at,
            purpose=purpose,
            parent_id=parent_id,
            onboarding_docs=onboarding_docs or [],
        )
        self._agents[agent_id] = gen

        # Register as child of parent
        if parent_id and parent_id in self._agents:
            parent = self._agents[parent_id]
            if agent_id not in parent.children_spawned:
                parent.children_spawned.append(agent_id)

        self._save()
        return gen

    def sunset(
        self,
        agent_id: str,
        reason: str,
        lessons: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        sunset_at: Optional[datetime] = None,
    ) -> bool:
        """Record the sunset of an agent."""
        gen = self._agents.get(agent_id)
        if gen is None:
            return False
        gen.sunset_at = sunset_at or datetime.now(timezone.utc)
        gen.sunset_reason = reason
        if lessons:
            gen.lessons_learned = lessons
        if patterns:
            gen.patterns_preserved = patterns
        self._save()
        return True

    def get(self, agent_id: str) -> Optional[AgentGeneration]:
        """Retrieve a single agent by ID."""
        return self._agents.get(agent_id)

    def get_history(self, include_sunset: bool = True) -> GenerationHistory:
        """Get full generation history."""
        gens = list(self._agents.values())
        if not include_sunset:
            gens = [g for g in gens if g.sunset_at is None]
        gens.sort(key=lambda g: g.generation)

        active = [g for g in gens if g.sunset_at is None]
        all_patterns: Dict[str, int] = {}
        for g in gens:
            for p in g.patterns_preserved:
                all_patterns[p] = all_patterns.get(p, 0) + 1

        surviving = [
            p for p, c in sorted(all_patterns.items(), key=lambda x: -x[1]) if c > 0
        ]

        return GenerationHistory(
            generations=gens,
            total_generations=len(gens),
            active_agents=len(active),
            surviving_patterns=surviving,
        )

    def get_lineage(self, agent_id: str) -> List[AgentGeneration]:
        """Get the lineage (ancestors) of an agent."""
        lineage: List[AgentGeneration] = []
        current = self._agents.get(agent_id)
        while current and current.parent_id:
            parent = self._agents.get(current.parent_id)
            if parent is None:
                break
            lineage.append(parent)
            current = parent
        lineage.reverse()
        return lineage

    def get_children(self, agent_id: str) -> List[AgentGeneration]:
        """Get direct children of an agent."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return []
        return [
            self._agents[cid] for cid in agent.children_spawned if cid in self._agents
        ]

    def _save(self) -> None:
        if self._store_path is None:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {aid: gen.to_dict() for aid, gen in self._agents.items()}
        tmp = self._store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._store_path)

    def _load(self) -> None:
        try:
            data = json.loads(self._store_path.read_text())  # type: ignore[union-attr]
            for aid, adata in data.items():
                self._agents[aid] = AgentGeneration.from_dict(adata)
        except (json.JSONDecodeError, OSError):
            pass
