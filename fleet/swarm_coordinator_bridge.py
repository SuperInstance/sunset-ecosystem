"""fleet/swarm_coordinator_bridge.py — Equipment-Swarm-Coordinator pattern integration.

Brings the Equipment-Swarm-Coordinator patterns into sunset-ecosystem:
- Agent roles (coordinator, executor, validator, specialist, observer)
- Conflict resolution strategies (voting, weighted, hierarchical, consensus)
- Knowledge isolation levels (strict, moderate, relaxed)
- Task decomposition strategies (parallel, sequential, pipeline, map-reduce, divide-conquer)

Usage:
    from fleet.swarm_coordinator_bridge import SwarmCoordinator, AgentRole, TaskDecomposer

    coordinator = SwarmCoordinator(max_agents=10)
    coordinator.register_agent("scout-1", AgentRole.SCOUT, capabilities=["search", "fetch"])
    coordinator.register_agent("builder-1", AgentRole.BUILDER, capabilities=["code", "test"])
    result = coordinator.resolve_conflict(["option-a", "option-b"], strategy="weighted")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class AgentRole(Enum):
    """Agent roles from Equipment-Swarm-Coordinator."""

    COORDINATOR = auto()  # Full knowledge, coordinates others
    EXECUTOR = auto()  # Partial knowledge, executes tasks
    VALIDATOR = auto()  # Partial knowledge, validates results
    SPECIALIST = auto()  # Limited knowledge, specialized tasks
    OBSERVER = auto()  # Minimal knowledge, observes and reports
    SCOUT = auto()  # Explores, gathers information
    BUILDER = auto()  # Constructs, implements solutions
    AUDITOR = auto()  # Reviews, checks for correctness


class KnowledgeIsolation(Enum):
    """Knowledge isolation levels."""

    STRICT = auto()  # Only explicitly granted knowledge
    MODERATE = auto()  # Knowledge at or below agent's level
    RELAXED = auto()  # All knowledge accessible


class DecompositionStrategy(Enum):
    """Task decomposition strategies."""

    PARALLEL = auto()
    SEQUENTIAL = auto()
    PIPELINE = auto()
    MAP_REDUCE = auto()
    DIVIDE_CONQUER = auto()


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies."""

    VOTING = auto()
    WEIGHTED = auto()
    HIERARCHICAL = auto()
    CONSENSUS = auto()


@dataclass
class AgentProfile:
    """Profile for a registered agent."""

    id: str
    role: AgentRole
    capabilities: list[str] = field(default_factory=list)
    trust_score: float = 1.0
    performance_score: float = 1.0
    hierarchy_level: int = 0
    knowledge_level: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskNode:
    """A task in a decomposition graph."""

    id: str
    description: str
    agent_id: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    completed: bool = False
    result: Any = None


@dataclass
class ConflictReport:
    """Report from conflict resolution."""

    winner: str
    strategy: str
    votes: dict
    confidence: float
    reasoning: str


class SwarmCoordinator:
    """
    Swarm coordinator with agent roles, conflict resolution, and task decomposition.

    Compatible with Equipment-Swarm-Coordinator patterns.
    """

    def __init__(self, max_agents: int = 10, knowledge_isolation: str = "moderate"):
        self.max_agents = max_agents
        self.knowledge_isolation = KnowledgeIsolation[knowledge_isolation.upper()]
        self.agents: dict[str, AgentProfile] = {}
        self.trust_matrix: dict[tuple[str, str], float] = {}

    def register_agent(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: list[str] = None,
        trust_score: float = 1.0,
        performance_score: float = 1.0,
        hierarchy_level: int = 0,
        knowledge_level: int = 0,
    ) -> AgentProfile:
        """Register a new agent in the swarm."""
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Max agents ({self.max_agents}) reached")
        profile = AgentProfile(
            id=agent_id,
            role=role,
            capabilities=capabilities or [],
            trust_score=trust_score,
            performance_score=performance_score,
            hierarchy_level=hierarchy_level,
            knowledge_level=knowledge_level,
        )
        self.agents[agent_id] = profile
        return profile

    def set_trust(self, from_agent: str, to_agent: str, score: float) -> None:
        """Set trust score between two agents."""
        self.trust_matrix[(from_agent, to_agent)] = max(0.0, min(1.0, score))

    def get_trust(self, from_agent: str, to_agent: str) -> float:
        """Get trust score between two agents."""
        return self.trust_matrix.get((from_agent, to_agent), 0.5)

    def resolve_conflict(
        self,
        options: list[str],
        strategy: str = "weighted",
        agent_votes: Optional[dict[str, str]] = None,
    ) -> ConflictReport:
        """
        Resolve conflict between options using a strategy.

        Args:
            options: List of options to choose from
            strategy: One of "voting", "weighted", "hierarchical", "consensus"
            agent_votes: Optional dict of {agent_id: option} for weighted/hierarchical
        """
        strategy_enum = ConflictResolutionStrategy[strategy.upper().replace("-", "_")]
        votes: dict[str, int] = {opt: 0 for opt in options}

        if strategy_enum == ConflictResolutionStrategy.VOTING:
            # Simple democratic voting
            if agent_votes:
                for choice in agent_votes.values():
                    if choice in votes:
                        votes[choice] += 1
            winner = max(votes, key=votes.get)
            total = sum(votes.values())
            confidence = votes[winner] / max(total, 1)
            return ConflictReport(
                winner=winner,
                strategy="voting",
                votes=votes,
                confidence=confidence,
                reasoning=f"Democratic vote: {winner} wins with {votes[winner]} votes",
            )

        elif strategy_enum == ConflictResolutionStrategy.WEIGHTED:
            # Weighted by agent trust and performance
            weighted_votes: dict[str, float] = {opt: 0.0 for opt in options}
            if agent_votes:
                for agent_id, choice in agent_votes.items():
                    if choice in weighted_votes and agent_id in self.agents:
                        agent = self.agents[agent_id]
                        weight = agent.trust_score * agent.performance_score
                        weighted_votes[choice] += weight
            winner = max(weighted_votes, key=weighted_votes.get)
            total = sum(weighted_votes.values())
            confidence = weighted_votes[winner] / max(total, 1e-9)
            return ConflictReport(
                winner=winner,
                strategy="weighted",
                votes={k: int(v) for k, v in weighted_votes.items()},
                confidence=confidence,
                reasoning=f"Weighted by trust×performance: {winner} wins",
            )

        elif strategy_enum == ConflictResolutionStrategy.HIERARCHICAL:
            # Highest hierarchy wins
            if agent_votes:
                best_agent = None
                best_level = -1
                for agent_id, choice in agent_votes.items():
                    if agent_id in self.agents:
                        agent = self.agents[agent_id]
                        if agent.hierarchy_level > best_level:
                            best_level = agent.hierarchy_level
                            best_agent = agent_id
                winner = (
                    agent_votes.get(best_agent, options[0])
                    if best_agent
                    else options[0]
                )
            else:
                winner = options[0]
            return ConflictReport(
                winner=winner,
                strategy="hierarchical",
                votes=votes,
                confidence=1.0,
                reasoning=f"Hierarchical: highest-level agent chose {winner}",
            )

        elif strategy_enum == ConflictResolutionStrategy.CONSENSUS:
            # Consensus requires > 50% agreement
            if agent_votes:
                for choice in agent_votes.values():
                    if choice in votes:
                        votes[choice] += 1
            total = sum(votes.values())
            if total > 0:
                winner = max(votes, key=votes.get)
                confidence = votes[winner] / total
                if confidence >= 0.5:
                    return ConflictReport(
                        winner=winner,
                        strategy="consensus",
                        votes=votes,
                        confidence=confidence,
                        reasoning=f"Consensus achieved: {winner} with {confidence:.0%} agreement",
                    )
            return ConflictReport(
                winner=options[0],
                strategy="consensus",
                votes=votes,
                confidence=0.0,
                reasoning="No consensus reached — defaulting to first option",
            )

        return ConflictReport(
            winner=options[0],
            strategy=strategy,
            votes=votes,
            confidence=0.0,
            reasoning="Unknown strategy",
        )

    def decompose_task(
        self,
        description: str,
        strategy: str = "parallel",
        subtasks: Optional[list[str]] = None,
    ) -> list[TaskNode]:
        """
        Decompose a task into subtasks.

        Args:
            description: Task description
            strategy: Decomposition strategy
            subtasks: Optional pre-defined subtask descriptions
        """
        strategy_enum = DecompositionStrategy[strategy.upper().replace("-", "_")]

        if subtasks:
            nodes = []
            for i, st in enumerate(subtasks):
                deps = []
                if strategy_enum == DecompositionStrategy.SEQUENTIAL:
                    deps = [subtasks[i - 1]] if i > 0 else []
                elif strategy_enum == DecompositionStrategy.PIPELINE:
                    deps = [f"stage-{i - 1}"] if i > 0 else []
                nodes.append(
                    TaskNode(
                        id=f"stage-{i}"
                        if strategy_enum == DecompositionStrategy.PIPELINE
                        else f"task-{i}",
                        description=st,
                        dependencies=deps,
                    )
                )
            return nodes

        # Default decomposition for common strategies
        if strategy_enum == DecompositionStrategy.MAP_REDUCE:
            return [
                TaskNode(id="map-1", description=f"Map: {description}"),
                TaskNode(id="map-2", description=f"Map: {description}"),
                TaskNode(
                    id="reduce",
                    description=f"Reduce: {description}",
                    dependencies=["map-1", "map-2"],
                ),
            ]
        elif strategy_enum == DecompositionStrategy.DIVIDE_CONQUER:
            return [
                TaskNode(id="divide", description=f"Divide: {description}"),
                TaskNode(
                    id="conquer-1",
                    description="Conquer left half",
                    dependencies=["divide"],
                ),
                TaskNode(
                    id="conquer-2",
                    description="Conquer right half",
                    dependencies=["divide"],
                ),
                TaskNode(
                    id="merge",
                    description="Merge results",
                    dependencies=["conquer-1", "conquer-2"],
                ),
            ]
        else:
            return [TaskNode(id="task-0", description=description)]

    def assign_task(self, task: TaskNode, required_caps: list[str]) -> Optional[str]:
        """Assign a task to the best available agent."""
        candidates = []
        for agent_id, profile in self.agents.items():
            if all(cap in profile.capabilities for cap in required_caps):
                score = profile.trust_score * profile.performance_score
                candidates.append((agent_id, score))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def can_access_knowledge(self, agent_id: str, knowledge_level: int) -> bool:
        """Check if an agent can access knowledge at a given level."""
        if agent_id not in self.agents:
            return False
        agent = self.agents[agent_id]
        if self.knowledge_isolation == KnowledgeIsolation.STRICT:
            return knowledge_level <= agent.knowledge_level
        elif self.knowledge_isolation == KnowledgeIsolation.MODERATE:
            return knowledge_level <= agent.hierarchy_level + 1
        else:  # RELAXED
            return True

    def to_dict(self) -> dict:
        """Serialize coordinator state."""
        return {
            "max_agents": self.max_agents,
            "knowledge_isolation": self.knowledge_isolation.name,
            "agents": {
                aid: {
                    "id": a.id,
                    "role": a.role.name,
                    "capabilities": a.capabilities,
                    "trust_score": a.trust_score,
                    "performance_score": a.performance_score,
                    "hierarchy_level": a.hierarchy_level,
                }
                for aid, a in self.agents.items()
            },
            "trust_matrix": {f"{k[0]}→{k[1]}": v for k, v in self.trust_matrix.items()},
        }

    def render_ascii(self) -> str:
        """Render ASCII swarm visualization."""
        lines = [
            "+" + "-" * 48 + "+",
            "| SWARM COORDINATOR                              |",
            "+" + "-" * 48 + "+",
        ]
        for agent_id, profile in self.agents.items():
            role_icon = {
                AgentRole.COORDINATOR: "👑",
                AgentRole.EXECUTOR: "⚡",
                AgentRole.VALIDATOR: "✓",
                AgentRole.SPECIALIST: "🔧",
                AgentRole.OBSERVER: "👁",
                AgentRole.SCOUT: "🔭",
                AgentRole.BUILDER: "🏗",
                AgentRole.AUDITOR: "🔍",
            }.get(profile.role, "?")
            lines.append(
                f"| {role_icon} {agent_id:12} | {profile.role.name:12} | trust={profile.trust_score:.2f} |"
            )
        lines.append("+" + "-" * 48 + "+")
        return "\n".join(lines)
