"""Tournament-style relative fitness for agent selection.

Agents compete head-to-head across three axes: ethos, pathos, logos.
Pareto analysis identifies dominated agents for sunsetting.
Winners breed to produce the next generation.
"""

from __future__ import annotations

__all__ = [
    "AgentScore",
    "TournamentMatch",
    "TournamentRound",
    "dominated_by",
    "breed",
    "sunset_candidates",
]

import random
import uuid
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


@dataclass(frozen=True)
class AgentScore:
    """Trinity score for a single agent.

    Attributes:
        agent_id: Unique identifier for the agent.
        ethos: Values alignment score [0, 1].
        pathos: Emotional resonance score [0, 1].
        logos: Logical relevance score [0, 1].
    """
    agent_id: str
    ethos: float
    pathos: float
    logos: float

    def __post_init__(self) -> None:
        for name, val in [("ethos", self.ethos), ("pathos", self.pathos), ("logos", self.logos)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    @property
    def product(self) -> float:
        """Trinity product: zero in any dimension kills the agent."""
        return self.ethos * self.pathos * self.logos

    def __repr__(self) -> str:
        return (
            f"AgentScore({self.agent_id!r}, "
            f"E={self.ethos:.2f} P={self.pathos:.2f} L={self.logos:.2f}, "
            f"prod={self.product:.4f})"
        )


@dataclass
class TournamentMatch:
    """One head-to-head comparison between two agents.

    The *scores* dict maps agent_id → float (composite score)
    used to determine the winner of this match.

    Attributes:
        agent_a: First agent's ID.
        agent_b: Second agent's ID.
        scores: Mapping of agent_id → composite score for this matchup.
        winner: ID of the winning agent (set after resolution).
    """
    agent_a: str
    agent_b: str
    scores: dict[str, float] = field(default_factory=dict)
    winner: str | None = None

    def resolve(self) -> str:
        """Determine the winner from scores. Returns winner ID."""
        if not self.scores:
            raise ValueError("No scores to resolve match")
        self.winner = max(self.scores, key=lambda aid: self.scores[aid])
        return self.winner

    def __repr__(self) -> str:
        return (
            f"TournamentMatch({self.agent_a!r} vs {self.agent_b!r}, "
            f"winner={self.winner!r})"
        )


@dataclass
class TournamentResult:
    """Ranked result for one agent from a tournament round.

    Attributes:
        agent_id: The agent.
        wins: Number of matches won.
        losses: Number of matches lost.
        scores: The agent's trinity scores.
        rank: Final rank (1 = best).
    """
    agent_id: str
    wins: int = 0
    losses: int = 0
    scores: AgentScore | None = None
    rank: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"TournamentResult({self.agent_id!r}, "
            f"W={self.wins} L={self.losses}, rank={self.rank})"
        )


class TournamentRound:
    """Runs all pairwise matches and produces ranked results.

    Args:
        population: List of AgentScore for every competing agent.
    """

    def __init__(self, population: list[AgentScore]) -> None:
        self._population = list(population)
        self._matches: list[TournamentMatch] = []
        self._results: dict[str, TournamentResult] = {}
        self._pareto_frontier: list[AgentScore] = []

    def __repr__(self) -> str:
        return (
            f"TournamentRound(population={len(self._population)}, "
            f"matches={len(self._matches)})"
        )

    @property
    def matches(self) -> list[TournamentMatch]:
        return list(self._matches)

    @property
    def results(self) -> dict[str, TournamentResult]:
        return dict(self._results)

    @property
    def pareto_frontier(self) -> list[AgentScore]:
        return list(self._pareto_frontier)

    def run(self) -> list[TournamentResult]:
        """Run all pairwise matches and rank agents.

        Returns:
            Sorted list of TournamentResult (best first).
        """
        self._matches.clear()
        self._results.clear()

        score_map = {s.agent_id: s for s in self._population}

        for aid in self._population:
            self._results[aid.agent_id] = TournamentResult(
                agent_id=aid.agent_id,
                scores=aid,
            )

        for a, b in combinations(self._population, 2):
            match = TournamentMatch(
                agent_a=a.agent_id,
                agent_b=b.agent_id,
                scores={
                    a.agent_id: a.product,
                    b.agent_id: b.product,
                },
            )
            match.resolve()
            self._matches.append(match)

            winner_id = match.winner
            loser_id = b.agent_id if winner_id == a.agent_id else a.agent_id
            self._results[winner_id].wins += 1
            self._results[loser_id].losses += 1

        ranked = sorted(
            self._results.values(),
            key=lambda r: (r.wins, r.win_rate),
            reverse=True,
        )
        for i, r in enumerate(ranked, 1):
            r.rank = i

        self._pareto_frontier = _compute_pareto_frontier(self._population)
        return ranked


def _compute_pareto_frontier(population: list[AgentScore]) -> list[AgentScore]:
    """Compute the Pareto frontier across ethos, pathos, logos.

    An agent is on the frontier if no other agent dominates it
    on all three axes.
    """
    frontier: list[AgentScore] = []
    for candidate in population:
        if not dominated_by(candidate, population):
            frontier.append(candidate)
    return frontier


def dominated_by(agent: AgentScore, population: list[AgentScore]) -> bool:
    """Check whether any agent in population dominates *agent* on all 3 axes.

    Agent A dominates agent B if A is >= B on ethos, pathos, AND logos,
    and strictly > on at least one axis.

    Args:
        agent: The candidate agent.
        population: The full population to check against.

    Returns:
        True if the agent is dominated by at least one other agent.
    """
    for other in population:
        if other.agent_id == agent.agent_id:
            continue
        if (
            other.ethos >= agent.ethos
            and other.pathos >= agent.pathos
            and other.logos >= agent.logos
            and (
                other.ethos > agent.ethos
                or other.pathos > agent.pathos
                or other.logos > agent.logos
            )
        ):
            return True
    return False


def breed(
    winners: list[AgentScore],
    num_children: int,
) -> list[dict[str, Any]]:
    """Breed child agent configs from tournament winners.

    Each child is a random crossover between two parent winners,
    with slight mutation on each axis.

    Args:
        winners: Winners from the tournament (Pareto frontier).
        num_children: How many children to produce.

    Returns:
        List of child config dicts with id, parent_a, parent_b,
        ethos, pathos, logos.
    """
    if len(winners) < 2:
        # Single parent — clone with mutation
        parent = winners[0] if winners else None
        children: list[dict[str, Any]] = []
        for _ in range(num_children):
            children.append({
                "id": uuid.uuid4().hex[:12],
                "parent_a": parent.agent_id if parent else None,
                "parent_b": None,
                "ethos": _mutate(parent.ethos if parent else 0.5),
                "pathos": _mutate(parent.pathos if parent else 0.5),
                "logos": _mutate(parent.logos if parent else 0.5),
            })
        return children

    children = []
    for _ in range(num_children):
        a, b = random.sample(winners, 2)
        child = {
            "id": uuid.uuid4().hex[:12],
            "parent_a": a.agent_id,
            "parent_b": b.agent_id,
            "ethos": _mutate(_crossover(a.ethos, b.ethos)),
            "pathos": _mutate(_crossover(a.pathos, b.pathos)),
            "logos": _mutate(_crossover(a.logos, b.logos)),
        }
        children.append(child)
    return children


def _crossover(a: float, b: float) -> float:
    """Random crossover between two parent values."""
    t = random.random()
    return a * t + b * (1 - t)


def _mutate(value: float, sigma: float = 0.05) -> float:
    """Gaussian mutation clamped to [0, 1]."""
    return max(0.0, min(1.0, random.gauss(value, sigma)))


def sunset_candidates(population: list[AgentScore]) -> list[AgentScore]:
    """Return agents that are dominated and should be sunset.

    Only agents NOT on the Pareto frontier are candidates for sunsetting.

    Args:
        population: Full population.

    Returns:
        List of dominated agents (candidates for sunset).
    """
    return [a for a in population if dominated_by(a, population)]
