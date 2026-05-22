"""Slot inheritance tax to prevent dynastic monopolies.

High-fitness agents can't just hoard slots across generations.
Tax increases with inheritance ratio AND fitness delta (rich get taxed more).
Tax revenue goes to a global pool that funds new-agent slots (social welfare).
"""

from __future__ import annotations

__all__ = [
    "InheritanceTax",
    "Agent",
]

from dataclasses import dataclass
from typing import Optional


@dataclass
class Agent:
    """Minimal agent record for dynasty scoring.

    Attributes:
        agent_id: Unique identifier.
        slots: Economic slots this agent controls.
        fitness: Current fitness score [0, 1].
        parent_id: Immediate parent (None for root agents).
    """
    agent_id: int
    slots: int
    fitness: float
    parent_id: Optional[int] = None


class InheritanceTax:
    """Slot inheritance tax to prevent dynastic monopolies.

    Tax brackets (inheritance ratio → base tax rate):
        0-30%   → 0%
        30-60%  → 15%
        60-90%  → 30%
        90-100% → 50%

    The actual rate is ``base_rate * (1 + fitness_delta)`` so that
    agents with a large fitness advantage pay progressively more.
    """

    TAX_BRACKETS = [
        (0.0, 0.3, 0.0),    # 0-30% inherited: 0% tax
        (0.3, 0.6, 0.15),   # 30-60% inherited: 15% tax
        (0.6, 0.9, 0.30),   # 60-90% inherited: 30% tax
        (0.9, 1.0, 0.50),   # 90-100% inherited: 50% tax
    ]

    DEFAULT_SLOTS = 10

    def __init__(self, tax_brackets: list[tuple] = None) -> None:
        self.tax_brackets = tax_brackets or list(self.TAX_BRACKETS)
        self.global_pool: int = 0

    def compute_tax(
        self,
        parent_slots: int,
        child_slots: int,
        parent_fitness: float,
        child_fitness: float,
    ) -> float:
        """Compute tax rate based on what fraction of parent slots child inherits.

        Args:
            parent_slots: Slots the parent currently holds.
            child_slots: Slots the child will inherit.
            parent_fitness: Parent's fitness [0, 1].
            child_fitness: Child's fitness [0, 1].

        Returns:
            Tax rate in [0, 1].
        """
        if parent_slots <= 0:
            return 0.0

        inheritance_ratio = child_slots / parent_slots

        # Base rate from brackets
        base_rate = 0.0
        for low, high, rate in self.tax_brackets:
            if low <= inheritance_ratio <= high:
                base_rate = rate
                break

        # Fitness delta multiplier: rich get taxed more
        fitness_delta = max(0.0, parent_fitness - child_fitness)
        fitness_multiplier = 1.0 + fitness_delta

        tax_rate = base_rate * fitness_multiplier
        return min(1.0, tax_rate)

    def apply_tax(
        self,
        parent_slots: int,
        child_slots: int,
        parent_fitness: float,
        child_fitness: float,
    ) -> tuple[int, int]:
        """Apply tax and return post-tax slot counts.

        Args:
            parent_slots: Slots the parent currently holds.
            child_slots: Slots the child will inherit.
            parent_fitness: Parent's fitness [0, 1].
            child_fitness: Child's fitness [0, 1].

        Returns:
            (parent_slots_after_tax, child_slots_after_tax).
            Tax revenue is added to ``self.global_pool``.
        """
        tax_rate = self.compute_tax(
            parent_slots, child_slots, parent_fitness, child_fitness
        )
        tax_amount = int(round(parent_slots * tax_rate))

        parent_after = max(0, parent_slots - tax_amount)
        child_after = child_slots

        self.global_pool += tax_amount

        return (parent_after, child_after)

    def fund_new_agent(self, requested_slots: int) -> int:
        """Grant slots from the global tax pool to a new agent.

        Args:
            requested_slots: How many slots the new agent wants.

        Returns:
            Slots actually granted (may be less than requested).
        """
        granted = min(requested_slots, self.global_pool)
        self.global_pool -= granted
        return granted

    def dynasty_score(self, agent_id: int, population: list[Agent]) -> float:
        """Measure how 'dynastic' an agent's lineage is.

        Returns the fraction of the agent's slots that were inherited
        versus earned independently.

        Args:
            agent_id: Target agent.
            population: Full population (must include agent and ancestors).

        Returns:
            Dynasty score in [0, 1]. 0 = fully earned, 1 = fully inherited.
        """
        agent_map = {a.agent_id: a for a in population}
        agent = agent_map.get(agent_id)
        if agent is None:
            return 0.0

        inherited_slots = 0
        if agent.parent_id is not None and agent.parent_id in agent_map:
            parent = agent_map[agent.parent_id]
            inherited_slots = min(agent.slots, parent.slots)

        earned_slots = max(0, agent.slots - inherited_slots)
        total = inherited_slots + earned_slots

        if total == 0:
            return 0.0

        return inherited_slots / total
