"""Shared tournament simulation core.

Extracted from tournament_sim.py and tournament_sweep.py to eliminate
duplication. Both simulators import from this module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Agent:
    """Trinity agent with ethos, pathos, logos."""

    ethos: float = None
    pathos: float = None
    logos: float = None

    def __post_init__(self):
        if self.ethos is None:
            self.ethos = random.random()
        if self.pathos is None:
            self.pathos = random.random()
        if self.logos is None:
            self.logos = random.random()

    def fitness(self) -> float:
        return self.ethos * self.pathos * self.logos

    def dominates(self, other: Agent) -> bool:
        return (
            self.ethos >= other.ethos
            and self.pathos >= other.pathos
            and self.logos >= other.logos
            and (
                self.ethos > other.ethos
                or self.pathos > other.pathos
                or self.logos > other.logos
            )
        )

    def __repr__(self) -> str:
        return (
            f"A(e={self.ethos:.3f},p={self.pathos:.3f},"
            f"l={self.logos:.3f},f={self.fitness():.4f})"
        )


def crossover(a: Agent, b: Agent) -> Agent:
    """Blend crossover between two parents."""
    alpha = random.random()
    return Agent(
        ethos=alpha * a.ethos + (1 - alpha) * b.ethos,
        pathos=alpha * a.pathos + (1 - alpha) * b.pathos,
        logos=alpha * a.logos + (1 - alpha) * b.logos,
    )


def mutate(agent: Agent, rate: float) -> Agent:
    """Gaussian mutation with clamping to [0, 1]."""
    return Agent(
        ethos=max(0, min(1, agent.ethos + random.gauss(0, rate))),
        pathos=max(0, min(1, agent.pathos + random.gauss(0, rate))),
        logos=max(0, min(1, agent.logos + random.gauss(0, rate))),
    )


def tournament_step(
    pop: List[Agent],
    thermal_cap: int,
    mutation_rate: float,
    *,
    track_breeding: bool = False,
) -> Tuple[List[Agent], int]:
    """Run one tournament step: compete winners, then breed offspring.

    :param pop: Current population.
    :param thermal_cap: Maximum population size after breeding.
    :param mutation_rate: Gaussian mutation standard deviation.
    :param track_breeding: If True, return the number of breeding events.
    :returns: (new_population, breeding_events) — breeding_events is 0
              when track_breeding=False.
    """
    winners = []
    random.shuffle(pop)
    for i in range(0, len(pop) - 1, 2):
        a, b = pop[i], pop[i + 1]
        if a.dominates(b):
            winners.append(a)
        elif b.dominates(a):
            winners.append(b)
        elif a.fitness() > b.fitness():
            winners.append(a)
        else:
            winners.append(b)
    if len(pop) % 2 == 1:
        winners.append(pop[-1])

    # Breed: fill up to near thermal cap
    offspring: List[Agent] = []
    breeding_events = 0
    while len(winners) + len(offspring) < thermal_cap and len(winners) >= 2:
        p1, p2 = random.sample(winners, 2)
        child = mutate(crossover(p1, p2), mutation_rate)
        offspring.append(child)
        breeding_events += 1

    return winners + offspring, breeding_events if track_breeding else 0


def diversity_metric(pop: List[Agent]) -> float:
    """Composite diversity: sum of std devs for ethos, pathos, logos."""
    if len(pop) < 2:
        return 0.0
    n = len(pop)
    mean_e = sum(a.ethos for a in pop) / n
    mean_p = sum(a.pathos for a in pop) / n
    mean_l = sum(a.logos for a in pop) / n
    std_e = math.sqrt(sum((a.ethos - mean_e) ** 2 for a in pop) / n)
    std_p = math.sqrt(sum((a.pathos - mean_p) ** 2 for a in pop) / n)
    std_l = math.sqrt(sum((a.logos - mean_l) ** 2 for a in pop) / n)
    return std_e + std_p + std_l


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))
