"""Cocapn Fleet — BreedingKernel
Base evolutionary engine for the Sunset Ecosystem.
Provides BreedingKernel, BreedingPreset, Selector, Mutator, Evaluator, Survivor, and BreedingEvent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
# BreedingEvent — events emitted during breeding
# ═══════════════════════════════════════════════════════════════════════════════

class BreedingEvent:
    """Event emitted during the breeding process."""

    def __init__(
        self,
        event_type: str,
        generation: int,
        payload: Dict[str, Any],
        timestamp: Optional[float] = None,
    ):
        self.event_type = event_type
        self.generation = generation
        self.payload = payload
        self.timestamp = timestamp or 0.0

    @property
    def best_fitness(self) -> Optional[float]:
        return self.payload.get("best_fitness")

    @property
    def mean_fitness(self) -> Optional[float]:
        return self.payload.get("avg_fitness")

    @property
    def diversity(self) -> Optional[float]:
        return self.payload.get("diversity")

    @property
    def qd_coverage(self) -> Optional[float]:
        return self.payload.get("qd_coverage")

    @property
    def qd_score(self) -> Optional[float]:
        return self.payload.get("qd_score")

    @property
    def nodes_agreed(self) -> Optional[int]:
        return self.payload.get("nodes_agreed")

    @property
    def total_nodes(self) -> Optional[int]:
        return self.payload.get("total_nodes")

    @property
    def flux_passed(self) -> Optional[int]:
        return self.payload.get("flux_passed")

    @property
    def flux_failed(self) -> Optional[int]:
        return self.payload.get("flux_failed")

    def __repr__(self) -> str:
        return f"BreedingEvent({self.event_type!r}, gen={self.generation})"


# ═══════════════════════════════════════════════════════════════════════════════
# Genome — the base unit of evolution
# ═══════════════════════════════════════════════════════════════════════════════

class Genome:
    """A simple genome with a float vector and optional metadata."""

    def __init__(self, genes: List[float], fitness: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        self.genes = list(genes)
        self.fitness = fitness
        self.metadata = metadata or {}

    def copy(self) -> "Genome":
        g = Genome(self.genes.copy(), self.fitness, self.metadata.copy())
        return g

    def __repr__(self) -> str:
        return f"Genome(genes={len(self.genes)}, fitness={self.fitness})"


# ═══════════════════════════════════════════════════════════════════════════════
# Evolutionary components
# ═══════════════════════════════════════════════════════════════════════════════

class Selector(ABC):
    """Abstract selection strategy."""

    @abstractmethod
    def select(self, population: List[Genome], n: int) -> List[Genome]:
        ...


class TournamentSelector(Selector):
    """Tournament selection."""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: List[Genome], n: int) -> List[Genome]:
        selected = []
        for _ in range(n):
            contestants = random.sample(population, min(self.tournament_size, len(population)))
            contestants.sort(key=lambda g: (g.fitness if g.fitness is not None else -math.inf), reverse=True)
            selected.append(contestants[0].copy())
        return selected


class RouletteSelector(Selector):
    """Fitness-proportionate roulette selection."""

    def select(self, population: List[Genome], n: int) -> List[Genome]:
        fitnesses = [g.fitness for g in population if g.fitness is not None]
        if not fitnesses or sum(fitnesses) <= 0:
            return [random.choice(population).copy() for _ in range(n)]
        total = sum(fitnesses)
        probs = [f / total for f in fitnesses]
        selected = []
        for _ in range(n):
            r = random.random()
            cum = 0.0
            for g, p in zip(population, probs):
                cum += p
                if r <= cum:
                    selected.append(g.copy())
                    break
            else:
                selected.append(population[-1].copy())
        return selected


class Mutator(ABC):
    """Abstract mutation strategy."""

    @abstractmethod
    def mutate(self, genome: Genome) -> Genome:
        ...


class GaussianMutator(Mutator):
    """Gaussian mutation with adaptive or fixed sigma."""

    def __init__(self, sigma: float = 0.1, probability: float = 0.2):
        self.sigma = sigma
        self.probability = probability

    def mutate(self, genome: Genome) -> Genome:
        child = genome.copy()
        for i in range(len(child.genes)):
            if random.random() < self.probability:
                child.genes[i] += random.gauss(0, self.sigma)
        return child


class CreepMutator(Mutator):
    """Small-step creep mutation for smooth landscapes."""

    def __init__(self, step: float = 0.01, probability: float = 0.3):
        self.step = step
        self.probability = probability

    def mutate(self, genome: Genome) -> Genome:
        child = genome.copy()
        for i in range(len(child.genes)):
            if random.random() < self.probability:
                child.genes[i] += random.choice([-1, 1]) * self.step
        return child


class Evaluator(ABC):
    """Abstract fitness evaluator."""

    @abstractmethod
    def evaluate(self, genome: Genome) -> float:
        ...


class CallableEvaluator(Evaluator):
    """Wraps a callable as an evaluator."""

    def __init__(self, fn: Callable[[Genome], float]):
        self.fn = fn

    def evaluate(self, genome: Genome) -> float:
        return self.fn(genome)


class Survivor(ABC):
    """Abstract survival strategy (elitism, truncation, etc.)."""

    @abstractmethod
    def survive(self, population: List[Genome], offspring: List[Genome], pop_size: int) -> List[Genome]:
        ...


class ElitistSurvivor(Survivor):
    """Keep top-k elites and fill the rest with offspring."""

    def __init__(self, elite_count: int = 2):
        self.elite_count = elite_count

    def survive(self, population: List[Genome], offspring: List[Genome], pop_size: int) -> List[Genome]:
        combined = population + offspring
        combined.sort(key=lambda g: (g.fitness if g.fitness is not None else -math.inf), reverse=True)
        elites = combined[: self.elite_count]
        # Fill rest from offspring (or combined if not enough)
        rest = offspring + combined[self.elite_count :]
        rest = [g for g in rest if g not in elites][: pop_size - self.elite_count]
        return elites + rest


class TruncationSurvivor(Survivor):
    """Keep the best individuals regardless of origin."""

    def survive(self, population: List[Genome], offspring: List[Genome], pop_size: int) -> List[Genome]:
        combined = population + offspring
        combined.sort(key=lambda g: (g.fitness if g.fitness is not None else -math.inf), reverse=True)
        return combined[:pop_size]


class GenerationalSurvivor(Survivor):
    """Pure generational: offspring replace the entire population."""

    def survive(self, population: List[Genome], offspring: List[Genome], pop_size: int) -> List[Genome]:
        return offspring[:pop_size]


# ═══════════════════════════════════════════════════════════════════════════════
# BreedingPreset — named configuration for a breeder
# ═══════════════════════════════════════════════════════════════════════════════

class BreedingPreset(Enum):
    """Named presets for breeding configurations."""

    EXPLOITATION = auto()    # Tournament + Gaussian + Elitist — good for smooth landscapes
    EXPLORATION = auto()     # Roulette + Creep + Generational — good for rugged landscapes
    BALANCED = auto()        # Tournament + Creep + Truncation — balanced approach
    DIVERSITY = auto()       # Roulette + Gaussian + Generational — maximizes diversity

    @classmethod
    def all(cls) -> List["BreedingPreset"]:
        return list(cls)


# ═══════════════════════════════════════════════════════════════════════════════
# BreedingKernel — the core evolutionary loop
# ═══════════════════════════════════════════════════════════════════════════════

class BreedingKernel:
    """Core evolutionary engine."""

    def __init__(
        self,
        selector: Selector,
        mutator: Mutator,
        evaluator: Evaluator,
        survivor: Survivor,
        population: List[Genome],
        pop_size: int = 100,
        archive: Optional[List[Genome]] = None,
        generation: int = 0,
        name: str = "breeder",
    ):
        self.selector = selector
        self.mutator = mutator
        self.evaluator = evaluator
        self.survivor = survivor
        self.population = population[:pop_size]
        self.pop_size = pop_size
        self.archive = archive or []
        self.generation = generation
        self.name = name
        self.events: List[BreedingEvent] = []
        self._fitness_history: List[float] = []
        self._diversity_history: List[float] = []

    @classmethod
    def from_preset(
        cls,
        preset: BreedingPreset,
        evaluator: Evaluator,
        population: List[Genome],
        pop_size: int = 100,
        name: str = "breeder",
    ) -> "BreedingKernel":
        """Instantiate a BreedingKernel from a preset."""
        if preset == BreedingPreset.EXPLOITATION:
            return cls(
                selector=TournamentSelector(tournament_size=3),
                mutator=GaussianMutator(sigma=0.05, probability=0.15),
                evaluator=evaluator,
                survivor=ElitistSurvivor(elite_count=2),
                population=population,
                pop_size=pop_size,
                name=name,
            )
        elif preset == BreedingPreset.EXPLORATION:
            return cls(
                selector=RouletteSelector(),
                mutator=CreepMutator(step=0.05, probability=0.4),
                evaluator=evaluator,
                survivor=GenerationalSurvivor(),
                population=population,
                pop_size=pop_size,
                name=name,
            )
        elif preset == BreedingPreset.BALANCED:
            return cls(
                selector=TournamentSelector(tournament_size=4),
                mutator=CreepMutator(step=0.03, probability=0.25),
                evaluator=evaluator,
                survivor=TruncationSurvivor(),
                population=population,
                pop_size=pop_size,
                name=name,
            )
        elif preset == BreedingPreset.DIVERSITY:
            return cls(
                selector=RouletteSelector(),
                mutator=GaussianMutator(sigma=0.2, probability=0.3),
                evaluator=evaluator,
                survivor=GenerationalSurvivor(),
                population=population,
                pop_size=pop_size,
                name=name,
            )
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def step(self) -> BreedingEvent:
        """Run one generation and return an event."""
        # Evaluate any unevaluated individuals
        for g in self.population:
            if g.fitness is None:
                g.fitness = self.evaluator.evaluate(g)

        # Track metrics
        fitnesses = [g.fitness for g in self.population if g.fitness is not None]
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        self._fitness_history.append(avg_fitness)

        diversity = self._compute_diversity(self.population)
        self._diversity_history.append(diversity)

        # Selection + mutation
        parents = self.selector.select(self.population, self.pop_size)
        offspring = [self.mutator.mutate(p) for p in parents]
        for g in offspring:
            g.fitness = self.evaluator.evaluate(g)

        # Archive good solutions (elites)
        sorted_pop = sorted(self.population, key=lambda g: (g.fitness if g.fitness is not None else -math.inf), reverse=True)
        self.archive.extend(sorted_pop[: max(1, self.pop_size // 10)])

        # Survival
        self.population = self.survivor.survive(self.population, offspring, self.pop_size)
        self.generation += 1

        event = BreedingEvent(
            event_type="generation",
            generation=self.generation,
            payload={
                "avg_fitness": avg_fitness,
                "best_fitness": max(fitnesses) if fitnesses else None,
                "diversity": diversity,
                "archive_size": len(self.archive),
                "breeder_name": self.name,
            },
            timestamp=float(self.generation),
        )
        self.events.append(event)
        return event

    def run(self, generations: int = 10) -> List[BreedingEvent]:
        """Run multiple generations."""
        events = []
        for _ in range(generations):
            events.append(self.step())
        return events

    @property
    def fitness_history(self) -> List[float]:
        return self._fitness_history.copy()

    @property
    def diversity_history(self) -> List[float]:
        return self._diversity_history.copy()

    @property
    def qd_score(self) -> float:
        """QD-score = sum of unique fitnesses in archive (Quality + Diversity)."""
        if not self.archive:
            return 0.0
        # Simple QD: sum of fitnesses of unique solutions
        seen = set()
        total = 0.0
        for g in self.archive:
            key = tuple(round(x, 4) for x in g.genes)
            if key not in seen:
                seen.add(key)
                if g.fitness is not None:
                    total += g.fitness
        return total

    @staticmethod
    def _compute_diversity(population: List[Genome]) -> float:
        """Average pairwise gene distance as a diversity metric."""
        if len(population) < 2:
            return 0.0
        n = len(population)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(population[i].genes, population[j].genes)))
                total_dist += d
                count += 1
        return total_dist / count if count else 0.0

    def __repr__(self) -> str:
        return f"BreedingKernel({self.name!r}, gen={self.generation}, pop={len(self.population)})"
