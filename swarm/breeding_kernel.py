"""BreedingKernel — Common interface for all 18 breeder types.

Every breeder in the fleet is a variation of the same kernel:

    selector.select() → mutator.crossover() → mutator.mutate()
    → evaluator.evaluate() → survivor.merge()

This module extracts that kernel and makes every breeder a preset.

Usage
-----
    from swarm.breeding_kernel import BreedingKernel, BreedingPreset

    kernel = BreedingKernel.from_preset(BreedingPreset.PYTHAGOREAN)
    for event in kernel.run(population, generations=100):
        print(f"Gen {event.generation}: best={event.best_fitness}")

    # Custom kernel
    kernel = BreedingKernel(
        selector=MySelector(),
        mutator=MyMutator(),
        evaluator=MyEvaluator(),
        survivor=MySurvivor(),
    )
"""
from __future__ import annotations

__all__ = [
    "BreedingEvent",
    "BreedingKernel",
    "BreedingPreset",
    "Selector",
    "Mutator",
    "Evaluator",
    "Survivor",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np


# ── Event ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class BreedingEvent:
    """Emitted during a breeding run."""
    generation: int
    best_fitness: float
    mean_fitness: float
    population_size: int
    diversity: float
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── ABCs ────────────────────────────────────────────────────────

class Selector(ABC):
    """Choose parents from the population."""

    @abstractmethod
    def select(self, population: List[Any], fitness: List[float], n_parents: int) -> List[Any]:
        """Return ``n_parents`` selected from ``population``."""
        ...


class Mutator(ABC):
    """Create offspring from parents."""

    @abstractmethod
    def crossover(self, parents: List[Any], n_offspring: int) -> List[Any]:
        """Return offspring created by crossover."""
        ...

    @abstractmethod
    def mutate(self, offspring: List[Any], rate: float = 0.1) -> List[Any]:
        """Return mutated offspring."""
        ...


class Evaluator(ABC):
    """Score individuals."""

    @abstractmethod
    def evaluate(self, individuals: List[Any]) -> List[float]:
        """Return fitness scores for each individual."""
        ...


class Survivor(ABC):
    """Merge old and new into next generation."""

    @abstractmethod
    def merge(self, old: List[Any], old_fitness: List[float],
              new: List[Any], new_fitness: List[float]) -> List[Any]:
        """Return the next generation population."""
        ...


# ── Presets ─────────────────────────────────────────────────────

class BreedingPreset(Enum):
    """Known breeder presets."""
    TOURNAMENT = auto()
    PYTHAGOREAN = auto()
    SPECTRAL = auto()
    HAMILTONIAN = auto()
    BOUNDED = auto()
    CAUSAL = auto()
    INFORMATION_THEORETIC = auto()
    ADVERSARIAL = auto()
    NCA = auto()
    GNN = auto()
    META_LEARNING = auto()
    SWARM_INTELLIGENCE = auto()
    DIFFERENTIAL = auto()
    SPATIAL = auto()
    ENSEMBLE = auto()
    SIM_REAL = auto()
    BFT_QD = auto()
    CONSTRAINT = auto()


# ── Kernel ──────────────────────────────────────────────────────

class BreedingKernel:
    """Common breeding kernel with pluggable strategies."""

    def __init__(
        self,
        selector: Selector,
        mutator: Mutator,
        evaluator: Evaluator,
        survivor: Survivor,
        population_size: int = 100,
    ):
        self.selector = selector
        self.mutator = mutator
        self.evaluator = evaluator
        self.survivor = survivor
        self.population_size = population_size

    @classmethod
    def from_preset(cls, preset: BreedingPreset, **kwargs: Any) -> BreedingKernel:
        """Create a kernel from a named preset."""
        # Each preset imports its own modules lazily to avoid circular deps
        if preset == BreedingPreset.TOURNAMENT:
            from simulators.tournament_core import TournamentSelector, TournamentMutator, TournamentEvaluator
            return cls(
                selector=TournamentSelector(),
                mutator=TournamentMutator(),
                evaluator=TournamentEvaluator(),
                survivor=TournamentSurvivor(),
                **kwargs,
            )
        # TODO: Add other 17 presets as modules are built
        raise NotImplementedError(f"Preset {preset.name} not yet wired")

    def run(self, population: List[Any], generations: int = 100,
            mutation_rate: float = 0.1) -> Iterator[BreedingEvent]:
        """Run breeding for ``generations`` and yield events."""
        import time
        fitness = self.evaluator.evaluate(population)
        for gen in range(generations):
            t0 = time.perf_counter()
            parents = self.selector.select(population, fitness, n_parents=max(2, len(population) // 4))
            offspring_cross = self.mutator.crossover(parents, n_offspring=len(population) // 2)
            offspring_mut = self.mutator.mutate(offspring_cross, rate=mutation_rate)
            offspring_fitness = self.evaluator.evaluate(offspring_mut)
            population = self.survivor.merge(population, fitness, offspring_mut, offspring_fitness)
            fitness = self.evaluator.evaluate(population)
            elapsed = (time.perf_counter() - t0) * 1000
            diversity = np.std(fitness) if fitness else 0.0
            yield BreedingEvent(
                generation=gen,
                best_fitness=max(fitness) if fitness else 0.0,
                mean_fitness=np.mean(fitness) if fitness else 0.0,
                population_size=len(population),
                diversity=float(diversity),
                elapsed_ms=elapsed,
            )


# ── Survivor implementation (default: merge best half) ────────────

class TournamentSurvivor(Survivor):
    """Keep the best individuals from old + new."""

    def merge(self, old: List[Any], old_fitness: List[float],
              new: List[Any], new_fitness: List[float]) -> List[Any]:
        combined = list(zip(old + new, old_fitness + new_fitness))
        combined.sort(key=lambda x: x[1], reverse=True)
        target = len(old)
        return [ind for ind, _ in combined[:target]]
