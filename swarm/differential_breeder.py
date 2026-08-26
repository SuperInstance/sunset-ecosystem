"""
Differential Breeder

Differential Evolution (DE) inspired breeding algorithm.
Uses vector differences between population members to guide mutation.
Particularly effective for continuous optimization.

Usage:
    from fleet.differential_breeder import DifferentialBreeder
    breeder = DifferentialBreeder(population_size=50, dimensions=10)
    breeder.initialize()
    for gen in range(100):
        breeder.evolve()
    best = breeder.get_best()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DEIndividual:
    """An individual in Differential Evolution."""

    genome: np.ndarray
    fitness: float = 0.0
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome": self.genome.tolist(),
            "fitness": self.fitness,
            "generation": self.generation,
        }


class DifferentialBreeder:
    """
    Differential Evolution breeder.

    Key hyperparameters:
    - F: differential weight (mutation factor)
    - CR: crossover probability
    - population_size: number of individuals
    """

    def __init__(
        self,
        population_size: int = 50,
        dimensions: int = 10,
        F: float = 0.8,
        CR: float = 0.9,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.population_size = population_size
        self.dimensions = dimensions
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.bounds = bounds or (-5.0, 5.0)
        self.population: List[DEIndividual] = []
        self.generation = 0
        self.best_individual: Optional[DEIndividual] = None

    def initialize(self):
        """Initialize random population."""
        self.population = []
        for i in range(self.population_size):
            genome = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
            individual = DEIndividual(genome=genome, generation=0)
            self.population.append(individual)

    def evaluate(self, fitness_fn: callable) -> float:
        """Evaluate entire population with fitness function."""
        best_fitness = float("-inf")
        for ind in self.population:
            ind.fitness = fitness_fn(ind.genome)
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                self.best_individual = ind
        return best_fitness

    def _select_donors(self, target_idx: int) -> Tuple[int, int, int]:
        """Select 3 distinct random donors different from target."""
        candidates = [i for i in range(self.population_size) if i != target_idx]
        donors = random.sample(candidates, 3)
        return donors[0], donors[1], donors[2]

    def _mutate(self, target_idx: int) -> np.ndarray:
        """Create mutant vector using DE/rand/1 strategy."""
        a, b, c = self._select_donors(target_idx)
        x_a = self.population[a].genome
        x_b = self.population[b].genome
        x_c = self.population[c].genome

        mutant = x_a + self.F * (x_b - x_c)

        # Clamp to bounds
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover between target and mutant."""
        trial = np.copy(target)
        j_rand = random.randint(0, self.dimensions - 1)

        for j in range(self.dimensions):
            if random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def evolve(self, fitness_fn: Optional[callable] = None) -> float:
        """
        One generation of DE evolution.
        Returns best fitness.
        """
        new_population = []

        for i, target in enumerate(self.population):
            # Mutation
            mutant = self._mutate(i)

            # Crossover
            trial_genome = self._crossover(target.genome, mutant)

            # Selection: evaluate trial and keep if better
            if fitness_fn:
                trial_fitness = fitness_fn(trial_genome)
                if trial_fitness >= target.fitness:
                    new_ind = DEIndividual(
                        genome=trial_genome,
                        fitness=trial_fitness,
                        generation=self.generation + 1,
                    )
                    new_population.append(new_ind)
                else:
                    new_population.append(target)
            else:
                # No fitness function: keep mutant regardless
                new_population.append(
                    DEIndividual(
                        genome=trial_genome,
                        generation=self.generation + 1,
                    )
                )

        self.population = new_population
        self.generation += 1

        # Update best
        if fitness_fn:
            self.evaluate(fitness_fn)

        return self.best_individual.fitness if self.best_individual else 0.0

    def get_best(self) -> Optional[DEIndividual]:
        """Get best individual."""
        return self.best_individual

    def get_diversity(self) -> float:
        """Compute population diversity (average pairwise distance)."""
        if not self.population:
            return 0.0
        total_dist = 0.0
        count = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.linalg.norm(
                    self.population[i].genome - self.population[j].genome
                )
                total_dist += dist
                count += 1
        return total_dist / count if count > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        fitnesses = [ind.fitness for ind in self.population]
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0,
            "worst_fitness": min(fitnesses) if fitnesses else 0,
            "diversity": self.get_diversity(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
            "best": self.best_individual.to_dict() if self.best_individual else None,
            "population": [ind.to_dict() for ind in self.population[:5]],  # Sample
        }
