"""swarm/adversarial_arena.py — Adversarial co-evolutionary arena.

Two populations compete:
  - Solvers: evolve solutions to a task
  - Testers: evolve test cases to break those solutions

Game-theoretic equilibrium: solvers get stronger, testers get harder.
This breeds robustness — solutions that survive adversarial testing.

Usage
-----
    from swarm.adversarial_arena import AdversarialArena, SolverGenome, TesterGenome

    def task_fn(solution, test_case):
        # Return: (solver_fitness, tester_fitness)
        # Solver wants to pass test, tester wants to break solver
        solver_score = evaluate_solution(solution, test_case)
        tester_score = 1.0 - solver_score  # Zero-sum
        return solver_score, tester_score

    arena = AdversarialArena(
        solver_pop_size=50, tester_pop_size=30,
        n_interactions_per_gen=10,
    )
    arena.initialize(solver_factory, tester_factory)

    for gen in range(100):
        arena.evaluate(task_fn)
        arena.breed()
        print(f"Gen {gen}: solver_best={arena.solver_best:.3f}, "
              f"tester_best={arena.tester_best:.3f}")
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SolverGenome:
    """A solution genome."""

    vector: np.ndarray
    fitness: float = 0.0
    robustness: float = 0.0  # Average fitness across all tests
    age: int = 0

    def copy(self) -> SolverGenome:
        return SolverGenome(
            vector=self.vector.copy(),
            fitness=self.fitness,
            robustness=self.robustness,
            age=self.age,
        )

    @staticmethod
    def random(
        dimension: int, bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> SolverGenome:
        low, high = bounds
        return SolverGenome(vector=np.random.uniform(low, high, dimension))


@dataclass
class TesterGenome:
    """A test case genome — evolves to break solvers."""

    vector: np.ndarray
    fitness: float = 0.0
    difficulty: float = 0.0  # Average solver score (lower = harder)
    age: int = 0

    def copy(self) -> TesterGenome:
        return TesterGenome(
            vector=self.vector.copy(),
            fitness=self.fitness,
            difficulty=self.difficulty,
            age=self.age,
        )

    @staticmethod
    def random(
        dimension: int, bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> TesterGenome:
        low, high = bounds
        return TesterGenome(vector=np.random.uniform(low, high, dimension))


class SolverMutation:
    """Mutation for solvers."""

    def __init__(self, rate: float = 0.1, strength: float = 0.5):
        self.rate = rate
        self.strength = strength

    def mutate(self, genome: SolverGenome) -> SolverGenome:
        child = genome.copy()
        mask = np.random.random(len(child.vector)) < self.rate
        noise = np.random.normal(0, self.strength, len(child.vector))
        child.vector += mask * noise
        return child


class TesterMutation:
    """Mutation for testers."""

    def __init__(self, rate: float = 0.1, strength: float = 0.5):
        self.rate = rate
        self.strength = strength

    def mutate(self, genome: TesterGenome) -> TesterGenome:
        child = genome.copy()
        mask = np.random.random(len(child.vector)) < self.rate
        noise = np.random.normal(0, self.strength, len(child.vector))
        child.vector += mask * noise
        return child


class SolverCrossover:
    """Crossover for solvers."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def crossover(
        self, p1: SolverGenome, p2: SolverGenome
    ) -> Tuple[SolverGenome, SolverGenome]:
        blend = np.random.random(len(p1.vector)) < 0.5
        c1_vec = np.where(blend, p1.vector, p2.vector)
        c2_vec = np.where(blend, p2.vector, p1.vector)
        # Add interpolation noise
        noise = np.random.normal(0, 0.1, len(p1.vector))
        c1_vec += noise
        c2_vec -= noise
        return SolverGenome(vector=c1_vec), SolverGenome(vector=c2_vec)


class TesterCrossover:
    """Crossover for testers."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def crossover(
        self, p1: TesterGenome, p2: TesterGenome
    ) -> Tuple[TesterGenome, TesterGenome]:
        blend = np.random.random(len(p1.vector)) < 0.5
        c1_vec = np.where(blend, p1.vector, p2.vector)
        c2_vec = np.where(blend, p2.vector, p1.vector)
        noise = np.random.normal(0, 0.1, len(p1.vector))
        c1_vec += noise
        c2_vec -= noise
        return TesterGenome(vector=c1_vec), TesterGenome(vector=c2_vec)


@dataclass
class AdversarialArena:
    """Adversarial co-evolutionary arena with two competing populations."""

    solver_pop_size: int = 50
    tester_pop_size: int = 30
    solver_dim: int = 10
    tester_dim: int = 10
    solver_bounds: Tuple[float, float] = (-1.0, 1.0)
    tester_bounds: Tuple[float, float] = (-1.0, 1.0)
    n_interactions_per_gen: int = 10
    solver_mutation_rate: float = 0.1
    tester_mutation_rate: float = 0.1
    solver_crossover_rate: float = 0.7
    tester_crossover_rate: float = 0.7
    elitism_count: int = 2
    max_age: int = 30

    solver_pop: List[SolverGenome] = field(default_factory=list, repr=False)
    tester_pop: List[TesterGenome] = field(default_factory=list, repr=False)

    solver_best: float = 0.0
    tester_best: float = 0.0
    solver_best_genome: Optional[SolverGenome] = None
    tester_best_genome: Optional[TesterGenome] = None

    generation: int = 0

    _solver_mut: SolverMutation = field(
        default_factory=lambda: SolverMutation(), repr=False
    )
    _tester_mut: TesterMutation = field(
        default_factory=lambda: TesterMutation(), repr=False
    )
    _solver_xo: SolverCrossover = field(
        default_factory=lambda: SolverCrossover(), repr=False
    )
    _tester_xo: TesterCrossover = field(
        default_factory=lambda: TesterCrossover(), repr=False
    )

    def initialize(
        self,
        solver_factory: Optional[Callable[[], SolverGenome]] = None,
        tester_factory: Optional[Callable[[], TesterGenome]] = None,
    ) -> None:
        """Initialize both populations."""
        if solver_factory:
            self.solver_pop = [solver_factory() for _ in range(self.solver_pop_size)]
        else:
            self.solver_pop = [
                SolverGenome.random(self.solver_dim, self.solver_bounds)
                for _ in range(self.solver_pop_size)
            ]

        if tester_factory:
            self.tester_pop = [tester_factory() for _ in range(self.tester_pop_size)]
        else:
            self.tester_pop = [
                TesterGenome.random(self.tester_dim, self.tester_bounds)
                for _ in range(self.tester_pop_size)
            ]

        self.solver_best = 0.0
        self.tester_best = 0.0
        self.solver_best_genome = None
        self.tester_best_genome = None
        self.generation = 0

    def evaluate(
        self,
        interaction_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
    ) -> None:
        """Evaluate all solver-tester interactions.

        Each solver interacts with n_interactions_per_gen randomly selected testers.
        Solver fitness = average score against all tests (robustness).
        Tester fitness = 1 - average solver score (difficulty).
        """
        # Reset fitnesses
        for s in self.solver_pop:
            s.fitness = 0.0
            s.robustness = 0.0
        for t in self.tester_pop:
            t.fitness = 0.0
            t.difficulty = 0.0

        # Track interaction counts
        solver_interactions = [0] * len(self.solver_pop)
        tester_interactions = [0] * len(self.tester_pop)

        # Each solver interacts with n testers
        for s_idx, solver in enumerate(self.solver_pop):
            testers = random.sample(
                self.tester_pop, min(self.n_interactions_per_gen, len(self.tester_pop))
            )
            for tester in testers:
                # Use identity comparison since TesterGenome contains numpy arrays
                t_idx = next(i for i, t in enumerate(self.tester_pop) if t is tester)
                solver_score, tester_score = interaction_fn(
                    solver.vector, tester.vector
                )

                solver.fitness += solver_score
                solver.robustness += solver_score
                tester.fitness += tester_score
                tester.difficulty += solver_score  # Track solver's struggle

                solver_interactions[s_idx] += 1
                tester_interactions[t_idx] += 1

        # Normalize
        for s, count in zip(self.solver_pop, solver_interactions):
            if count > 0:
                s.fitness /= count
                s.robustness /= count
                s.age += 1

        for t, count in zip(self.tester_pop, tester_interactions):
            if count > 0:
                t.fitness /= count
                t.difficulty /= count
                t.age += 1

        # Update best
        best_solver = max(self.solver_pop, key=lambda s: s.fitness)
        if best_solver.fitness > self.solver_best:
            self.solver_best = best_solver.fitness
            self.solver_best_genome = best_solver.copy()

        best_tester = max(self.tester_pop, key=lambda t: t.fitness)
        if best_tester.fitness > self.tester_best:
            self.tester_best = best_tester.fitness
            self.tester_best_genome = best_tester.copy()

    def breed(self) -> None:
        """Breed both populations independently."""
        self._breed_solvers()
        self._breed_testers()
        self.generation += 1

    def _breed_solvers(self) -> None:
        """Breed solver population."""
        self.solver_pop.sort(key=lambda s: s.fitness, reverse=True)
        new_pop = [s.copy() for s in self.solver_pop[: self.elitism_count]]

        while len(new_pop) < self.solver_pop_size:
            p1 = self._tournament_select_solver()
            p2 = self._tournament_select_solver()

            if random.random() < self.solver_crossover_rate:
                c1, c2 = self._solver_xo.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if random.random() < self.solver_mutation_rate:
                c1 = self._solver_mut.mutate(c1)
            if random.random() < self.solver_mutation_rate:
                c2 = self._solver_mut.mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < self.solver_pop_size:
                new_pop.append(c2)

        # Age culling
        self.solver_pop = [s for s in new_pop if s.age < self.max_age]
        while len(self.solver_pop) < self.solver_pop_size:
            self.solver_pop.append(
                SolverGenome.random(self.solver_dim, self.solver_bounds)
            )

    def _breed_testers(self) -> None:
        """Breed tester population."""
        self.tester_pop.sort(key=lambda t: t.fitness, reverse=True)
        new_pop = [t.copy() for t in self.tester_pop[: self.elitism_count]]

        while len(new_pop) < self.tester_pop_size:
            p1 = self._tournament_select_tester()
            p2 = self._tournament_select_tester()

            if random.random() < self.tester_crossover_rate:
                c1, c2 = self._tester_xo.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if random.random() < self.tester_mutation_rate:
                c1 = self._tester_mut.mutate(c1)
            if random.random() < self.tester_mutation_rate:
                c2 = self._tester_mut.mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < self.tester_pop_size:
                new_pop.append(c2)

        # Age culling
        self.tester_pop = [t for t in new_pop if t.age < self.max_age]
        while len(self.tester_pop) < self.tester_pop_size:
            self.tester_pop.append(
                TesterGenome.random(self.tester_dim, self.tester_bounds)
            )

    def _tournament_select_solver(self, k: int = 3) -> SolverGenome:
        contestants = random.sample(self.solver_pop, min(k, len(self.solver_pop)))
        return max(contestants, key=lambda s: s.fitness)

    def _tournament_select_tester(self, k: int = 3) -> TesterGenome:
        contestants = random.sample(self.tester_pop, min(k, len(self.tester_pop)))
        return max(contestants, key=lambda t: t.fitness)

    def get_solver_stats(self) -> Dict:
        """Get statistics for solver population."""
        fitnesses = [s.fitness for s in self.solver_pop]
        robustness = [s.robustness for s in self.solver_pop]
        return {
            "generation": self.generation,
            "best_fitness": self.solver_best,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "mean_robustness": sum(robustness) / len(robustness) if robustness else 0,
            "pop_size": len(self.solver_pop),
        }

    def get_tester_stats(self) -> Dict:
        """Get statistics for tester population."""
        fitnesses = [t.fitness for t in self.tester_pop]
        difficulties = [t.difficulty for t in self.tester_pop]
        return {
            "generation": self.generation,
            "best_fitness": self.tester_best,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "mean_difficulty": sum(difficulties) / len(difficulties)
            if difficulties
            else 0,
            "pop_size": len(self.tester_pop),
        }

    def get_coevolution_stats(self) -> Dict:
        """Get co-evolution dynamics statistics."""
        return {
            "generation": self.generation,
            "solver_best": self.solver_best,
            "tester_best": self.tester_best,
            "arms_race_index": self.solver_best - (1.0 - self.tester_best),
            "solver_stats": self.get_solver_stats(),
            "tester_stats": self.get_tester_stats(),
        }
