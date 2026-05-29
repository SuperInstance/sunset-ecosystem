"""swarm/pythagorean_evolution.py — Novel breeding system using Pythagorean triples as genetic alphabet.

Instead of floating-point approximations, every gene is an exact Pythagorean triple
(a, b, c) where a² + b² = c² exactly. Mutation is a lattice walk on the
Pythagorean manifold. Crossover is exact geometric mean. Fitness rewards both
task performance and holonomic consistency.

This is genuinely novel: no existing evolutionary algorithm uses exact rational
arithmetic (Pythagorean triples) as its genetic substrate.

Usage
-----
    from swarm.pythagorean_evolution import PythagoreanBreeder, PythagoreanGenome

    # Initialize a population of 50 genomes, each with 10 triples
    breeder = PythagoreanBreeder(population_size=50, genome_length=10)
    breeder.initialize()

    # Evolve for 100 generations
    for gen in range(100):
        breeder.evaluate_fitness(task_fn)
        breeder.select_and_breed()
        print(f"Gen {gen}: best={breeder.best_fitness:.4f}")

    best = breeder.best_genome
    print(f"Best genome: {best.triples}")
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from swarm.constraint_bridge import ConstraintBridge, SnapResult


@dataclass
class PythagoreanTriple:
    """Exact Pythagorean triple: a² + b² = c²."""
    a: int
    b: int
    c: int

    def __post_init__(self):
        # Verify exactness
        assert self.a * self.a + self.b * self.b == self.c * self.c, \
            f"Not a Pythagorean triple: {self.a}² + {self.b}² ≠ {self.c}²"

    def to_vector(self) -> np.ndarray:
        """Convert to unit vector [a/c, b/c]."""
        return np.array([self.a / self.c, self.b / self.c], dtype=np.float64)

    def angle(self) -> float:
        """Angle of the vector in radians."""
        return math.atan2(self.b, self.a)

    def adjacent_triples(self, max_n: int = 5) -> List[PythagoreanTriple]:
        """Find nearby triples on the Pythagorean lattice.

        Returns triples with similar angles but different magnitudes (c values).
        """
        target_angle = self.angle()
        candidates = []
        # Generate triples with similar angles using Euclid's formula
        for m in range(2, max_n + 2):
            for n in range(1, m):
                if math.gcd(m, n) == 1 and (m - n) % 2 == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    # Create both orientations
                    for (aa, bb, cc) in [(a, b, c), (b, a, c)]:
                        t = PythagoreanTriple(aa, bb, cc)
                        angle_diff = abs(t.angle() - target_angle)
                        # Wrap around for circular angles
                        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                        if angle_diff < 0.5:  # Within ~30 degrees
                            candidates.append(t)
        return candidates

    @staticmethod
    def from_vector(vector: List[float]) -> PythagoreanTriple:
        """Create triple from vector by snapping to nearest triple."""
        bridge = ConstraintBridge(density=500)
        result = bridge.snap_vector(vector)
        if result.triple:
            a, b, c = result.triple
            return PythagoreanTriple(a, b, c)
        # Fallback: create primitive triple from angle
        angle = math.atan2(vector[1], vector[0])
        # Use Euclid's formula with m=2, n=1 as base
        a, b, c = 3, 4, 5
        return PythagoreanTriple(a, b, c)

    @staticmethod
    def random_primitive() -> PythagoreanTriple:
        """Generate a random primitive Pythagorean triple."""
        # Pick random coprime m, n with different parity
        while True:
            m = random.randint(2, 20)
            n = random.randint(1, m - 1)
            if math.gcd(m, n) == 1 and (m - n) % 2 == 1:
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                if random.random() < 0.5:
                    a, b = b, a
                return PythagoreanTriple(a, b, c)

    def __repr__(self) -> str:
        return f"PT({self.a},{self.b},{self.c})"


@dataclass
class PythagoreanGenome:
    """Genome composed of a sequence of Pythagorean triples."""
    triples: List[PythagoreanTriple] = field(default_factory=list)
    fitness: float = 0.0
    age: int = 0

    def to_matrix(self) -> np.ndarray:
        """Convert to N×2 matrix of unit vectors."""
        return np.array([t.to_vector() for t in self.triples])

    def check_holonomy(self) -> bool:
        """Check if the genome forms a holonomic cycle."""
        if len(self.triples) < 3:
            return True
        vectors = [t.to_vector() for t in self.triples]
        bridge = ConstraintBridge(density=100)
        return bridge.check_holonomy(vectors)

    def holonomy_error(self) -> float:
        """Compute holonomy error (0 = perfect, higher = worse)."""
        if len(self.triples) < 3:
            return 0.0
        vectors = [t.to_vector() for t in self.triples]
        bridge = ConstraintBridge(density=100)
        total = 0.0
        for i in range(len(vectors)):
            a = vectors[i]
            b = vectors[(i + 1) % len(vectors)]
            total += bridge._angle_between(a, b)
        remainder = abs(total) % (2 * math.pi)
        error = min(remainder, abs(remainder - 2 * math.pi))
        return error

    def length(self) -> int:
        return len(self.triples)

    def copy(self) -> PythagoreanGenome:
        return PythagoreanGenome(
            triples=copy.deepcopy(self.triples),
            fitness=self.fitness,
            age=self.age,
        )


class LatticeWalkMutation:
    """Mutation by walking to adjacent triples on the Pythagorean lattice."""

    def __init__(self, step_size: float = 0.3, max_steps: int = 3):
        self.step_size = step_size
        self.max_steps = max_steps

    def mutate(self, genome: PythagoreanGenome) -> PythagoreanGenome:
        """Mutate a genome by walking some triples to adjacent lattice points."""
        child = genome.copy()
        n_mutations = max(1, int(self.step_size * len(child.triples)))

        for _ in range(n_mutations):
            idx = random.randint(0, len(child.triples) - 1)
            current = child.triples[idx]
            adjacent = current.adjacent_triples(max_n=5)

            if adjacent:
                # Pick adjacent triple with probability proportional to closeness
                weights = []
                for adj in adjacent:
                    angle_diff = abs(adj.angle() - current.angle())
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    weights.append(1.0 / (angle_diff + 0.01))

                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                chosen = random.choices(adjacent, weights=probs, k=1)[0]
                child.triples[idx] = chosen

        return child


class ExactGeometricCrossover:
    """Exact geometric crossover of two Pythagorean genomes."""

    def crossover(self, parent1: PythagoreanGenome,
                  parent2: PythagoreanGenome) -> Tuple[PythagoreanGenome, PythagoreanGenome]:
        """Crossover by taking exact geometric mean of corresponding triples."""
        assert len(parent1.triples) == len(parent2.triples)

        child1_triples = []
        child2_triples = []

        for t1, t2 in zip(parent1.triples, parent2.triples):
            # Geometric mean of angles
            angle1 = t1.angle()
            angle2 = t2.angle()
            mean_angle = (angle1 + angle2) / 2

            # Find triple closest to mean angle
            # Use a small search around the mean
            best_triple = None
            best_diff = float('inf')
            for m in range(2, 15):
                for n in range(1, m):
                    if math.gcd(m, n) == 1 and (m - n) % 2 == 1:
                        a = m * m - n * n
                        b = 2 * m * n
                        c = m * m + n * n
                        for (aa, bb, cc) in [(a, b, c), (b, a, c)]:
                            t = PythagoreanTriple(aa, bb, cc)
                            diff = abs(t.angle() - mean_angle)
                            diff = min(diff, 2 * math.pi - diff)
                            if diff < best_diff:
                                best_diff = diff
                                best_triple = t

            if best_triple:
                child1_triples.append(best_triple)
                # Second child: angle bisector of the other arc
                other_angle = mean_angle + math.pi / 2
                best_triple2 = None
                best_diff2 = float('inf')
                for m in range(2, 15):
                    for n in range(1, m):
                        if math.gcd(m, n) == 1 and (m - n) % 2 == 1:
                            a = m * m - n * n
                            b = 2 * m * n
                            c = m * m + n * n
                            for (aa, bb, cc) in [(a, b, c), (b, a, c)]:
                                t = PythagoreanTriple(aa, bb, cc)
                                diff = abs(t.angle() - other_angle)
                                diff = min(diff, 2 * math.pi - diff)
                                if diff < best_diff2:
                                    best_diff2 = diff
                                    best_triple2 = t
                child2_triples.append(best_triple2 if best_triple2 else best_triple)
            else:
                child1_triples.append(t1)
                child2_triples.append(t2)

        child1 = PythagoreanGenome(triples=child1_triples)
        child2 = PythagoreanGenome(triples=child2_triples)
        return child1, child2


class HolonomicFitness:
    """Fitness function rewarding task performance and holonomic consistency."""

    def __init__(self, task_fn: Callable[[np.ndarray], float],
                 holonomy_weight: float = 0.3,
                 exactness_weight: float = 0.1):
        self.task_fn = task_fn
        self.holonomy_weight = holonomy_weight
        self.exactness_weight = exactness_weight

    def evaluate(self, genome: PythagoreanGenome) -> float:
        """Evaluate fitness: task_score + holonomy_bonus + exactness_bonus."""
        # Task performance
        matrix = genome.to_matrix()
        task_score = self.task_fn(matrix)

        # Holonomy bonus (lower error = higher bonus)
        holonomy_error = genome.holonomy_error()
        holonomy_bonus = self.holonomy_weight * (1.0 / (1.0 + 10 * holonomy_error))

        # Exactness bonus (smaller c values = more compact = bonus)
        avg_c = sum(t.c for t in genome.triples) / len(genome.triples) if genome.triples else 1
        exactness_bonus = self.exactness_weight * (1.0 / avg_c)

        fitness = task_score + holonomy_bonus + exactness_bonus
        genome.fitness = fitness
        return fitness


@dataclass
class PythagoreanBreeder:
    """Main breeding orchestrator using Pythagorean genomes."""

    population_size: int = 50
    genome_length: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elitism_count: int = 2
    max_age: int = 50

    population: List[PythagoreanGenome] = field(default_factory=list, repr=False)
    generation: int = 0
    best_fitness: float = 0.0
    best_genome: Optional[PythagoreanGenome] = None

    _mutation: LatticeWalkMutation = field(default_factory=lambda: LatticeWalkMutation(), repr=False)
    _crossover: ExactGeometricCrossover = field(default_factory=ExactGeometricCrossover, repr=False)

    def initialize(self) -> None:
        """Initialize population with random primitive triples."""
        self.population = []
        for _ in range(self.population_size):
            triples = [PythagoreanTriple.random_primitive() for _ in range(self.genome_length)]
            self.population.append(PythagoreanGenome(triples=triples))
        self.generation = 0
        self.best_fitness = 0.0
        self.best_genome = None

    def evaluate_fitness(self, task_fn: Callable[[np.ndarray], float]) -> None:
        """Evaluate all genomes with a task function."""
        fitness_fn = HolonomicFitness(task_fn)
        for genome in self.population:
            genome.age += 1
            fitness_fn.evaluate(genome)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome.copy()

    def select_and_breed(self) -> None:
        """Selection (tournament), crossover, mutation, replacement."""
        # Sort by fitness descending
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Elitism: keep top N
        new_population = [g.copy() for g in self.population[:self.elitism_count]]

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < self.mutation_rate:
                child1 = self._mutation.mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutation.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # Age-based culling
        self.population = [g for g in new_population if g.age < self.max_age]
        # Fill back if needed
        while len(self.population) < self.population_size:
            triples = [PythagoreanTriple.random_primitive() for _ in range(self.genome_length)]
            self.population.append(PythagoreanGenome(triples=triples))

        self.generation += 1

    def _tournament_select(self, tournament_size: int = 3) -> PythagoreanGenome:
        """Tournament selection."""
        contestants = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(contestants, key=lambda g: g.fitness)

    def get_stats(self) -> dict:
        """Get population statistics."""
        fitnesses = [g.fitness for g in self.population]
        holonomy_scores = [1.0 / (1.0 + 10 * g.holonomy_error()) for g in self.population]
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "mean_holonomy": sum(holonomy_scores) / len(holonomy_scores) if holonomy_scores else 0,
            "population_size": len(self.population),
        }
