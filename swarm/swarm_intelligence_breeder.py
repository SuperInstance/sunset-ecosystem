"""
Swarm Intelligence Breeder

Uses swarm intelligence algorithms (Particle Swarm Optimization + Ant Colony
Optimization) to guide the population toward high-fitness regions.

Each "particle" (genome) has:
- Position: current genome values
- Velocity: mutation direction and magnitude
- Personal best: best position found so far
- Neighborhood best: best position found by neighbors

The swarm collectively searches the fitness landscape, with particles
attracted to both their own best and the swarm's best.

Key innovations:
- PSO velocity update: cognitive (personal) + social (swarm) components
- ACO pheromone trails: mark successful paths through genome space
- Dynamic neighborhood topology: small-world networks for information spread
- Adaptive inertia: balance exploration vs exploitation over time

References:
- Kennedy & Eberhart (1995) - Particle Swarm Optimization
- Dorigo & Stützle (2004) - Ant Colony Optimization
- Mendes et al. (2004) - Fully informed particle swarm
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Particle:
    """A particle in the swarm."""

    genome: Dict[str, float]
    velocity: Dict[str, float]
    fitness: float = 0.0
    # Personal best
    best_genome: Optional[Dict[str, float]] = None
    best_fitness: float = -float("inf")
    # Particle ID
    id: int = 0

    def update_personal_best(self):
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_genome = self.genome.copy()


@dataclass
class PheromoneTrail:
    """Pheromone trail on a path between two genome states."""

    source: Dict[str, float]
    target: Dict[str, float]
    strength: float = 1.0
    last_update: float = 0.0

    def decay(self, rate: float = 0.1):
        self.strength *= 1 - rate


class SwarmTopology:
    """
    Dynamic neighborhood topology for the swarm.
    Small-world network: mostly local connections + random long-range links.
    """

    def __init__(
        self, n_particles: int, k_neighbors: int = 3, rewire_prob: float = 0.2
    ):
        self.n_particles = n_particles
        self.k_neighbors = k_neighbors
        self.rewire_prob = rewire_prob
        self.neighbors: Dict[int, List[int]] = {}
        self._build_ring_lattice()
        self._rewire()

    def _build_ring_lattice(self):
        """Build a ring lattice where each node connects to k nearest neighbors."""
        for i in range(self.n_particles):
            self.neighbors[i] = []
            for j in range(1, self.k_neighbors + 1):
                left = (i - j) % self.n_particles
                right = (i + j) % self.n_particles
                self.neighbors[i].append(left)
                self.neighbors[i].append(right)
            # Remove duplicates
            self.neighbors[i] = list(set(self.neighbors[i]))

    def _rewire(self):
        """Rewire edges with probability rewire_prob (Watts-Strogatz)."""
        for i in range(self.n_particles):
            for j_idx, j in enumerate(list(self.neighbors[i])):
                if random.random() < self.rewire_prob:
                    # Rewire to random node
                    new_neighbor = random.randint(0, self.n_particles - 1)
                    if new_neighbor != i and new_neighbor not in self.neighbors[i]:
                        self.neighbors[i][j_idx] = new_neighbor

    def get_neighborhood_best(
        self, particles: List[Particle], particle_idx: int
    ) -> Optional[Particle]:
        """Get the best particle in the neighborhood."""
        neighbor_indices = self.neighbors.get(particle_idx, [])
        if not neighbor_indices:
            return None

        best = None
        best_fitness = -float("inf")
        for idx in neighbor_indices:
            if 0 <= idx < len(particles):
                if particles[idx].best_fitness > best_fitness:
                    best_fitness = particles[idx].best_fitness
                    best = particles[idx]
        return best

    def get_clustering_coefficient(self, idx: int) -> float:
        """Local clustering coefficient for a particle."""
        neighbors = self.neighbors.get(idx, [])
        if len(neighbors) < 2:
            return 0.0

        triangles = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1 :]:
                n1_neighbors = set(self.neighbors.get(n1, []))
                if n2 in n1_neighbors:
                    triangles += 1

        max_edges = len(neighbors) * (len(neighbors) - 1) / 2
        return triangles / max_edges if max_edges > 0 else 0.0


class SwarmIntelligenceBreeder:
    """
    Breeding daemon using swarm intelligence.

    Combines PSO velocity updates with evolutionary selection pressure.
    """

    def __init__(
        self,
        population_size: int = 50,
        w_inertia: float = 0.7,  # Inertia weight
        c_cognitive: float = 1.5,  # Cognitive coefficient
        c_social: float = 1.5,  # Social coefficient
        max_velocity: float = 0.5,
        topology_neighbors: int = 3,
    ):
        self.population_size = population_size
        self.w_inertia = w_inertia
        self.c_cognitive = c_cognitive
        self.c_social = c_social
        self.max_velocity = max_velocity

        self.topology = SwarmTopology(population_size, topology_neighbors)
        self.particles: List[Particle] = []
        self.generation = 0
        self.pheromones: List[PheromoneTrail] = []

        # Global best
        self.global_best_genome: Optional[Dict[str, float]] = None
        self.global_best_fitness: float = -float("inf")

    def initialize(
        self,
        task_fn: Callable[[Dict[str, float]], Any],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """Initialize swarm with random particles."""
        self.particles = []
        for i in range(self.population_size):
            genome = {}
            velocity = {}

            if bounds:
                for gene, (low, high) in bounds.items():
                    genome[gene] = random.uniform(low, high)
                    velocity[gene] = random.uniform(
                        -self.max_velocity, self.max_velocity
                    )
            else:
                for gene in range(10):  # Default 10 genes
                    genome[f"gene_{gene}"] = random.gauss(0, 1)
                    velocity[f"gene_{gene}"] = random.gauss(0, self.max_velocity)

            result = task_fn(genome)
            fitness = (
                result.get("fitness", 0.0)
                if isinstance(result, dict)
                else float(result)
            )

            particle = Particle(
                genome=genome,
                velocity=velocity,
                fitness=fitness,
                best_genome=genome.copy(),
                best_fitness=fitness,
                id=i,
            )
            self.particles.append(particle)

            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_genome = genome.copy()

    def _update_velocity(
        self, particle: Particle, neighborhood_best: Optional[Particle]
    ):
        """PSO velocity update equation."""
        for gene in particle.velocity:
            if gene not in particle.genome:
                continue

            # Inertia component
            inertia = self.w_inertia * particle.velocity[gene]

            # Cognitive component: pull toward personal best
            if particle.best_genome and gene in particle.best_genome:
                cognitive = (
                    self.c_cognitive
                    * random.random()
                    * (particle.best_genome[gene] - particle.genome[gene])
                )
            else:
                cognitive = 0

            # Social component: pull toward neighborhood best
            if (
                neighborhood_best
                and neighborhood_best.best_genome
                and gene in neighborhood_best.best_genome
            ):
                social = (
                    self.c_social
                    * random.random()
                    * (neighborhood_best.best_genome[gene] - particle.genome[gene])
                )
            else:
                social = 0

            # Pheromone component: bias toward trails
            pheromone_bias = self._pheromone_bias(particle.genome, gene)

            new_velocity = inertia + cognitive + social + pheromone_bias
            # Clamp velocity
            particle.velocity[gene] = max(
                -self.max_velocity, min(self.max_velocity, new_velocity)
            )

    def _pheromone_bias(self, genome: Dict[str, float], gene: str) -> float:
        """Compute bias from pheromone trails for a gene."""
        bias = 0.0
        for trail in self.pheromones:
            if trail.strength < 0.1:
                continue
            # If this gene is near a pheromone source, add bias
            if gene in trail.source and gene in trail.target:
                source_val = trail.source[gene]
                target_val = trail.target[gene]
                current_val = genome.get(gene, 0)
                # Bias toward target if current is closer to source
                if abs(current_val - source_val) < abs(current_val - target_val):
                    bias += trail.strength * (target_val - current_val) * 0.1
        return bias

    def _update_position(self, particle: Particle):
        """Update position (genome) using velocity."""
        for gene in particle.genome:
            if gene in particle.velocity:
                particle.genome[gene] += particle.velocity[gene]

    def _deposit_pheromone(
        self,
        old_genome: Dict[str, float],
        new_genome: Dict[str, float],
        fitness_improvement: float,
    ):
        """Deposit pheromone if fitness improved."""
        if fitness_improvement > 0:
            trail = PheromoneTrail(
                source=old_genome,
                target=new_genome,
                strength=min(1.0, fitness_improvement),
                last_update=time.time() if "time" in dir() else 0,
            )
            self.pheromones.append(trail)
            # Keep only top 100 trails
            self.pheromones.sort(key=lambda t: t.strength, reverse=True)
            self.pheromones = self.pheromones[:100]

    def _decay_pheromones(self, rate: float = 0.05):
        """Decay all pheromone trails."""
        for trail in self.pheromones:
            trail.decay(rate)
        self.pheromones = [t for t in self.pheromones if t.strength > 0.01]

    def breed_generation(
        self, task_fn: Callable[[Dict[str, float]], Any]
    ) -> List[Tuple[Dict[str, float], float]]:
        """Run one generation of swarm breeding."""
        self.generation += 1
        self._decay_pheromones()

        for i, particle in enumerate(self.particles):
            old_genome = particle.genome.copy()
            old_fitness = particle.fitness

            # Get neighborhood best
            neighborhood_best = self.topology.get_neighborhood_best(self.particles, i)

            # Update velocity and position
            self._update_velocity(particle, neighborhood_best)
            self._update_position(particle)

            # Evaluate new position
            result = task_fn(particle.genome)
            fitness = (
                result.get("fitness", 0.0)
                if isinstance(result, dict)
                else float(result)
            )
            particle.fitness = fitness

            # Update personal best
            particle.update_personal_best()

            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_genome = particle.genome.copy()

            # Deposit pheromone if improved
            improvement = fitness - old_fitness
            self._deposit_pheromone(old_genome, particle.genome, improvement)

        # Return population as (genome, fitness) tuples
        return [(p.genome, p.fitness) for p in self.particles]

    def get_swarm_stats(self) -> Dict:
        """Get statistics about the swarm."""
        fitnesses = [p.fitness for p in self.particles]
        personal_bests = [p.best_fitness for p in self.particles]

        return {
            "generation": self.generation,
            "n_particles": len(self.particles),
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0,
            "best_fitness": self.global_best_fitness,
            "avg_personal_best": np.mean(personal_bests) if personal_bests else 0,
            "avg_velocity": np.mean(
                [np.mean(list(p.velocity.values())) for p in self.particles]
            ),
            "diversity": self._compute_diversity(),
            "pheromone_trails": len(self.pheromones),
            "avg_clustering": np.mean(
                [
                    self.topology.get_clustering_coefficient(i)
                    for i in range(len(self.particles))
                ]
            )
            if self.particles
            else 0,
        }

    def _compute_diversity(self) -> float:
        """Compute swarm diversity as average pairwise distance."""
        if len(self.particles) < 2:
            return 0.0

        distances = []
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i + 1 :]:
                # Compute Euclidean distance between genomes
                common_keys = set(p1.genome.keys()) & set(p2.genome.keys())
                if not common_keys:
                    continue
                dist = np.sqrt(
                    sum((p1.genome[k] - p2.genome[k]) ** 2 for k in common_keys)
                )
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def get_particle_states(self) -> List[Dict]:
        """Get detailed state of all particles."""
        return [
            {
                "id": p.id,
                "fitness": p.fitness,
                "best_fitness": p.best_fitness,
                "velocity_magnitude": np.sqrt(sum(v**2 for v in p.velocity.values())),
                "neighbors": self.topology.neighbors.get(p.id, []),
            }
            for p in self.particles
        ]
