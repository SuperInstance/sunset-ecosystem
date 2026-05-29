"""
Tests for Swarm Intelligence Breeder.

Covers: Particle, PheromoneTrail, SwarmTopology, SwarmIntelligenceBreeder.
"""

import random

import numpy as np
import pytest

from swarm.swarm_intelligence_breeder import (
    Particle,
    PheromoneTrail,
    SwarmTopology,
    SwarmIntelligenceBreeder,
)


class TestSwarmTopology:
    def test_ring_lattice(self):
        t = SwarmTopology(n_particles=10, k_neighbors=2)
        for i in range(10):
            assert len(t.neighbors[i]) >= 2

    def test_rewire(self):
        t = SwarmTopology(n_particles=10, k_neighbors=2, rewire_prob=0.5)
        for i in range(10):
            assert len(t.neighbors[i]) >= 2

    def test_get_neighborhood_best(self):
        t = SwarmTopology(n_particles=5, k_neighbors=1)
        particles = [
            Particle(genome={"g": 1.0}, velocity={"g": 0.0}, fitness=10.0, best_fitness=10.0, id=i)
            for i in range(5)
        ]
        # particle 1 is neighbor of particle 0 (k=1 ring lattice), particle 2 is not
        particles[1].best_fitness = 100.0
        best = t.get_neighborhood_best(particles, 0)
        assert best is not None
        assert best.id == 1

    def test_clustering_coefficient(self):
        t = SwarmTopology(n_particles=5, k_neighbors=2, rewire_prob=0.0)
        cc = t.get_clustering_coefficient(0)
        assert 0 <= cc <= 1.0

    def test_clustering_single_neighbor(self):
        t = SwarmTopology(n_particles=5, k_neighbors=1)
        cc = t.get_clustering_coefficient(0)
        assert cc == 0.0  # Need at least 2 neighbors for triangles


class TestParticle:
    def test_update_personal_best(self):
        p = Particle(genome={"g": 1.0}, velocity={"g": 0.0}, fitness=10.0)
        p.best_fitness = 10.0
        p.fitness = 20.0
        p.update_personal_best()
        assert p.best_fitness == 20.0
        assert p.best_genome == {"g": 1.0}

    def test_no_update_personal_best(self):
        p = Particle(genome={"g": 1.0}, velocity={"g": 0.0}, fitness=5.0)
        p.best_fitness = 10.0
        p.update_personal_best()
        assert p.best_fitness == 10.0


class TestPheromoneTrail:
    def test_decay(self):
        t = PheromoneTrail(source={}, target={}, strength=1.0)
        t.decay(0.1)
        assert t.strength == 0.9

    def test_decay_multiple(self):
        t = PheromoneTrail(source={}, target={}, strength=1.0)
        for _ in range(10):
            t.decay(0.1)
        assert abs(t.strength - 0.9**10) < 1e-10


class TestSwarmIntelligenceBreeder:
    def test_init(self):
        breeder = SwarmIntelligenceBreeder(population_size=20)
        assert breeder.population_size == 20
        assert breeder.w_inertia == 0.7
        assert breeder.c_cognitive == 1.5
        assert breeder.c_social == 1.5

    def test_initialize(self):
        breeder = SwarmIntelligenceBreeder(population_size=10)
        breeder.initialize(
            task_fn=lambda g: {"fitness": sum(g.values())},
            bounds={"g1": (-5, 5), "g2": (-5, 5)}
        )
        assert len(breeder.particles) == 10
        assert all(p.fitness >= -10 for p in breeder.particles)

    def test_initialize_without_bounds(self):
        breeder = SwarmIntelligenceBreeder(population_size=10)
        breeder.initialize(
            task_fn=lambda g: {"fitness": sum(g.values())}
        )
        assert len(breeder.particles) == 10
        assert all("gene_0" in p.genome for p in breeder.particles)

    def test_breed_generation(self):
        breeder = SwarmIntelligenceBreeder(population_size=10)
        breeder.initialize(
            task_fn=lambda g: {"fitness": sum(g.values())},
            bounds={"g1": (-5, 5), "g2": (-5, 5)}
        )

        pop = breeder.breed_generation(
            task_fn=lambda g: {"fitness": sum(g.values())}
        )
        assert len(pop) == 10
        assert breeder.generation == 1

    def test_global_best_updated(self):
        breeder = SwarmIntelligenceBreeder(population_size=10)
        breeder.initialize(
            task_fn=lambda g: {"fitness": g.get("g1", 0) * 10},
            bounds={"g1": (-5, 5)}
        )
        initial_best = breeder.global_best_fitness
        for _ in range(5):
            breeder.breed_generation(task_fn=lambda g: {"fitness": g.get("g1", 0) * 10})
        assert breeder.global_best_fitness >= initial_best

    def test_velocity_update(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        particle = Particle(
            genome={"g1": 1.0}, velocity={"g1": 0.5},
            best_genome={"g1": 2.0}, best_fitness=10.0, id=0
        )
        neighborhood_best = Particle(
            genome={"g1": 3.0}, velocity={"g1": 0.0},
            best_genome={"g1": 3.0}, best_fitness=20.0, id=1
        )
        breeder._update_velocity(particle, neighborhood_best)
        assert "g1" in particle.velocity

    def test_pheromone_deposit(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        old = {"g1": 1.0}
        new = {"g1": 2.0}
        breeder._deposit_pheromone(old, new, 1.0)
        assert len(breeder.pheromones) == 1
        assert breeder.pheromones[0].strength > 0

    def test_pheromone_decay(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        breeder._deposit_pheromone({"g1": 1.0}, {"g1": 2.0}, 1.0)
        initial = len(breeder.pheromones)
        breeder._decay_pheromones(rate=1.0)  # Full decay
        assert len(breeder.pheromones) == 0

    def test_diversity_computation(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        breeder.particles = [
            Particle(genome={"g1": 1.0}, velocity={"g1": 0.0}),
            Particle(genome={"g1": 2.0}, velocity={"g1": 0.0}),
            Particle(genome={"g1": 3.0}, velocity={"g1": 0.0}),
        ]
        diversity = breeder._compute_diversity()
        assert diversity > 0

    def test_diversity_single_particle(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        breeder.particles = [
            Particle(genome={"g1": 1.0}, velocity={"g1": 0.0}),
        ]
        diversity = breeder._compute_diversity()
        assert diversity == 0.0

    def test_swarm_stats(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        breeder.initialize(
            task_fn=lambda g: {"fitness": sum(g.values())},
            bounds={"g1": (0, 1)}
        )
        stats = breeder.get_swarm_stats()
        assert "generation" in stats
        assert "n_particles" in stats
        assert "best_fitness" in stats
        assert "diversity" in stats

    def test_particle_states(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        breeder.initialize(
            task_fn=lambda g: {"fitness": sum(g.values())},
            bounds={"g1": (0, 1)}
        )
        states = breeder.get_particle_states()
        assert len(states) == 5
        assert all("id" in s for s in states)
        assert all("fitness" in s for s in states)

    def test_position_update(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        particle = Particle(
            genome={"g1": 1.0}, velocity={"g1": 0.5}
        )
        breeder._update_position(particle)
        assert particle.genome["g1"] == 1.5

    def test_max_velocity_clamping(self):
        breeder = SwarmIntelligenceBreeder(population_size=5, max_velocity=0.5)
        particle = Particle(
            genome={"g1": 1.0}, velocity={"g1": 0.0},
            best_genome={"g1": 100.0}, best_fitness=10.0, id=0
        )
        neighborhood_best = Particle(
            genome={"g1": 50.0}, velocity={"g1": 0.0},
            best_genome={"g1": 50.0}, best_fitness=20.0, id=1
        )
        breeder._update_velocity(particle, neighborhood_best)
        assert abs(particle.velocity["g1"]) <= 0.5

    def test_no_neighborhood_best(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        particle = Particle(
            genome={"g1": 1.0}, velocity={"g1": 0.5}
        )
        breeder._update_velocity(particle, None)
        # Velocity should still be updated (inertia + cognitive)
        assert "g1" in particle.velocity

    def test_pheromone_limit(self):
        breeder = SwarmIntelligenceBreeder(population_size=5)
        for i in range(150):
            breeder._deposit_pheromone({"g1": i}, {"g1": i+1}, 1.0)
        assert len(breeder.pheromones) <= 100

    def test_return_format(self):
        breeder = SwarmIntelligenceBreeder(population_size=10)
        breeder.initialize(
            task_fn=lambda g: {"fitness": sum(g.values())},
            bounds={"g1": (0, 1), "g2": (0, 1)}
        )
        pop = breeder.breed_generation(task_fn=lambda g: {"fitness": sum(g.values())})
        assert all(isinstance(genome, dict) for genome, _ in pop)
        assert all(isinstance(fitness, float) for _, fitness in pop)
