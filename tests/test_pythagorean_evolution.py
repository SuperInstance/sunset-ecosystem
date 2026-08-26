"""Tests for Pythagorean Evolution — exact rational breeding system.

Covers PythagoreanTriple, Genome, LatticeWalkMutation, ExactGeometricCrossover,
HolonomicFitness, and PythagoreanBreeder.
"""

import math
import numpy as np
import pytest

from swarm.pythagorean_evolution import (
    PythagoreanTriple,
    PythagoreanGenome,
    LatticeWalkMutation,
    ExactGeometricCrossover,
    HolonomicFitness,
    PythagoreanBreeder,
)


class TestPythagoreanTriple:
    def test_valid_triple(self):
        t = PythagoreanTriple(3, 4, 5)
        assert t.a == 3
        assert t.b == 4
        assert t.c == 5

    def test_invalid_triple_raises(self):
        with pytest.raises(AssertionError):
            PythagoreanTriple(1, 2, 3)

    def test_to_vector(self):
        t = PythagoreanTriple(3, 4, 5)
        v = t.to_vector()
        assert pytest.approx(v[0]) == 0.6
        assert pytest.approx(v[1]) == 0.8

    def test_angle(self):
        t = PythagoreanTriple(3, 4, 5)
        angle = t.angle()
        assert pytest.approx(angle) == math.atan2(4, 3)

    def test_random_primitive(self):
        t = PythagoreanTriple.random_primitive()
        assert t.a * t.a + t.b * t.b == t.c * t.c

    def test_from_vector(self):
        t = PythagoreanTriple.from_vector([0.6, 0.8])
        assert t.a * t.a + t.b * t.b == t.c * t.c

    def test_adjacent_triples(self):
        t = PythagoreanTriple(3, 4, 5)
        adjacent = t.adjacent_triples(max_n=5)
        assert len(adjacent) > 0
        for adj in adjacent:
            assert adj.a * adj.a + adj.b * adj.b == adj.c * adj.c

    def test_repr(self):
        t = PythagoreanTriple(3, 4, 5)
        assert "PT(3,4,5)" == repr(t)


class TestPythagoreanGenome:
    def test_init(self):
        triples = [PythagoreanTriple(3, 4, 5), PythagoreanTriple(5, 12, 13)]
        genome = PythagoreanGenome(triples=triples)
        assert genome.length() == 2

    def test_to_matrix(self):
        triples = [PythagoreanTriple(3, 4, 5), PythagoreanTriple(5, 12, 13)]
        genome = PythagoreanGenome(triples=triples)
        matrix = genome.to_matrix()
        assert matrix.shape == (2, 2)
        assert pytest.approx(matrix[0][0]) == 0.6
        assert pytest.approx(matrix[0][1]) == 0.8

    def test_check_holonomy_short(self):
        triples = [PythagoreanTriple(3, 4, 5), PythagoreanTriple(5, 12, 13)]
        genome = PythagoreanGenome(triples=triples)
        assert genome.check_holonomy() is True

    def test_check_holonomy_cycle(self):
        # Square cycle: (1,0), (0,1), (-1,0), (0,-1)
        # Use triples: (3,0,3) is degenerate, so use (3,4,5) and rotate
        t1 = PythagoreanTriple(3, 4, 5)
        t2 = PythagoreanTriple(5, 12, 13)  # Different angle
        t3 = PythagoreanTriple(8, 15, 17)
        t4 = PythagoreanTriple(7, 24, 25)
        genome = PythagoreanGenome(triples=[t1, t2, t3, t4])
        # Just check it doesn't crash
        genome.holonomy_error()

    def test_holonomy_error(self):
        genome = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)])
        assert genome.holonomy_error() == 0.0

    def test_copy(self):
        triples = [PythagoreanTriple(3, 4, 5)]
        genome = PythagoreanGenome(triples=triples, fitness=1.5, age=10)
        copy = genome.copy()
        assert copy.fitness == 1.5
        assert copy.age == 10
        assert len(copy.triples) == 1


class TestLatticeWalkMutation:
    def test_mutate(self):
        triples = [PythagoreanTriple(3, 4, 5) for _ in range(5)]
        genome = PythagoreanGenome(triples=triples)
        mutator = LatticeWalkMutation(step_size=0.2, max_steps=5)
        child = mutator.mutate(genome)
        assert len(child.triples) == len(genome.triples)
        # At least one triple might be different

    def test_mutate_preserves_exactness(self):
        triples = [PythagoreanTriple(3, 4, 5) for _ in range(3)]
        genome = PythagoreanGenome(triples=triples)
        mutator = LatticeWalkMutation(step_size=0.5, max_steps=5)
        child = mutator.mutate(genome)
        for t in child.triples:
            assert t.a * t.a + t.b * t.b == t.c * t.c


class TestExactGeometricCrossover:
    def test_crossover(self):
        p1 = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)] * 3)
        p2 = PythagoreanGenome(triples=[PythagoreanTriple(5, 12, 13)] * 3)
        crossover = ExactGeometricCrossover()
        c1, c2 = crossover.crossover(p1, p2)
        assert len(c1.triples) == 3
        assert len(c2.triples) == 3
        for t in c1.triples + c2.triples:
            assert t.a * t.a + t.b * t.b == t.c * t.c

    def test_crossover_exactness(self):
        p1 = PythagoreanGenome(triples=[PythagoreanTriple(3, 4, 5)] * 5)
        p2 = PythagoreanGenome(triples=[PythagoreanTriple(5, 12, 13)] * 5)
        crossover = ExactGeometricCrossover()
        c1, c2 = crossover.crossover(p1, p2)
        for t in c1.triples:
            assert t.a * t.a + t.b * t.b == t.c * t.c
        for t in c2.triples:
            assert t.a * t.a + t.b * t.b == t.c * t.c


class TestHolonomicFitness:
    def test_evaluate(self):
        def task_fn(matrix):
            return float(np.sum(matrix))

        triples = [PythagoreanTriple(3, 4, 5)]
        genome = PythagoreanGenome(triples=triples)
        fitness_fn = HolonomicFitness(
            task_fn, holonomy_weight=0.3, exactness_weight=0.1
        )
        score = fitness_fn.evaluate(genome)
        assert score > 0

    def test_holonomy_bonus(self):
        def task_fn(matrix):
            return 0.0

        # Low holonomy error should get bonus
        t1 = PythagoreanTriple(3, 4, 5)
        t2 = PythagoreanTriple(5, 12, 13)
        genome = PythagoreanGenome(triples=[t1, t2])
        fitness_fn = HolonomicFitness(task_fn, holonomy_weight=1.0)
        score = fitness_fn.evaluate(genome)
        assert score > 0  # Should get some holonomy bonus


class TestPythagoreanBreeder:
    def test_initialize(self):
        breeder = PythagoreanBreeder(population_size=10, genome_length=5)
        breeder.initialize()
        assert len(breeder.population) == 10
        for genome in breeder.population:
            assert genome.length() == 5
            for t in genome.triples:
                assert t.a * t.a + t.b * t.b == t.c * t.c

    def test_evaluate_fitness(self):
        breeder = PythagoreanBreeder(population_size=10, genome_length=3)
        breeder.initialize()
        breeder.evaluate_fitness(lambda m: float(np.sum(m)))
        assert breeder.best_fitness > 0
        assert breeder.best_genome is not None

    def test_select_and_breed(self):
        breeder = PythagoreanBreeder(population_size=10, genome_length=3)
        breeder.initialize()
        breeder.evaluate_fitness(lambda m: float(np.sum(m)))
        breeder.select_and_breed()
        assert len(breeder.population) == 10
        assert breeder.generation == 1

    def test_full_evolution(self):
        breeder = PythagoreanBreeder(
            population_size=20, genome_length=5, mutation_rate=0.3, crossover_rate=0.7
        )
        breeder.initialize()

        def task_fn(matrix):
            return float(np.sum(matrix))

        best_fitness_history = []
        for gen in range(10):
            breeder.evaluate_fitness(task_fn)
            breeder.select_and_breed()
            best_fitness_history.append(breeder.best_fitness)

        assert breeder.generation == 10
        assert breeder.best_fitness > 0

    def test_stats(self):
        breeder = PythagoreanBreeder(population_size=10, genome_length=3)
        breeder.initialize()
        stats = breeder.get_stats()
        assert stats["generation"] == 0
        assert stats["population_size"] == 10

    def test_elitism(self):
        breeder = PythagoreanBreeder(
            population_size=10, genome_length=3, elitism_count=2
        )
        breeder.initialize()
        breeder.evaluate_fitness(lambda m: float(np.sum(m)))
        best_before = breeder.best_fitness
        breeder.select_and_breed()
        breeder.evaluate_fitness(lambda m: float(np.sum(m)))
        assert (
            breeder.best_fitness >= best_before * 0.9
        )  # Should not drop too much due to elitism

    def test_age_culling(self):
        breeder = PythagoreanBreeder(population_size=10, genome_length=3, max_age=2)
        breeder.initialize()
        for gen in range(5):
            breeder.evaluate_fitness(lambda m: float(np.sum(m)))
            breeder.select_and_breed()
        # After 5 generations, old genomes should be culled
        assert all(g.age < 2 for g in breeder.population)

    def test_triples_exact_through_evolution(self):
        breeder = PythagoreanBreeder(population_size=10, genome_length=5)
        breeder.initialize()
        for gen in range(5):
            breeder.evaluate_fitness(lambda m: float(np.sum(m)))
            breeder.select_and_breed()
        for genome in breeder.population:
            for t in genome.triples:
                assert t.a * t.a + t.b * t.b == t.c * t.c
