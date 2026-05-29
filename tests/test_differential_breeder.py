import pytest
import numpy as np
from swarm.differential_breeder import DEIndividual, DifferentialBreeder


class TestDEIndividual:
    def test_to_dict(self):
        ind = DEIndividual(genome=np.array([1.0, 2.0]), fitness=0.5)
        d = ind.to_dict()
        assert d["genome"] == [1.0, 2.0]
        assert d["fitness"] == 0.5


class TestDifferentialBreeder:
    def test_init(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        assert db.population_size == 10
        assert db.dimensions == 3

    def test_initialize(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        assert len(db.population) == 10
        assert all(len(ind.genome) == 3 for ind in db.population)

    def test_evaluate(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        best = db.evaluate(lambda g: -np.sum(g**2))
        assert best <= 0
        assert db.best_individual is not None

    def test_evolve(self):
        db = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        db.initialize()
        fitness_fn = lambda g: -np.sum(g**2)
        db.evaluate(fitness_fn)
        best_before = db.best_individual.fitness
        db.evolve(fitness_fn)
        assert db.generation == 1

    def test_evolve_no_fitness_fn(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        db.evolve()  # No fitness function
        assert db.generation == 1
        assert len(db.population) == 10

    def test_get_best(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        db.evaluate(lambda g: -np.sum(g**2))
        assert db.get_best() is not None

    def test_get_diversity(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        div = db.get_diversity()
        assert div > 0

    def test_get_stats(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        db.evaluate(lambda g: -np.sum(g**2))
        stats = db.get_stats()
        assert "best_fitness" in stats
        assert "avg_fitness" in stats
        assert "diversity" in stats

    def test_to_dict(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        db.evaluate(lambda g: -np.sum(g**2))
        d = db.to_dict()
        assert "stats" in d
        assert "best" in d

    def test_bounds_respected(self):
        db = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        db.initialize()
        for ind in db.population:
            assert np.all(ind.genome >= -1)
            assert np.all(ind.genome <= 1)

    def test_mutant_in_bounds(self):
        db = DifferentialBreeder(population_size=10, dimensions=3, bounds=(-1, 1))
        db.initialize()
        mutant = db._mutate(0)
        assert np.all(mutant >= -1)
        assert np.all(mutant <= 1)

    def test_crossover_length(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        target = db.population[0].genome
        mutant = db._mutate(0)
        trial = db._crossover(target, mutant)
        assert len(trial) == 3

    def test_select_donors_distinct(self):
        db = DifferentialBreeder(population_size=10, dimensions=3)
        db.initialize()
        a, b, c = db._select_donors(0)
        assert a != 0 and b != 0 and c != 0
        assert a != b and b != c and a != c

    def test_fitness_improvement(self):
        db = DifferentialBreeder(population_size=20, dimensions=2, F=0.8, CR=0.9)
        db.initialize()
        fitness_fn = lambda g: -np.sum((g - 1.0)**2)
        db.evaluate(fitness_fn)
        best_before = db.best_individual.fitness
        for _ in range(50):
            db.evolve(fitness_fn)
        best_after = db.best_individual.fitness
        assert best_after >= best_before

    def test_dimensions_1(self):
        db = DifferentialBreeder(population_size=5, dimensions=1)
        db.initialize()
        assert len(db.population) == 5
        assert len(db.population[0].genome) == 1
