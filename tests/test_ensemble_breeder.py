import pytest
import numpy as np
from swarm.ensemble_breeder import BreederWeight, EnsembleBreeder


class MockBreeder:
    """Mock breeder for testing."""

    def __init__(self, best_fitness: float = 0.0):
        self._best_fitness = best_fitness
        self.generation = 0

    def initialize(self):
        pass

    def evolve(self, fitness_fn):
        self.generation += 1

    def get_best(self):
        return type(
            "obj", (object,), {"genome": np.array([1.0]), "fitness": self._best_fitness}
        )()

    def evaluate(self, fitness_fn):
        return self._best_fitness


class TestBreederWeight:
    def test_to_dict(self):
        w = BreederWeight("a", 0.5, 1.0)
        d = w.to_dict()
        assert d["breeder_name"] == "a"
        assert d["weight"] == 0.5


class TestEnsembleBreeder:
    def test_init(self):
        eb = EnsembleBreeder()
        assert eb.population_size == 50
        assert eb.breeders == {}

    def test_add_breeder(self):
        eb = EnsembleBreeder()
        b = MockBreeder()
        eb.add_breeder("test", b)
        assert "test" in eb.breeders
        assert eb.weights["test"].weight == 1.0

    def test_remove_breeder(self):
        eb = EnsembleBreeder()
        b = MockBreeder()
        eb.add_breeder("test", b)
        assert eb.remove_breeder("test") is True
        assert eb.remove_breeder("test") is False

    def test_initialize(self):
        eb = EnsembleBreeder()
        b = MockBreeder()
        eb.add_breeder("test", b)
        eb.initialize()
        # Should not raise

    def test_evaluate(self):
        eb = EnsembleBreeder()
        eb.add_breeder("a", MockBreeder(1.0))
        eb.add_breeder("b", MockBreeder(2.0))
        perfs = eb.evaluate(lambda x: x)
        assert perfs["a"] == 1.0
        assert perfs["b"] == 2.0

    def test_evolve(self):
        eb = EnsembleBreeder()
        eb.add_breeder("a", MockBreeder(1.0))
        eb.add_breeder("b", MockBreeder(2.0))
        genome, fitness = eb.evolve(lambda x: x)
        assert fitness == 2.0
        assert eb.generation == 1

    def test_evolve_updates_weights(self):
        eb = EnsembleBreeder()
        eb.add_breeder("a", MockBreeder(1.0))
        eb.add_breeder("b", MockBreeder(2.0))
        eb.evolve(lambda x: x)
        # b should have higher weight
        assert eb.weights["b"].weight > eb.weights["a"].weight

    def test_get_ensemble_stats(self):
        eb = EnsembleBreeder()
        eb.add_breeder("a", MockBreeder(1.0))
        eb.evolve(lambda x: x)
        stats = eb.get_ensemble_stats()
        assert stats["breeders"] == 1
        assert len(stats["best_history"]) == 1

    def test_export_json(self):
        eb = EnsembleBreeder()
        eb.add_breeder("a", MockBreeder(1.0))
        eb.evolve(lambda x: x)
        j = eb.export_json()
        assert "a" in j
        assert "history" in j

    def test_to_dict(self):
        eb = EnsembleBreeder()
        eb.add_breeder("a", MockBreeder(1.0))
        d = eb.to_dict()
        assert "stats" in d
