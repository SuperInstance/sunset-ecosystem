import pytest
import numpy as np
from swarm.adaptive_breeder import AdaptiveBreeder, BreederStrategy


class MockBreeder:
    def __init__(self, best_fitness: float = 0.0):
        self._best_fitness = best_fitness
        self.generation = 0

    def evolve(self, fitness_fn):
        self.generation += 1

    def get_best(self):
        return type(
            "obj", (object,), {"genome": np.array([1.0]), "fitness": self._best_fitness}
        )()


class TestBreederStrategy:
    def test_to_dict(self):
        s = BreederStrategy("a", None, 0.5, 1.0, 10)
        d = s.to_dict()
        assert d["name"] == "a"
        assert d["success_rate"] == 0.5


class TestAdaptiveBreeder:
    def test_init(self):
        ab = AdaptiveBreeder()
        assert ab.population_size == 50
        assert ab._current_strategy is None

    def test_add_strategy(self):
        ab = AdaptiveBreeder()
        b = MockBreeder()
        ab.add_strategy("test", b)
        assert "test" in ab._strategies
        assert ab._current_strategy == "test"

    def test_remove_strategy(self):
        ab = AdaptiveBreeder()
        b = MockBreeder()
        ab.add_strategy("test", b)
        assert ab.remove_strategy("test") is True
        assert ab.remove_strategy("test") is False

    def test_select_strategy(self):
        ab = AdaptiveBreeder()
        ab.add_strategy("a", MockBreeder(1.0))
        ab.add_strategy("b", MockBreeder(2.0))
        selected = ab.select_strategy()
        assert selected in ab._strategies

    def test_evolve(self):
        ab = AdaptiveBreeder()
        ab.add_strategy("test", MockBreeder(2.0))
        genome, fitness = ab.evolve(lambda x: x)
        assert fitness == 2.0
        assert ab.generation == 1

    def test_evolve_updates_success_rate(self):
        ab = AdaptiveBreeder()
        ab.add_strategy("test", MockBreeder(2.0))
        ab.evolve(lambda x: x)
        assert ab._strategies["test"].success_rate > 0

    def test_evolve_no_strategies(self):
        ab = AdaptiveBreeder()
        with pytest.raises(ValueError):
            ab.evolve(lambda x: x)

    def test_get_strategy_stats(self):
        ab = AdaptiveBreeder()
        ab.add_strategy("a", MockBreeder(1.0))
        ab.evolve(lambda x: x)
        stats = ab.get_strategy_stats()
        assert "strategies" in stats
        assert stats["current"] == "a"

    def test_export_json(self):
        ab = AdaptiveBreeder()
        ab.add_strategy("a", MockBreeder(1.0))
        ab.evolve(lambda x: x)
        j = ab.export_json()
        assert "a" in j
        assert "history" in j

    def test_to_dict(self):
        ab = AdaptiveBreeder()
        ab.add_strategy("a", MockBreeder(1.0))
        d = ab.to_dict()
        assert "stats" in d
