"""Tests for the Meta-Learning Breeder."""

import pytest
import math

from swarm.meta_learning_breeder import (
    MetaLearningBreeder,
    ProblemFingerprint,
    StrategyRecord,
)


class TestProblemFingerprint:
    def test_to_key(self):
        fp = ProblemFingerprint(10, ("thermal", "memory"), "smooth")
        assert fp.to_key() == "10:memory,thermal:smooth"

    def test_to_key_sorted(self):
        fp = ProblemFingerprint(5, ("z", "a"), "rugged")
        assert fp.to_key() == "5:a,z:rugged"


class TestStrategyRecord:
    def test_init(self):
        r = StrategyRecord("mutate")
        assert r.ema_rate == 0.5

    def test_update_success(self):
        r = StrategyRecord("mutate")
        r.update(True)
        assert r.attempts == 1
        assert r.successes == 1
        assert r.ema_rate > 0.5

    def test_update_failure(self):
        r = StrategyRecord("mutate")
        r.update(False)
        assert r.attempts == 1
        assert r.successes == 0
        assert r.ema_rate < 0.5

    def test_score(self):
        r = StrategyRecord("mutate")
        assert r.score() > 0

    def test_score_exploration_bonus(self):
        r = StrategyRecord("mutate", attempts=0)
        assert r.score() > 0.5  # bonus for untried

        r2 = StrategyRecord("mutate", attempts=100, successes=100, ema_rate=1.0)
        # With many attempts, bonus is tiny, so ema_rate dominates
        assert r2.score() > 0.9  # close to 1.0
        assert r2.score() > r.score()  # high success rate wins over exploration bonus


class TestMetaLearningBreeder:
    def test_init(self):
        b = MetaLearningBreeder()
        assert b.get_strategy_stats() == {}

    def test_add_strategy(self):
        b = MetaLearningBreeder()
        b.add_strategy("add_noise", lambda g: [x + 0.1 for x in g])
        assert len(b.strategies) == 1

    def test_remove_strategy(self):
        b = MetaLearningBreeder()
        b.add_strategy("a", lambda g: g)
        b.add_strategy("b", lambda g: g)
        b.remove_strategy("a")
        assert len(b.strategies) == 1

    def test_fingerprint(self):
        b = MetaLearningBreeder()
        fp = b.fingerprint([1.0, 2.0, 3.0], ["thermal"])
        assert fp.dim == 3
        assert fp.constraint_types == ("thermal",)

    def test_select_strategy_uniform_when_no_history(self):
        b = MetaLearningBreeder()
        b.add_strategy("a", lambda g: g)
        b.add_strategy("b", lambda g: g)
        fp = ProblemFingerprint(2, (), "smooth")
        name, func = b.select_strategy(fp)
        assert name in ("a", "b")

    def test_select_strategy_prefers_successful(self):
        b = MetaLearningBreeder()
        b.add_strategy("good", lambda g: g)
        b.add_strategy("bad", lambda g: g)
        fp = ProblemFingerprint(2, (), "smooth")

        # Good always improves, bad never does
        for _ in range(20):
            b.learn(fp, "good", 0.0, 1.0)
            b.learn(fp, "bad", 1.0, 0.0)

        # With low temperature, should strongly prefer good
        b.temperature = 0.1
        picks = [b.select_strategy(fp)[0] for _ in range(50)]
        good_count = sum(1 for p in picks if p == "good")
        assert good_count > 30  # strong preference

    def test_mutate(self):
        b = MetaLearningBreeder()
        b.add_strategy("add", lambda g: [x + 1 for x in g])
        fp = ProblemFingerprint(2, (), "smooth")
        child, name = b.mutate([0.0, 0.0], fp)
        assert name == "add"
        assert child == [1.0, 1.0]

    def test_learn(self):
        b = MetaLearningBreeder()
        b.add_strategy("add", lambda g: g)
        fp = ProblemFingerprint(2, (), "smooth")
        b.learn(fp, "add", 0.0, 1.0)
        stats = b.get_strategy_stats(fp)
        assert stats["add"]["successes"] == 1

    def test_learn_failure(self):
        b = MetaLearningBreeder()
        b.add_strategy("add", lambda g: g)
        fp = ProblemFingerprint(2, (), "smooth")
        b.learn(fp, "add", 1.0, 0.0)
        stats = b.get_strategy_stats(fp)
        assert stats["add"]["successes"] == 0

    def test_evolve_improves_fitness(self):
        b = MetaLearningBreeder()
        b.add_strategy("add", lambda g: [x + 0.1 for x in g])
        b.add_strategy("sub", lambda g: [x - 0.1 for x in g])

        # Fitness is sum of squares - maximize by going positive
        fitness = lambda g: sum(x * x for x in g)
        population = [[0.0, 0.0] for _ in range(10)]

        result = b.evolve(population, fitness, [], "smooth", generations=5)
        best = max(result, key=lambda x: x[1])
        assert best[1] > 0  # should have improved from 0

    def test_evolve_learns_preference(self):
        b = MetaLearningBreeder()
        b.add_strategy("add", lambda g: [x + 0.5 for x in g])
        b.add_strategy("sub", lambda g: [x - 0.5 for x in g])

        # Fitness: sum of elements - maximize by going positive
        fitness = lambda g: sum(g)
        population = [[0.0, 0.0] for _ in range(10)]

        b.evolve(population, fitness, [], "smooth", generations=10)

        # Should learn that "add" is better for this fitness landscape
        fp = b.fingerprint([0.0, 0.0], [], "smooth")
        stats = b.get_strategy_stats(fp)
        if "add" in stats and "sub" in stats:
            assert stats["add"]["rate"] > stats["sub"]["rate"]

    def test_get_strategy_stats_aggregate(self):
        b = MetaLearningBreeder()
        b.add_strategy("a", lambda g: g)
        fp1 = ProblemFingerprint(2, (), "smooth")
        fp2 = ProblemFingerprint(3, (), "rugged")
        b.learn(fp1, "a", 0.0, 1.0)
        b.learn(fp2, "a", 0.0, 1.0)
        stats = b.get_strategy_stats()
        assert stats["a"]["attempts"] == 2

    def test_to_dict(self):
        b = MetaLearningBreeder()
        b.add_strategy("a", lambda g: g)
        d = b.to_dict()
        assert d["strategies"] == ["a"]
        assert d["temperature"] == 1.0

    def test_empty_strategies(self):
        b = MetaLearningBreeder()
        fp = ProblemFingerprint(2, (), "smooth")
        name, func = b.select_strategy(fp)
        assert name == "none"
