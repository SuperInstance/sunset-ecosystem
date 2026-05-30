"""Tests for simulators.tournament_core — tournament simulation primitives."""

import random

import pytest

from simulators.tournament_core import (
    Agent,
    crossover,
    diversity_metric,
    mutate,
    tournament_step,
    _mean,
    _std,
)


class TestAgent:
    def test_default_init(self):
        random.seed(42)
        a = Agent()
        assert 0.0 <= a.ethos <= 1.0
        assert 0.0 <= a.pathos <= 1.0
        assert 0.0 <= a.logos <= 1.0

    def test_explicit_init(self):
        a = Agent(ethos=0.5, pathos=0.6, logos=0.7)
        assert a.ethos == 0.5
        assert a.pathos == 0.6
        assert a.logos == 0.7

    def test_fitness(self):
        a = Agent(ethos=0.5, pathos=0.6, logos=0.7)
        assert a.fitness() == pytest.approx(0.5 * 0.6 * 0.7)

    def test_dominates_true(self):
        a = Agent(ethos=0.8, pathos=0.8, logos=0.8)
        b = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        assert a.dominates(b) is True
        assert b.dominates(a) is False

    def test_dominates_false(self):
        a = Agent(ethos=0.8, pathos=0.5, logos=0.5)
        b = Agent(ethos=0.5, pathos=0.8, logos=0.5)
        assert a.dominates(b) is False
        assert b.dominates(a) is False

    def test_repr(self):
        a = Agent(ethos=0.5, pathos=0.6, logos=0.7)
        assert "A(e=0.500" in repr(a)


class TestCrossover:
    def test_blend(self):
        random.seed(42)
        a = Agent(ethos=1.0, pathos=0.0, logos=0.5)
        b = Agent(ethos=0.0, pathos=1.0, logos=0.5)
        child = crossover(a, b)
        assert 0.0 <= child.ethos <= 1.0
        assert 0.0 <= child.pathos <= 1.0

    def test_no_error(self):
        a = Agent(ethos=0.1, pathos=0.2, logos=0.3)
        b = Agent(ethos=0.4, pathos=0.5, logos=0.6)
        child = crossover(a, b)
        assert isinstance(child, Agent)


class TestMutate:
    def test_mutation_range(self):
        random.seed(42)
        a = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        mutated = mutate(a, rate=0.1)
        assert 0.0 <= mutated.ethos <= 1.0
        assert 0.0 <= mutated.pathos <= 1.0
        assert 0.0 <= mutated.logos <= 1.0

    def test_clamping(self):
        a = Agent(ethos=0.99, pathos=0.01, logos=0.5)
        # Force positive mutation by seeding
        random.seed(1)
        mutated = mutate(a, rate=0.5)
        assert mutated.ethos <= 1.0
        assert mutated.pathos >= 0.0


class TestTournamentStep:
    def test_reduces_or_maintains(self):
        random.seed(42)
        pop = [Agent() for _ in range(20)]
        new_pop, _ = tournament_step(pop, thermal_cap=15, mutation_rate=0.1)
        assert len(new_pop) <= 20

    def test_track_breeding(self):
        random.seed(42)
        pop = [Agent() for _ in range(20)]
        new_pop, events = tournament_step(
            pop, thermal_cap=20, mutation_rate=0.1, track_breeding=True
        )
        assert events >= 0
        assert len(new_pop) <= 20

    def test_small_pop(self):
        random.seed(42)
        pop = [Agent() for _ in range(2)]
        new_pop, _ = tournament_step(pop, thermal_cap=10, mutation_rate=0.1)
        assert len(new_pop) >= 1


class TestDiversityMetric:
    def test_zero_for_single(self):
        pop = [Agent(ethos=0.5, pathos=0.5, logos=0.5)]
        assert diversity_metric(pop) == 0.0

    def test_positive_for_diverse(self):
        pop = [
            Agent(ethos=0.1, pathos=0.1, logos=0.1),
            Agent(ethos=0.9, pathos=0.9, logos=0.9),
        ]
        assert diversity_metric(pop) > 0.0

    def test_zero_for_empty(self):
        assert diversity_metric([]) == 0.0


class TestStats:
    def test_mean(self):
        assert _mean([1, 2, 3]) == 2.0
        assert _mean([]) == 0.0

    def test_std(self):
        assert _std([1, 1, 1]) == 0.0
        assert _std([1, 2, 3]) > 0.0
        assert _std([]) == 0.0
        assert _std([5]) == 0.0
