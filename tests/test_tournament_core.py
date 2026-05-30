"""Tests for the shared tournament simulation core."""

from __future__ import annotations

import pytest
from simulators.tournament_core import (
    Agent,
    crossover,
    mutate,
    tournament_step,
    diversity_metric,
    _mean,
    _std,
)


class TestAgent:
    def test_default_random_init(self):
        a = Agent()
        assert 0 <= a.ethos <= 1
        assert 0 <= a.pathos <= 1
        assert 0 <= a.logos <= 1

    def test_explicit_init(self):
        a = Agent(ethos=0.5, pathos=0.6, logos=0.7)
        assert a.ethos == 0.5
        assert a.pathos == 0.6
        assert a.logos == 0.7

    def test_fitness(self):
        a = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        assert a.fitness() == 0.125

    def test_dominates_true(self):
        a = Agent(ethos=0.8, pathos=0.8, logos=0.8)
        b = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        assert a.dominates(b)

    def test_dominates_false_equal(self):
        a = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        b = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        assert not a.dominates(b)

    def test_dominates_false_worse(self):
        a = Agent(ethos=0.3, pathos=0.3, logos=0.3)
        b = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        assert not a.dominates(b)

    def test_repr(self):
        a = Agent(ethos=0.5, pathos=0.6, logos=0.7)
        r = repr(a)
        assert "e=0.500" in r
        assert "p=0.600" in r
        assert "l=0.700" in r


class TestCrossover:
    def test_blend(self):
        a = Agent(ethos=1.0, pathos=0.0, logos=0.0)
        b = Agent(ethos=0.0, pathos=1.0, logos=0.0)
        c = crossover(a, b)
        assert 0 <= c.ethos <= 1
        assert 0 <= c.pathos <= 1
        assert 0 <= c.logos <= 1


class TestMutate:
    def test_clamping(self):
        a = Agent(ethos=0.99, pathos=0.01, logos=0.5)
        m = mutate(a, rate=0.5)
        assert 0 <= m.ethos <= 1
        assert 0 <= m.pathos <= 1
        assert 0 <= m.logos <= 1

    def test_preserves_when_rate_zero(self):
        a = Agent(ethos=0.5, pathos=0.5, logos=0.5)
        m = mutate(a, rate=0.0)
        assert abs(m.ethos - 0.5) < 0.001
        assert abs(m.pathos - 0.5) < 0.001
        assert abs(m.logos - 0.5) < 0.001


class TestTournamentStep:
    def test_reduces_population(self):
        pop = [Agent() for _ in range(20)]
        result, _ = tournament_step(pop, thermal_cap=15, mutation_rate=0.1)
        assert len(result) <= 15

    def test_track_breeding(self):
        pop = [Agent() for _ in range(20)]
        result, events = tournament_step(pop, thermal_cap=30, mutation_rate=0.1, track_breeding=True)
        assert events >= 0
        assert len(result) <= 30

    def test_no_track_breeding(self):
        pop = [Agent() for _ in range(20)]
        result, events = tournament_step(pop, thermal_cap=30, mutation_rate=0.1, track_breeding=False)
        assert events == 0

    def test_thermal_cap_below_winners(self):
        pop = [Agent() for _ in range(20)]
        result, _ = tournament_step(pop, thermal_cap=5, mutation_rate=0.1)
        # Winners are ~10 (half of 20), thermal_cap=5 only affects breeding.
        # The function returns all winners + offspring up to cap.
        # When cap < winners, no breeding happens and all winners are kept.
        assert len(result) == 10  # ~half the population (winners only)
        assert len(result) >= 5  # at least more than cap (no breeding)


class TestDiversityMetric:
    def test_homogeneous_population(self):
        pop = [Agent(ethos=0.5, pathos=0.5, logos=0.5) for _ in range(10)]
        assert diversity_metric(pop) == pytest.approx(0.0, abs=0.001)

    def test_diverse_population(self):
        pop = [Agent(ethos=i/10, pathos=i/10, logos=i/10) for i in range(10)]
        assert diversity_metric(pop) > 0.1

    def test_single_agent(self):
        pop = [Agent()]
        assert diversity_metric(pop) == 0.0


class TestStats:
    def test_mean(self):
        assert _mean([1, 2, 3]) == 2.0

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_std(self):
        assert _std([1, 2, 3]) == pytest.approx(0.816, abs=0.01)

    def test_std_single(self):
        assert _std([1]) == 0.0

    def test_std_empty(self):
        assert _std([]) == 0.0
