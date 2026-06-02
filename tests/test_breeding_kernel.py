"""Tests for breeding_kernel.py — evolutionary engine."""

import math
import pytest
from swarm.breeding_kernel import (
    Genome,
    TournamentSelector,
    RouletteSelector,
    GaussianMutator,
    CreepMutator,
    CallableEvaluator,
    ElitistSurvivor,
    TruncationSurvivor,
    GenerationalSurvivor,
    BreedingPreset,
    BreedingKernel,
    BreedingEvent,
)


class TestGenome:
    def test_init(self):
        g = Genome([1.0, 2.0, 3.0], fitness=5.0, metadata={"x": 1})
        assert g.genes == [1.0, 2.0, 3.0]
        assert g.fitness == 5.0
        assert g.metadata == {"x": 1}

    def test_copy_isolation(self):
        g = Genome([1.0, 2.0], fitness=1.0)
        c = g.copy()
        c.genes[0] = 99.0
        assert g.genes[0] == 1.0

    def test_repr(self):
        g = Genome([1.0], fitness=2.0)
        assert "Genome" in repr(g)


class TestSelectors:
    def test_tournament_select(self):
        pop = [Genome([i], fitness=float(i)) for i in range(10)]
        sel = TournamentSelector(tournament_size=3)
        result = sel.select(pop, n=5)
        assert len(result) == 5
        assert all(isinstance(g, Genome) for g in result)

    def test_roulette_select(self):
        pop = [Genome([i], fitness=float(i + 1)) for i in range(10)]
        sel = RouletteSelector()
        result = sel.select(pop, n=5)
        assert len(result) == 5

    def test_roulette_zero_fitness(self):
        pop = [Genome([i], fitness=0.0) for i in range(5)]
        sel = RouletteSelector()
        result = sel.select(pop, n=3)
        assert len(result) == 3


class TestMutators:
    def test_gaussian_mutate(self):
        g = Genome([0.0, 0.0, 0.0])
        m = GaussianMutator(sigma=0.1, probability=1.0)
        child = m.mutate(g)
        assert child.genes != g.genes
        assert len(child.genes) == 3

    def test_creep_mutate(self):
        g = Genome([0.0, 0.0])
        m = CreepMutator(step=0.5, probability=1.0)
        child = m.mutate(g)
        # Each gene should change by exactly +/- step
        for a, b in zip(g.genes, child.genes):
            assert abs(b - a) == pytest.approx(0.5)


class TestSurvivors:
    def test_elitist_survive(self):
        pop = [Genome([i], fitness=float(i)) for i in range(5)]
        off = [Genome([i + 10], fitness=float(i + 10)) for i in range(5)]
        s = ElitistSurvivor(elite_count=2)
        result = s.survive(pop, off, pop_size=5)
        assert len(result) == 5
        # Elites should be the 2 highest-fitness individuals overall
        # (offspring have higher fitness, so they could be elites)
        assert all(isinstance(g, Genome) for g in result)
        assert len(set(id(g) for g in result)) == 5  # all unique objects

    def test_elitist_survive_preserves_original_elites(self):
        """When offspring are worse, original elites are preserved."""
        pop = [Genome([i], fitness=float(i + 10)) for i in range(5)]
        off = [Genome([i], fitness=float(i)) for i in range(5)]
        s = ElitistSurvivor(elite_count=2)
        result = s.survive(pop, off, pop_size=5)
        assert len(result) == 5
        # Top 2 should be from population since they have higher fitness
        assert result[0].fitness == 14.0
        assert result[1].fitness == 13.0

    def test_truncation_survive(self):
        pop = [Genome([i], fitness=float(i)) for i in range(3)]
        off = [Genome([i + 10], fitness=float(i + 10)) for i in range(3)]
        s = TruncationSurvivor()
        result = s.survive(pop, off, pop_size=4)
        assert len(result) == 4
        # Best 4 overall
        assert result[0].fitness == 12.0

    def test_generational_survive(self):
        pop = [Genome([i], fitness=float(i)) for i in range(5)]
        off = [Genome([i + 100], fitness=float(i + 100)) for i in range(5)]
        s = GenerationalSurvivor()
        result = s.survive(pop, off, pop_size=5)
        assert len(result) == 5
        assert all(g.fitness >= 100 for g in result)


class TestEvaluator:
    def test_callable_evaluator(self):
        ev = CallableEvaluator(lambda g: sum(g.genes))
        g = Genome([1.0, 2.0, 3.0])
        assert ev.evaluate(g) == 6.0


class TestBreedingEvent:
    def test_properties(self):
        e = BreedingEvent(
            "generation",
            5,
            {
                "best_fitness": 10.0,
                "avg_fitness": 5.0,
                "diversity": 0.5,
                "qd_coverage": 0.8,
                "qd_score": 100.0,
                "nodes_agreed": 3,
                "total_nodes": 5,
                "flux_passed": 7,
                "flux_failed": 2,
            },
        )
        assert e.best_fitness == 10.0
        assert e.mean_fitness == 5.0
        assert e.diversity == 0.5
        assert e.qd_coverage == 0.8
        assert e.qd_score == 100.0
        assert e.nodes_agreed == 3
        assert e.total_nodes == 5
        assert e.flux_passed == 7
        assert e.flux_failed == 2
        assert "generation" in repr(e)


class TestBreedingKernel:
    def _make_kernel(self, pop_size=10):
        pop = [Genome([float(i)], fitness=float(i)) for i in range(pop_size)]
        ev = CallableEvaluator(lambda g: sum(g.genes))
        return BreedingKernel(
            selector=TournamentSelector(),
            mutator=GaussianMutator(sigma=0.01, probability=0.1),
            evaluator=ev,
            survivor=ElitistSurvivor(elite_count=2),
            population=pop,
            pop_size=pop_size,
            name="test",
        )

    def test_init(self):
        bk = self._make_kernel()
        assert bk.name == "test"
        assert bk.pop_size == 10
        assert bk.generation == 0

    def test_from_preset_exploitation(self):
        pop = [Genome([0.0], fitness=0.0) for _ in range(10)]
        ev = CallableEvaluator(lambda g: 1.0)
        bk = BreedingKernel.from_preset(BreedingPreset.EXPLOITATION, ev, pop, pop_size=10)
        assert bk.name == "breeder"
        assert bk.pop_size == 10

    def test_from_preset_exploration(self):
        pop = [Genome([0.0], fitness=0.0) for _ in range(10)]
        ev = CallableEvaluator(lambda g: 1.0)
        bk = BreedingKernel.from_preset(BreedingPreset.EXPLORATION, ev, pop, pop_size=10)
        assert bk.pop_size == 10

    def test_from_preset_balanced(self):
        pop = [Genome([0.0], fitness=0.0) for _ in range(10)]
        ev = CallableEvaluator(lambda g: 1.0)
        bk = BreedingKernel.from_preset(BreedingPreset.BALANCED, ev, pop, pop_size=10)
        assert bk.pop_size == 10

    def test_from_preset_diversity(self):
        pop = [Genome([0.0], fitness=0.0) for _ in range(10)]
        ev = CallableEvaluator(lambda g: 1.0)
        bk = BreedingKernel.from_preset(BreedingPreset.DIVERSITY, ev, pop, pop_size=10)
        assert bk.pop_size == 10

    def test_step_evaluates_and_evolve(self):
        bk = self._make_kernel()
        event = bk.step()
        assert isinstance(event, BreedingEvent)
        assert bk.generation == 1
        assert len(bk.fitness_history) == 1
        assert len(bk.diversity_history) == 1
        assert event.payload["avg_fitness"] is not None

    def test_run_multiple_generations(self):
        bk = self._make_kernel()
        events = bk.run(generations=5)
        assert len(events) == 5
        assert bk.generation == 5

    def test_archive_grows(self):
        bk = self._make_kernel()
        bk.run(generations=3)
        assert len(bk.archive) > 0

    def test_qd_score(self):
        bk = self._make_kernel()
        assert bk.qd_score == 0.0  # empty archive at start
        bk.run(generations=2)
        assert bk.qd_score >= 0.0

    def test_diversity_computation(self):
        pop = [Genome([0.0]), Genome([1.0]), Genome([2.0])]
        d = BreedingKernel._compute_diversity(pop)
        assert d > 0.0

    def test_diversity_single_genome(self):
        d = BreedingKernel._compute_diversity([Genome([0.0])])
        assert d == 0.0

    def test_repr(self):
        bk = self._make_kernel()
        assert "test" in repr(bk)
        assert "gen=0" in repr(bk)
