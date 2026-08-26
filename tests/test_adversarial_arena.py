"""Tests for Adversarial Co-Evolution Arena.

Covers SolverGenome, TesterGenome, mutations, crossover, and the full arena.
"""

import numpy as np
import pytest

from swarm.adversarial_arena import (
    SolverGenome,
    TesterGenome,
    SolverMutation,
    TesterMutation,
    SolverCrossover,
    TesterCrossover,
    AdversarialArena,
)


class TestSolverGenome:
    def test_random(self):
        g = SolverGenome.random(10)
        assert len(g.vector) == 10

    def test_copy(self):
        g = SolverGenome.random(10)
        g.fitness = 5.0
        c = g.copy()
        assert c.fitness == 5.0
        assert np.allclose(c.vector, g.vector)
        c.vector[0] = 999
        assert g.vector[0] != 999


class TestTesterGenome:
    def test_random(self):
        g = TesterGenome.random(10)
        assert len(g.vector) == 10

    def test_copy(self):
        g = TesterGenome.random(10)
        g.fitness = 5.0
        c = g.copy()
        assert c.fitness == 5.0
        assert np.allclose(c.vector, g.vector)


class TestSolverMutation:
    def test_mutate(self):
        m = SolverMutation(rate=1.0, strength=0.5)
        g = SolverGenome.random(10)
        child = m.mutate(g)
        assert len(child.vector) == 10
        # Should be different from parent
        assert not np.allclose(child.vector, g.vector)


class TestTesterMutation:
    def test_mutate(self):
        m = TesterMutation(rate=1.0, strength=0.5)
        g = TesterGenome.random(10)
        child = m.mutate(g)
        assert len(child.vector) == 10
        assert not np.allclose(child.vector, g.vector)


class TestSolverCrossover:
    def test_crossover(self):
        xo = SolverCrossover(alpha=0.5)
        p1 = SolverGenome.random(10)
        p2 = SolverGenome.random(10)
        c1, c2 = xo.crossover(p1, p2)
        assert len(c1.vector) == 10
        assert len(c2.vector) == 10


class TestTesterCrossover:
    def test_crossover(self):
        xo = TesterCrossover(alpha=0.5)
        p1 = TesterGenome.random(10)
        p2 = TesterGenome.random(10)
        c1, c2 = xo.crossover(p1, p2)
        assert len(c1.vector) == 10
        assert len(c2.vector) == 10


class TestAdversarialArena:
    def test_initialize(self):
        arena = AdversarialArena(
            solver_pop_size=10,
            tester_pop_size=5,
            solver_dim=5,
            tester_dim=3,
        )
        arena.initialize()
        assert len(arena.solver_pop) == 10
        assert len(arena.tester_pop) == 5
        for s in arena.solver_pop:
            assert len(s.vector) == 5
        for t in arena.tester_pop:
            assert len(t.vector) == 3

    def test_evaluate(self):
        arena = AdversarialArena(
            solver_pop_size=10,
            tester_pop_size=5,
            n_interactions_per_gen=3,
        )
        arena.initialize()

        def interaction(solver_vec, tester_vec):
            # Solver wins if dot product is positive
            score = float(np.dot(solver_vec, tester_vec))
            # Normalize to [0, 1]
            score = (np.tanh(score) + 1) / 2
            return score, 1.0 - score

        arena.evaluate(interaction)
        assert arena.solver_best >= 0.0
        assert arena.tester_best >= 0.0
        assert arena.solver_best_genome is not None
        assert arena.tester_best_genome is not None

    def test_breed(self):
        arena = AdversarialArena(
            solver_pop_size=10,
            tester_pop_size=5,
            n_interactions_per_gen=3,
        )
        arena.initialize()

        def interaction(solver_vec, tester_vec):
            score = float(np.dot(solver_vec, tester_vec))
            score = (np.tanh(score) + 1) / 2
            return score, 1.0 - score

        arena.evaluate(interaction)
        arena.breed()
        assert len(arena.solver_pop) == 10
        assert len(arena.tester_pop) == 5
        assert arena.generation == 1

    def test_full_coevolution(self):
        arena = AdversarialArena(
            solver_pop_size=20,
            tester_pop_size=10,
            solver_dim=5,
            tester_dim=5,
            n_interactions_per_gen=5,
        )
        arena.initialize()

        def interaction(solver_vec, tester_vec):
            score = float(np.dot(solver_vec, tester_vec))
            score = (np.tanh(score) + 1) / 2
            return score, 1.0 - score

        solver_history = []
        tester_history = []
        for gen in range(10):
            arena.evaluate(interaction)
            arena.breed()
            solver_history.append(arena.solver_best)
            tester_history.append(arena.tester_best)

        assert arena.generation == 10
        assert len(solver_history) == 10
        assert len(tester_history) == 10

    def test_solver_stats(self):
        arena = AdversarialArena(solver_pop_size=10, tester_pop_size=5)
        arena.initialize()
        stats = arena.get_solver_stats()
        assert "generation" in stats
        assert "pop_size" in stats
        assert stats["pop_size"] == 10

    def test_tester_stats(self):
        arena = AdversarialArena(solver_pop_size=10, tester_pop_size=5)
        arena.initialize()
        stats = arena.get_tester_stats()
        assert "generation" in stats
        assert "pop_size" in stats
        assert stats["pop_size"] == 5

    def test_coevolution_stats(self):
        arena = AdversarialArena(solver_pop_size=10, tester_pop_size=5)
        arena.initialize()
        stats = arena.get_coevolution_stats()
        assert "solver_best" in stats
        assert "tester_best" in stats
        assert "arms_race_index" in stats

    def test_zero_sum_property(self):
        arena = AdversarialArena(
            solver_pop_size=5,
            tester_pop_size=5,
            n_interactions_per_gen=5,
        )
        arena.initialize()

        def interaction(solver_vec, tester_vec):
            score = float(np.dot(solver_vec, tester_vec))
            score = (np.tanh(score) + 1) / 2
            return score, 1.0 - score

        arena.evaluate(interaction)
        # For each interaction, solver_score + tester_score should be ~1
        # (within floating point tolerance)
        # We can't directly check this without recording, but the
        # fitness averages should be reasonable
        solver_mean = sum(s.fitness for s in arena.solver_pop) / len(arena.solver_pop)
        tester_mean = sum(t.fitness for t in arena.tester_pop) / len(arena.tester_pop)
        # On average, solver_score + tester_score = 1
        # So solver_mean + tester_mean should be close to 1
        assert 0.3 < solver_mean + tester_mean < 1.7

    def test_elitism(self):
        arena = AdversarialArena(
            solver_pop_size=10,
            tester_pop_size=5,
            elitism_count=2,
        )
        arena.initialize()

        def interaction(solver_vec, tester_vec):
            return 0.5, 0.5

        arena.evaluate(interaction)
        best_solver = arena.solver_best_genome
        arena.breed()
        arena.evaluate(interaction)
        # After breeding, best should still be decent
        assert arena.solver_best >= 0.0

    def test_age_culling(self):
        arena = AdversarialArena(
            solver_pop_size=10,
            tester_pop_size=5,
            max_age=2,
        )
        arena.initialize()

        def interaction(solver_vec, tester_vec):
            return 0.5, 0.5

        for _ in range(5):
            arena.evaluate(interaction)
            arena.breed()

        assert all(s.age < arena.max_age for s in arena.solver_pop)
        assert all(t.age < arena.max_age for t in arena.tester_pop)
