"""Tests for NCA Breeder — Neural Cellular Automata indirect encoding.

Covers NCARule, NCAMutation, NCACrossover, and NCABreeder.
"""

import numpy as np
import pytest

from swarm.nca_breeder import (
    NCARule,
    NCAMutation,
    NCACrossover,
    NCABreeder,
)


class TestNCARule:
    def test_random(self):
        rule = NCARule.random(n_channels=3, n_kernels=3, kernel_size=3, n_steps=16)
        assert rule.n_channels == 3
        assert rule.n_kernels == 3
        assert rule.kernel_size == 3
        assert rule.kernel_weights.shape == (3, 3, 3, 3)
        assert rule.update_weights.shape == (3, 9, 1, 1)
        assert len(rule.update_bias) == 3

    def test_copy(self):
        rule = NCARule.random(n_channels=3, n_steps=16)
        rule.fitness = 1.5
        copy = rule.copy()
        assert copy.fitness == 1.5
        assert np.allclose(copy.kernel_weights, rule.kernel_weights)
        # Ensure independent
        copy.kernel_weights[0, 0, 0, 0] = 999
        assert rule.kernel_weights[0, 0, 0, 0] != 999

    def test_grow(self):
        rule = NCARule.random(n_channels=3, n_kernels=3, kernel_size=3, n_steps=8)
        phenotype = rule.grow()
        assert phenotype.shape == (3, 16, 16)  # Default seed size

    def test_grow_custom_seed(self):
        rule = NCARule.random(n_channels=3, n_steps=4)
        seed = np.random.randn(3, 8, 8) * 0.1
        phenotype = rule.grow(seed=seed)
        assert phenotype.shape == (3, 8, 8)

    def test_phenotype_real(self):
        rule = NCARule.random(n_channels=3, n_steps=8)
        phenotype = rule.grow()
        # Should be real-valued
        assert np.isrealobj(phenotype)
        assert not np.any(np.isnan(phenotype))
        assert not np.any(np.isinf(phenotype))

    def test_perceive_shape(self):
        rule = NCARule.random(n_channels=3, n_kernels=3)
        state = np.random.randn(3, 16, 16)
        perceptions = rule._perceive(state)
        assert perceptions.shape == (3, 3, 16, 16)

    def test_update_shape(self):
        rule = NCARule.random(n_channels=3, n_kernels=3)
        perceptions = np.random.randn(3, 3, 16, 16)
        delta = rule._update(perceptions)
        assert delta.shape == (3, 16, 16)

    def test_conv2d(self):
        rule = NCARule.random(n_channels=1, n_kernels=1, kernel_size=3)
        image = np.random.randn(16, 16)
        kernel = np.ones((3, 3)) / 9.0
        result = rule._conv2d(image, kernel)
        assert result.shape == (16, 16)

    def test_sobel_kernels_initialized(self):
        rule = NCARule.random(n_channels=3, n_kernels=3, kernel_size=3)
        # Check Sobel x is present
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        assert np.allclose(rule.kernel_weights[1, 0], sobel_x)
        # Check Sobel y is present
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        assert np.allclose(rule.kernel_weights[2, 0], sobel_y)


class TestNCAMutation:
    def test_mutate(self):
        rule = NCARule.random(n_channels=3, n_steps=8)
        mutator = NCAMutation(
            kernel_rate=1.0,
            weight_rate=1.0,
            bias_rate=1.0,
            kernel_strength=0.1,
            weight_strength=0.1,
            bias_strength=0.1,
        )
        child = mutator.mutate(rule)
        assert child.n_channels == 3
        # Should be different from parent
        assert not np.allclose(child.kernel_weights, rule.kernel_weights)

    def test_mutate_structure_preserved(self):
        rule = NCARule.random(n_channels=3, n_kernels=3, n_steps=8)
        mutator = NCAMutation(kernel_rate=0.5, weight_rate=0.5, bias_rate=0.5)
        child = mutator.mutate(rule)
        assert child.kernel_weights.shape == rule.kernel_weights.shape
        assert child.update_weights.shape == rule.update_weights.shape
        assert len(child.update_bias) == len(rule.update_bias)


class TestNCACrossover:
    def test_crossover(self):
        p1 = NCARule.random(n_channels=3, n_steps=8)
        p2 = NCARule.random(n_channels=3, n_steps=8)
        xo = NCACrossover(alpha=0.5)
        c1, c2 = xo.crossover(p1, p2)
        assert c1.n_channels == 3
        assert c2.n_channels == 3
        assert c1.kernel_weights.shape == p1.kernel_weights.shape

    def test_crossover_parameters_mixed(self):
        p1 = NCARule.random(n_channels=3, n_steps=8)
        p2 = NCARule.random(n_channels=3, n_steps=8)
        # Set different values
        p1.kernel_weights[:] = 1.0
        p2.kernel_weights[:] = 2.0
        xo = NCACrossover(alpha=0.5)
        c1, c2 = xo.crossover(p1, p2)
        # Children should have mix of values
        assert not np.allclose(c1.kernel_weights, p1.kernel_weights)
        assert not np.allclose(c1.kernel_weights, p2.kernel_weights)


class TestNCABreeder:
    def test_initialize(self):
        breeder = NCABreeder(
            population_size=10,
            n_channels=3,
            n_steps=8,
        )
        breeder.initialize()
        assert len(breeder.population) == 10
        for rule in breeder.population:
            assert rule.n_channels == 3

    def test_evaluate(self):
        breeder = NCABreeder(
            population_size=5,
            n_channels=3,
            n_steps=4,
        )
        breeder.initialize()

        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))

        breeder.evaluate(task_fn)
        assert breeder.best_fitness >= 0
        assert breeder.best_rule is not None
        assert all(r.fitness >= 0 for r in breeder.population)

    def test_select_and_breed(self):
        breeder = NCABreeder(
            population_size=5,
            n_channels=3,
            n_steps=4,
        )
        breeder.initialize()

        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))

        breeder.evaluate(task_fn)
        breeder.select_and_breed()
        assert len(breeder.population) == 5
        assert breeder.generation == 1

    def test_full_evolution(self):
        breeder = NCABreeder(
            population_size=5,
            n_channels=3,
            n_steps=4,
            mutation_rate=0.5,
            crossover_rate=0.5,
        )
        breeder.initialize()

        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))

        best_history = []
        for gen in range(3):
            breeder.evaluate(task_fn)
            breeder.select_and_breed()
            best_history.append(breeder.best_fitness)

        assert breeder.generation == 3
        assert len(best_history) == 3

    def test_stats(self):
        breeder = NCABreeder(population_size=5, n_channels=3, n_steps=4)
        breeder.initialize()
        stats = breeder.get_stats()
        assert stats["generation"] == 0
        assert stats["population_size"] == 5

    def test_elitism(self):
        breeder = NCABreeder(
            population_size=5,
            n_channels=3,
            n_steps=4,
            elitism_count=2,
        )
        breeder.initialize()

        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))

        breeder.evaluate(task_fn)
        best_before = breeder.best_fitness
        breeder.select_and_breed()
        breeder.evaluate(task_fn)
        assert breeder.best_fitness >= best_before * 0.5

    def test_age_culling(self):
        breeder = NCABreeder(
            population_size=5,
            n_channels=3,
            n_steps=4,
            max_age=2,
        )
        breeder.initialize()

        def task_fn(phenotype):
            return float(np.sum(phenotype ** 2))

        for _ in range(5):
            breeder.evaluate(task_fn)
            breeder.select_and_breed()

        assert all(r.age < 2 for r in breeder.population)

    def test_different_channels(self):
        for n_ch in [1, 3, 5]:
            breeder = NCABreeder(
                population_size=3,
                n_channels=n_ch,
                n_steps=4,
            )
            breeder.initialize()
            assert all(r.n_channels == n_ch for r in breeder.population)

    def test_band_limit_not_applicable(self):
        # NCA doesn't use band limiting, but ensure it still works
        breeder = NCABreeder(population_size=3, n_channels=3, n_steps=4)
        breeder.initialize()
        assert len(breeder.population) == 3

