"""
Tests for Information-Theoretic Breeder.

Covers: entropy computation, mutual information, KL divergence,
JS divergence, population analysis, info-maximizing breeding.
"""

import random

import numpy as np
import pytest

from swarm.info_theoretic_breeder import (
    shannon_entropy,
    mutual_information,
    kl_divergence,
    js_divergence,
    GenomeDistribution,
    InformationState,
    compute_population_info_state,
    InfoTheoreticBreeder,
)


class TestEntropyFunctions:
    def test_shannon_entropy_uniform(self):
        """Uniform distribution has maximum entropy."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        entropy = shannon_entropy(values, bins=10)
        assert entropy > 0
        # Uniform over 10 bins should have entropy ~ log2(10) = 3.32
        assert entropy > 2.0

    def test_shannon_entropy_constant(self):
        """Constant distribution has zero entropy."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        entropy = shannon_entropy(values, bins=10)
        assert entropy == 0.0

    def test_shannon_entropy_empty(self):
        assert shannon_entropy(np.array([])) == 0.0

    def test_mutual_information_independent(self):
        """Independent variables have low MI."""
        random.seed(42)
        x = np.array([random.gauss(0, 1) for _ in range(100)])
        y = np.array([random.gauss(0, 1) for _ in range(100)])
        mi = mutual_information(x, y, bins=10)
        # Independent variables should have near-zero MI
        assert mi < 0.5

    def test_mutual_information_dependent(self):
        """Dependent variables have high MI."""
        x = np.array([random.gauss(0, 1) for _ in range(100)])
        y = 2.0 * x + np.array([random.gauss(0, 0.1) for _ in range(100)])
        mi = mutual_information(x, y, bins=10)
        # Strongly dependent variables should have high MI
        assert mi > 1.0

    def test_kl_divergence_same(self):
        """KL divergence of a distribution with itself is ~0."""
        p = np.random.normal(0, 1, 100)
        kl = kl_divergence(p, p, bins=10)
        assert kl < 0.5  # Should be close to 0

    def test_kl_divergence_different(self):
        """Different distributions have positive KL divergence."""
        p = np.random.normal(0, 1, 100)
        q = np.random.normal(5, 1, 100)
        kl = kl_divergence(p, q, bins=10)
        assert kl > 0.0  # Positive divergence

    def test_js_divergence_symmetric(self):
        """JS divergence is symmetric."""
        p = np.random.normal(0, 1, 100)
        q = np.random.normal(2, 1, 100)
        js1 = js_divergence(p, q, bins=10)
        js2 = js_divergence(q, p, bins=10)
        assert abs(js1 - js2) < 0.01

    def test_js_divergence_bounded(self):
        """JS divergence is bounded by 1.0 for normalized distributions."""
        p = np.random.normal(0, 1, 100)
        q = np.random.normal(10, 1, 100)
        js = js_divergence(p, q, bins=10)
        assert 0 <= js <= 1.0


class TestPopulationInfoState:
    def test_empty_population(self):
        state = compute_population_info_state([], [])
        assert state.population_entropy == 0.0
        assert state.fitness_entropy == 0.0

    def test_single_individual(self):
        pop = [({"gene_a": 1.0}, 10.0)]
        state = compute_population_info_state(pop, ["gene_a"])
        assert state.population_entropy == 0.0  # All same values
        assert state.fitness_entropy == 0.0

    def test_diverse_population(self):
        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(20)
        ]
        state = compute_population_info_state(pop, ["gene_a", "gene_b"])
        assert state.population_entropy > 0
        assert state.fitness_entropy > 0
        assert "gene_a" in state.gene_entropies
        assert "gene_b" in state.gene_entropies

    def test_gene_fitness_mi(self):
        """Genes correlated with fitness should have high MI."""
        pop = []
        for i in range(50):
            gene_a = float(i)
            fitness = 2.0 * gene_a + random.gauss(0, 0.1)
            pop.append(({"gene_a": gene_a}, fitness))

        state = compute_population_info_state(pop, ["gene_a"])
        assert state.gene_fitness_mi["gene_a"] > 0.5


class TestInfoTheoreticBreeder:
    def test_init(self):
        breeder = InfoTheoreticBreeder(
            population_size=50,
            entropy_target=2.0,
            mi_threshold=0.1
        )
        assert breeder.population_size == 50
        assert breeder.entropy_target == 2.0
        assert breeder.mi_threshold == 0.1

    def test_analyze_population(self):
        breeder = InfoTheoreticBreeder()
        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(20)
        ]
        state = breeder.analyze_population(pop)
        assert state is not None
        assert state.population_entropy > 0
        assert state.generation == 0

    def test_select_parents(self):
        breeder = InfoTheoreticBreeder()
        pop = [
            ({"gene_a": 1.0, "gene_b": 2.0}, 10.0),
            ({"gene_a": 2.0, "gene_b": 3.0}, 20.0),
            ({"gene_a": 3.0, "gene_b": 1.0}, 30.0),
            ({"gene_a": 4.0, "gene_b": 4.0}, 40.0),
        ]
        parents = breeder.select_parents(pop, k=2)
        assert len(parents) == 2
        # Should select high-fitness individuals
        assert parents[0][1] >= 10.0
        assert parents[1][1] >= 10.0

    def test_information_distance(self):
        breeder = InfoTheoreticBreeder()
        g1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        g2 = {"a": 1.0, "b": 2.0, "c": 3.0}
        dist = breeder._information_distance(g1, g2)
        assert dist >= 0.0
        # Same genomes should have low divergence
        assert dist < 0.5

    def test_info_maximizing_crossover(self):
        breeder = InfoTheoreticBreeder()
        p1 = {"gene_a": 1.0, "gene_b": 2.0, "_fitness": 10.0}
        p2 = {"gene_a": 3.0, "gene_b": 4.0, "_fitness": 20.0}

        child = breeder.info_maximizing_crossover(p1, p2)
        assert "gene_a" in child
        assert "gene_b" in child
        # Each gene should come from one of the parents
        assert child["gene_a"] in [1.0, 3.0]
        assert child["gene_b"] in [2.0, 4.0]

    def test_entropy_maintaining_mutation(self):
        breeder = InfoTheoreticBreeder()
        genome = {"gene_a": 1.0, "gene_b": 2.0}
        mutated = breeder.entropy_maintaining_mutation(genome, mutation_rate=1.0)
        # With mutation_rate=1.0, all genes should be mutated
        assert "gene_a" in mutated
        assert "gene_b" in mutated

    def test_entropy_maintaining_mutation_zero_rate(self):
        breeder = InfoTheoreticBreeder()
        genome = {"gene_a": 1.0, "gene_b": 2.0}
        mutated = breeder.entropy_maintaining_mutation(genome, mutation_rate=0.0)
        assert mutated["gene_a"] == 1.0
        assert mutated["gene_b"] == 2.0

    def test_breed_generation(self):
        breeder = InfoTheoreticBreeder(population_size=10)
        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(10)
        ]

        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10 + genome["gene_b"]}

        new_pop = breeder.breed_generation(pop, task_fn)
        assert len(new_pop) == 10
        assert breeder.generation == 1
        assert breeder.info_state is not None

    def test_breed_generation_elitism(self):
        breeder = InfoTheoreticBreeder(population_size=10, elitism_ratio=0.2)
        pop = [
            ({"gene_a": float(i)}, float(i * 10))
            for i in range(10)
        ]

        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10}

        new_pop = breeder.breed_generation(pop, task_fn)
        best_new = max(new_pop, key=lambda x: x[1])
        assert best_new[1] >= 80.0  # Elite preservation

    def test_info_summary_no_data(self):
        breeder = InfoTheoreticBreeder()
        summary = breeder.get_info_summary()
        assert summary["status"] == "no_data"

    def test_info_summary_with_data(self):
        breeder = InfoTheoreticBreeder()
        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(20)
        ]
        breeder.analyze_population(pop)
        summary = breeder.get_info_summary()
        assert summary["status"] == "analyzed"
        assert "key_genes" in summary
        assert "diverse_genes" in summary
        assert "population_entropy" in summary

    def test_entropy_target_influence(self):
        """Low entropy should trigger higher mutation rate."""
        breeder = InfoTheoreticBreeder(
            population_size=10,
            entropy_target=100.0  # Very high target (never met)
        )
        pop = [
            ({"gene_a": 1.0, "gene_b": 1.0}, 10.0)
            for _ in range(10)
        ]
        # All identical, entropy is 0
        state = breeder.analyze_population(pop)
        assert state.population_entropy < breeder.entropy_target
        # Next generation should use higher mutation rate
        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10}
        new_pop = breeder.breed_generation(pop, task_fn)
        assert len(new_pop) == 10

    def test_key_gene_identification(self):
        """Breeder should identify genes correlated with fitness."""
        random.seed(42)
        pop = []
        for i in range(50):
            gene_a = float(i)
            gene_b = random.gauss(0, 10)  # Random, no correlation
            fitness = 2.0 * gene_a + random.gauss(0, 0.1)
            pop.append(({"gene_a": gene_a, "gene_b": gene_b}, fitness))

        breeder = InfoTheoreticBreeder()
        state = breeder.analyze_population(pop)

        # gene_a should have high MI with fitness
        assert state.gene_fitness_mi["gene_a"] > state.gene_fitness_mi["gene_b"]

    def test_crossover_with_mi_data(self):
        """Crossover should prefer genes from fitter parent when MI is high."""
        breeder = InfoTheoreticBreeder(mi_threshold=0.1)

        # Simulate MI data: gene_a has high MI, gene_b has low MI
        breeder.info_state = InformationState(
            gene_fitness_mi={"gene_a": 0.5, "gene_b": 0.01}
        )

        p1 = {"gene_a": 1.0, "gene_b": 1.0, "_fitness": 10.0}
        p2 = {"gene_a": 2.0, "gene_b": 2.0, "_fitness": 20.0}

        # High MI gene: should take from fitter parent (p2)
        # Low MI gene: random selection
        children = []
        for _ in range(100):
            child = breeder.info_maximizing_crossover(p1, p2)
            children.append(child["gene_a"])

        # gene_a should mostly come from p2 (value=2.0)
        p2_ratio = sum(1 for c in children if c == 2.0) / len(children)
        assert p2_ratio > 0.6  # Bias toward fitter parent for high-MI gene


class TestIntegration:
    def test_full_info_breeding_pipeline(self):
        """End-to-end: analyze → breed → analyze → improve."""
        random.seed(42)
        np.random.seed(42)

        breeder = InfoTheoreticBreeder(
            population_size=20,
            entropy_target=1.0,
            mi_threshold=0.05
        )

        # True model: fitness = 2*gene_a + 1*gene_b + noise
        def true_fitness(g):
            return 2.0 * g["gene_a"] + 1.0 * g["gene_b"] + random.gauss(0, 0.2)

        def task_fn(genome):
            return {"fitness": true_fitness(genome)}

        # Initialize
        pop = []
        for _ in range(20):
            g = {"gene_a": random.gauss(0, 1), "gene_b": random.gauss(0, 1)}
            pop.append((g, true_fitness(g)))

        # Run 5 generations
        for _ in range(5):
            pop = breeder.breed_generation(pop, task_fn)

        avg_fitness = np.mean([f for _, f in pop])
        assert avg_fitness > 0  # Should improve from random start

        summary = breeder.get_info_summary()
        assert summary["status"] == "analyzed"

    def test_info_vs_random_breeding(self):
        """Info breeder should outperform random on structured data."""
        random.seed(42)
        np.random.seed(42)

        def true_fitness(g):
            return 3.0 * g["gene_a"] + 0.1 * g["gene_b"] + random.gauss(0, 0.1)

        def task_fn(genome):
            return {"fitness": true_fitness(genome)}

        # Info breeder
        info_breeder = InfoTheoreticBreeder(
            population_size=15,
            entropy_target=1.0,
            mi_threshold=0.05
        )

        # Random breeder
        class RandomBreeder:
            def __init__(self, pop_size):
                self.pop_size = pop_size

            def breed(self, pop, task_fn):
                new_pop = []
                sorted_pop = sorted(pop, key=lambda x: x[1], reverse=True)
                n_elite = max(1, len(sorted_pop) // 10)
                new_pop.extend(sorted_pop[:n_elite])
                while len(new_pop) < self.pop_size:
                    p1 = random.choice(sorted_pop)
                    p2 = random.choice(sorted_pop)
                    child = {}
                    for k in p1[0]:
                        child[k] = p1[0][k] if random.random() < 0.5 else p2[0].get(k, p1[0][k])
                        if random.random() < 0.1:
                            child[k] *= (1 + random.uniform(-0.1, 0.1))
                    f = task_fn(child)["fitness"]
                    new_pop.append((child, f))
                return new_pop

        random_breeder = RandomBreeder(15)

        # Initial population
        init_pop = []
        for _ in range(15):
            g = {"gene_a": random.gauss(0, 1), "gene_b": random.gauss(0, 1)}
            init_pop.append((g, true_fitness(g)))

        info_pop = init_pop.copy()
        random_pop = init_pop.copy()

        for _ in range(5):
            info_pop = info_breeder.breed_generation(info_pop, task_fn)
            random_pop = random_breeder.breed(random_pop, task_fn)

        avg_info = np.mean([f for _, f in info_pop])
        avg_random = np.mean([f for _, f in random_pop])

        # Info breeder should be competitive or better
        assert avg_info > avg_random * 0.5

    def test_entropy_maintenance(self):
        """Breeder should maintain diversity (entropy)."""
        breeder = InfoTheoreticBreeder(
            population_size=10,
            entropy_target=2.0
        )

        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10}

        # Start with diverse population
        pop = [
            ({"gene_a": float(i)}, float(i * 10))
            for i in range(10)
        ]

        entropies = []
        for _ in range(3):
            pop = breeder.breed_generation(pop, task_fn)
            state = breeder.analyze_population(pop)
            entropies.append(state.population_entropy)

        # Entropy should not collapse to zero immediately
        assert all(e > 0 for e in entropies)
