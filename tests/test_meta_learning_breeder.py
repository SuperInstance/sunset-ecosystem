"""
Tests for Meta-Learning Breeder.

Covers: ProblemFeatures, StrategyConfig, StrategyPerformance,
MetaLearningModel, MetaLearningBreeder.
"""

import random

import numpy as np
import pytest

from swarm.meta_learning_breeder import (
    ProblemFeatures,
    StrategyConfig,
    StrategyPerformance,
    MetaLearningModel,
    MetaLearningBreeder,
)


class TestProblemFeatures:
    def test_to_vector(self):
        f = ProblemFeatures(
            dimensionality=10,
            modality="multimodal",
            noise_level=0.1,
            separable=False,
            evaluation_cost=5.0
        )
        vec = f.to_vector()
        assert len(vec) == 5
        assert vec[0] == 10.0
        assert vec[1] == 1.0  # multimodal
        assert vec[2] == 0.1
        assert vec[3] == 0.0  # not separable
        assert vec[4] == 5.0

    def test_unknown_modality(self):
        f = ProblemFeatures(modality="unknown")
        vec = f.to_vector()
        assert vec[1] == -1.0


class TestStrategyConfig:
    def test_to_vector(self):
        c = StrategyConfig(
            mutation_rate=0.2,
            crossover_rate=0.9,
            mutation_type="cauchy",
            crossover_type="blend",
            selection_type="rank",
            population_size=100
        )
        vec = c.to_vector()
        assert len(vec) == 9
        assert vec[0] == 0.2
        assert vec[2] == 2.0  # cauchy
        assert vec[3] == 3.0  # blend
        assert vec[4] == 2.0  # rank
        assert vec[6] == 100.0

    def test_from_vector(self):
        vec = np.array([0.15, 0.85, 1.0, 2.0, 0.0, 0.15, 80.0, 1.0, 0.5])
        c = StrategyConfig.from_vector(vec)
        assert c.mutation_rate == 0.15
        assert c.crossover_type == "two_point"
        assert c.selection_type == "tournament"
        assert c.population_size == 80

    def test_clamping(self):
        vec = np.array([-1.0, 2.0, 10.0, -5.0, 100.0, -0.5, 5.0, 0.0, 2.0])
        c = StrategyConfig.from_vector(vec)
        assert c.mutation_rate == 0.001  # clamped to min
        assert c.crossover_rate == 1.0  # clamped to max
        assert c.population_size == 10  # clamped to min


class TestMetaLearningModel:
    def test_empty_prediction(self):
        model = MetaLearningModel()
        pred = model.predict_performance(
            ProblemFeatures(), StrategyConfig()
        )
        assert pred == 0.5  # Neutral

    def test_add_example(self):
        model = MetaLearningModel()
        perf = StrategyPerformance(
            strategy_config=StrategyConfig(mutation_rate=0.1),
            problem_features=ProblemFeatures(dimensionality=5),
            final_fitness=100.0,
            convergence_speed=10.0,
            success_rate=0.9
        )
        model.add_example(perf)
        assert len(model.examples) == 1

    def test_predict_performance(self):
        model = MetaLearningModel(k=2)
        # Add examples
        for i in range(10):
            model.add_example(StrategyPerformance(
                strategy_config=StrategyConfig(mutation_rate=0.1 + i * 0.01),
                problem_features=ProblemFeatures(dimensionality=5, noise_level=0.1),
                final_fitness=50.0 + i * 10,
                convergence_speed=10.0,
                success_rate=0.9
            ))

        pred = model.predict_performance(
            ProblemFeatures(dimensionality=5, noise_level=0.1),
            StrategyConfig(mutation_rate=0.15)
        )
        assert pred > 0

    def test_recommend_strategy(self):
        model = MetaLearningModel()
        for i in range(20):
            model.add_example(StrategyPerformance(
                strategy_config=StrategyConfig(mutation_rate=0.1 + i * 0.05),
                problem_features=ProblemFeatures(dimensionality=10),
                final_fitness=100.0 if i > 10 else 50.0,
                convergence_speed=10.0,
                success_rate=0.9
            ))

        rec = model.recommend_strategy(
            ProblemFeatures(dimensionality=10),
            n_candidates=5
        )
        assert rec is not None
        assert rec.mutation_rate > 0

    def test_problem_classes(self):
        model = MetaLearningModel()
        model.add_example(StrategyPerformance(
            strategy_config=StrategyConfig(),
            problem_features=ProblemFeatures(modality="unimodal"),
            final_fitness=100.0, convergence_speed=10.0, success_rate=0.9
        ))
        model.add_example(StrategyPerformance(
            strategy_config=StrategyConfig(),
            problem_features=ProblemFeatures(modality="multimodal"),
            final_fitness=80.0, convergence_speed=20.0, success_rate=0.7
        ))

        classes = model.get_problem_classes()
        assert "unimodal" in classes
        assert "multimodal" in classes

    def test_best_strategy_per_class(self):
        model = MetaLearningModel()
        model.add_example(StrategyPerformance(
            strategy_config=StrategyConfig(mutation_rate=0.1),
            problem_features=ProblemFeatures(modality="unimodal"),
            final_fitness=100.0, convergence_speed=10.0, success_rate=0.9
        ))
        model.add_example(StrategyPerformance(
            strategy_config=StrategyConfig(mutation_rate=0.5),
            problem_features=ProblemFeatures(modality="unimodal"),
            final_fitness=50.0, convergence_speed=10.0, success_rate=0.9
        ))

        best = model.get_best_strategy_per_class()
        assert best["unimodal"].mutation_rate == 0.1

    def test_to_dict(self):
        model = MetaLearningModel()
        model.add_example(StrategyPerformance(
            strategy_config=StrategyConfig(mutation_rate=0.1),
            problem_features=ProblemFeatures(),
            final_fitness=100.0, convergence_speed=10.0, success_rate=0.9
        ))
        d = model.to_dict()
        assert d["n_examples"] == 1
        assert "best_strategies" in d


class TestMetaLearningBreeder:
    def test_init(self):
        breeder = MetaLearningBreeder(population_size=20)
        assert breeder.population_size == 20
        assert breeder.meta_model is not None

    def test_extract_problem_features(self):
        breeder = MetaLearningBreeder()
        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(20)
        ]
        features = breeder.extract_problem_features(pop)
        assert features.dimensionality == 2
        assert features.noise_level >= 0

    def test_extract_multimodal(self):
        breeder = MetaLearningBreeder()
        # Create population with high local variance
        pop = []
        for i in range(20):
            f = 100.0 if i == 10 else random.gauss(0, 1)
            pop.append(({"g": float(i)}, f))
        features = breeder.extract_problem_features(pop)
        assert features.modality == "multimodal"

    def test_breed_generation(self):
        breeder = MetaLearningBreeder(population_size=10)
        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(10)
        ]

        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10 + genome["gene_b"]}

        new_pop = breeder.breed_generation(pop, task_fn)
        assert len(new_pop) == 10
        assert breeder.generation == 1

    def test_strategy_adaptation(self):
        breeder = MetaLearningBreeder(
            population_size=10,
            adaptation_interval=1
        )
        # Add some examples to meta model
        for i in range(10):
            breeder.meta_model.add_example(StrategyPerformance(
                strategy_config=StrategyConfig(mutation_rate=0.1 + i * 0.05),
                problem_features=ProblemFeatures(dimensionality=2),
                final_fitness=50.0 + i * 5,
                convergence_speed=10.0, success_rate=0.9
            ))

        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(10)
        ]

        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10 + genome["gene_b"]}

        old_rate = breeder.current_strategy.mutation_rate
        breeder.breed_generation(pop, task_fn)
        # Strategy may have adapted
        assert breeder.generation == 1

    def test_mutation_types(self):
        breeder = MetaLearningBreeder()
        genome = {"gene_a": 1.0, "gene_b": 2.0}

        for mut_type in ["gaussian", "uniform", "cauchy", "adaptive"]:
            breeder.current_strategy.mutation_type = mut_type
            mutated = breeder.mutate(genome.copy())
            assert "gene_a" in mutated
            assert "gene_b" in mutated

    def test_crossover_types(self):
        breeder = MetaLearningBreeder()
        p1 = {"gene_a": 1.0, "gene_b": 2.0}
        p2 = {"gene_a": 3.0, "gene_b": 4.0}

        for cross_type in ["uniform", "blend", "one_point"]:
            breeder.current_strategy.crossover_type = cross_type
            child = breeder.crossover(p1, p2)
            assert "gene_a" in child
            assert "gene_b" in child

    def test_selection_types(self):
        breeder = MetaLearningBreeder()
        pop = [
            ({"g": 1.0}, 10.0),
            ({"g": 2.0}, 50.0),
            ({"g": 3.0}, 30.0),
        ]

        for sel_type in ["tournament", "roulette", "rank"]:
            breeder.current_strategy.selection_type = sel_type
            parents = breeder.select_parents(pop, k=2)
            assert len(parents) == 2

    def test_get_meta_summary(self):
        breeder = MetaLearningBreeder()
        summary = breeder.get_meta_summary()
        assert summary["generation"] == 0
        assert summary["meta_model_examples"] == 0

    def test_record_final_performance(self):
        breeder = MetaLearningBreeder()
        breeder.history.append((10.0, StrategyConfig()))
        breeder.record_final_performance(100.0, 10.0, 0.9)
        assert len(breeder.meta_model.examples) == 1

    def test_to_dict(self):
        breeder = MetaLearningBreeder()
        d = breeder.get_meta_summary()
        assert "current_strategy" in d
        assert "history_length" in d

    def test_crossover_missing_gene(self):
        breeder = MetaLearningBreeder()
        p1 = {"gene_a": 1.0, "gene_b": 2.0}
        p2 = {"gene_a": 3.0}
        child = breeder.crossover(p1, p2)
        assert child["gene_a"] in [1.0, 3.0]
        assert child["gene_b"] == 2.0

    def test_extract_separable(self):
        breeder = MetaLearningBreeder()
        pop = []
        for i in range(20):
            a = float(i)
            b = random.gauss(0, 1)
            f = 2.0 * a + 0.1 * b
            pop.append(({"gene_a": a, "gene_b": b}, f))
        features = breeder.extract_problem_features(pop)
        # gene_a is strongly correlated with fitness, so separable
        assert bool(features.separable) is True
