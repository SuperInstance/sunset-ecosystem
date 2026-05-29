"""
Meta-Learning Breeder

Learns which breeding strategies (mutation rates, crossover types, selection
methods) work best for different problem classes. Maintains a meta-level model
that predicts which configuration will perform best on a new problem based on
its features.

Key innovations:
- Problem feature extraction: dimensionality, modality, noise level, etc.
- Strategy performance tracking: which strategy worked on which problem class
- Meta-optimization: optimize the optimizer's hyperparameters
- Transfer learning: apply knowledge from solved problems to new ones

References:
- Schmidhuber (1987) - Evolutionary Principles in Self-Referential Learning
- Vanschoren (2018) - Meta-Learning: A Survey
- Hutter et al. (2019) - Automated Machine Learning: Methods, Systems, Challenges
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProblemFeatures:
    """Features extracted from a problem to identify its class."""
    dimensionality: int = 1
    modality: str = "unknown"  # unimodal, multimodal, deceptive
    noise_level: float = 0.0
    separable: bool = True
    bounds: List[Tuple[float, float]] = field(default_factory=list)
    known_optimum: Optional[float] = None
    evaluation_cost: float = 1.0  # Cost per evaluation

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for meta-learning."""
        modal_map = {"unimodal": 0.0, "multimodal": 1.0, "deceptive": 2.0, "unknown": -1.0}
        return np.array([
            float(self.dimensionality),
            modal_map.get(self.modality, -1.0),
            self.noise_level,
            1.0 if self.separable else 0.0,
            self.evaluation_cost,
        ])


@dataclass
class StrategyConfig:
    """A breeding strategy configuration."""
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    mutation_type: str = "gaussian"  # gaussian, uniform, cauchy, adaptive
    crossover_type: str = "uniform"  # uniform, one_point, two_point, blend
    selection_type: str = "tournament"  # tournament, roulette, rank
    elitism_ratio: float = 0.1
    population_size: int = 50
    # Dynamic parameters
    mutation_decay: float = 1.0  # 1.0 = no decay, <1.0 = decay over time
    diversity_pressure: float = 0.0  # 0.0 = none, 1.0 = strong

    def to_vector(self) -> np.ndarray:
        """Convert to configuration vector."""
        mut_map = {"gaussian": 0.0, "uniform": 1.0, "cauchy": 2.0, "adaptive": 3.0}
        cross_map = {"uniform": 0.0, "one_point": 1.0, "two_point": 2.0, "blend": 3.0}
        sel_map = {"tournament": 0.0, "roulette": 1.0, "rank": 2.0}
        return np.array([
            self.mutation_rate,
            self.crossover_rate,
            mut_map.get(self.mutation_type, 0.0),
            cross_map.get(self.crossover_type, 0.0),
            sel_map.get(self.selection_type, 0.0),
            self.elitism_ratio,
            float(self.population_size),
            self.mutation_decay,
            self.diversity_pressure,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "StrategyConfig":
        """Create from configuration vector."""
        mut_types = ["gaussian", "uniform", "cauchy", "adaptive"]
        cross_types = ["uniform", "one_point", "two_point", "blend"]
        sel_types = ["tournament", "roulette", "rank"]
        return cls(
            mutation_rate=max(0.001, min(1.0, v[0])),
            crossover_rate=max(0.0, min(1.0, v[1])),
            mutation_type=mut_types[int(v[2]) % len(mut_types)],
            crossover_type=cross_types[int(v[3]) % len(cross_types)],
            selection_type=sel_types[int(v[4]) % len(sel_types)],
            elitism_ratio=max(0.0, min(1.0, v[5])),
            population_size=max(10, int(v[6])),
            mutation_decay=max(0.5, min(1.5, v[7])),
            diversity_pressure=max(0.0, min(1.0, v[8])),
        )


@dataclass
class StrategyPerformance:
    """Performance record of a strategy on a problem."""
    strategy_config: StrategyConfig
    problem_features: ProblemFeatures
    final_fitness: float
    convergence_speed: float  # Generations to reach 90% of final fitness
    success_rate: float  # Fraction of runs that found good solution
    n_evaluations: int = 0

    def to_meta_example(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns (problem_features, strategy_config, final_fitness)."""
        return (
            self.problem_features.to_vector(),
            self.strategy_config.to_vector(),
            self.final_fitness
        )


class MetaLearningModel:
    """
    Simple meta-learning model using k-NN regression.
    Predicts strategy performance based on problem similarity.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.examples: List[StrategyPerformance] = []

    def add_example(self, perf: StrategyPerformance):
        self.examples.append(perf)

    def predict_performance(self,
                            problem_features: ProblemFeatures,
                            strategy_config: StrategyConfig) -> float:
        """
        Predict performance of a strategy on a problem.
        Uses k-NN in problem space, weighted by strategy similarity.
        """
        if not self.examples:
            return 0.5  # Neutral prediction

        # Compute distances in problem feature space
        problem_vec = problem_features.to_vector()
        distances = []
        for ex in self.examples:
            ex_vec = ex.problem_features.to_vector()
            dist = np.linalg.norm(problem_vec - ex_vec)
            distances.append((dist, ex))

        # Get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]

        # Weighted average of performance, weighted by strategy similarity
        weights = []
        predictions = []
        strategy_vec = strategy_config.to_vector()

        for dist, ex in neighbors:
            if dist == 0:
                # Exact match: return exact performance
                return ex.final_fitness

            # Weight by inverse problem distance and strategy similarity
            ex_strategy_vec = ex.strategy_config.to_vector()
            strategy_sim = 1.0 / (1.0 + np.linalg.norm(strategy_vec - ex_strategy_vec))
            weight = strategy_sim / (1.0 + dist)

            weights.append(weight)
            predictions.append(ex.final_fitness)

        if not weights:
            return 0.5

        total_weight = sum(weights)
        return sum(w * p for w, p in zip(weights, predictions)) / total_weight

    def recommend_strategy(self,
                           problem_features: ProblemFeatures,
                           n_candidates: int = 10) -> StrategyConfig:
        """
        Recommend the best strategy configuration for a problem.
        Samples candidates and picks the one with highest predicted performance.
        """
        if not self.examples:
            # No data: return default strategy
            return StrategyConfig()

        candidates = []
        for _ in range(n_candidates):
            # Sample a random strategy around known good ones
            if self.examples:
                base = random.choice(self.examples).strategy_config
                vec = base.to_vector()
                # Add noise
                noise = np.random.normal(0, 0.1, len(vec))
                vec = vec + noise
                candidate = StrategyConfig.from_vector(vec)
            else:
                candidate = StrategyConfig()

            pred = self.predict_performance(problem_features, candidate)
            candidates.append((candidate, pred))

        # Pick best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_problem_classes(self) -> Dict[str, List[StrategyPerformance]]:
        """Group examples by problem modality."""
        classes = {}
        for ex in self.examples:
            key = ex.problem_features.modality
            if key not in classes:
                classes[key] = []
            classes[key].append(ex)
        return classes

    def get_best_strategy_per_class(self) -> Dict[str, StrategyConfig]:
        """Get the best strategy for each problem class."""
        classes = self.get_problem_classes()
        best = {}
        for modality, examples in classes.items():
            best_ex = max(examples, key=lambda x: x.final_fitness)
            best[modality] = best_ex.strategy_config
        return best

    def to_dict(self) -> Dict:
        return {
            "n_examples": len(self.examples),
            "problem_classes": list(self.get_problem_classes().keys()),
            "best_strategies": {
                k: {
                    "mutation_rate": v.mutation_rate,
                    "crossover_type": v.crossover_type,
                    "selection_type": v.selection_type,
                }
                for k, v in self.get_best_strategy_per_class().items()
            },
        }


class MetaLearningBreeder:
    """
    Breeding daemon that meta-learns optimal strategies.

    1. Extracts problem features from the fitness landscape
    2. Queries the meta-learning model for best strategy
    3. Adapts strategy during evolution based on progress
    4. Records performance for future meta-learning
    """

    def __init__(self,
                 population_size: int = 50,
                 meta_model: Optional[MetaLearningModel] = None,
                 adaptation_interval: int = 5):
        self.population_size = population_size
        self.meta_model = meta_model or MetaLearningModel()
        self.adaptation_interval = adaptation_interval

        self.current_strategy = StrategyConfig(population_size=population_size)
        self.generation = 0
        self.history: List[Tuple[float, StrategyConfig]] = []  # (avg_fitness, strategy)

    def extract_problem_features(self,
                                  population: List[Tuple[Dict[str, float], float]]) -> ProblemFeatures:
        """
        Extract problem features from current population state.
        """
        if not population:
            return ProblemFeatures()

        fitnesses = [f for _, f in population]
        genomes = [g for g, _ in population]

        # Dimensionality
        dim = len(genomes[0]) if genomes else 1

        # Noise level: coefficient of variation of fitness
        mean_fit = np.mean(fitnesses)
        std_fit = np.std(fitnesses)
        noise = std_fit / abs(mean_fit) if mean_fit != 0 else 0.0

        # Modality estimate: count local maxima in sample
        # Simple heuristic: high variance in local neighborhoods suggests multimodal
        sorted_fitness = sorted(fitnesses)
        local_diffs = [sorted_fitness[i+1] - sorted_fitness[i] for i in range(len(sorted_fitness)-1)]
        modality = "multimodal" if np.std(local_diffs) > np.mean(local_diffs) * 2 else "unimodal"

        # Separability: check if fitness correlates with individual genes
        separable = True
        if genomes and dim > 1:
            gene_corrs = []
            for gene_key in genomes[0]:
                gene_vals = [g.get(gene_key, 0.0) for g in genomes]
                if np.std(gene_vals) > 0 and np.std(fitnesses) > 0:
                    corr = np.corrcoef(gene_vals, fitnesses)[0, 1]
                    gene_corrs.append(abs(corr))
            # If genes are strongly correlated individually, landscape is separable
            separable = np.mean(gene_corrs) > 0.5 if gene_corrs else True

        return ProblemFeatures(
            dimensionality=dim,
            modality=modality,
            noise_level=noise,
            separable=separable,
        )

    def adapt_strategy(self, population: List[Tuple[Dict[str, float], float]]):
        """
        Adapt strategy based on meta-learning model and current progress.
        """
        if self.generation % self.adaptation_interval != 0:
            return

        features = self.extract_problem_features(population)
        self.current_strategy = self.meta_model.recommend_strategy(features)
        self.current_strategy.population_size = self.population_size

    def mutate(self, genome: Dict[str, float]) -> Dict[str, float]:
        """Mutate using current strategy."""
        mutated = genome.copy()
        rate = self.current_strategy.mutation_rate

        for key in mutated:
            if key.startswith("_"):
                continue
            if random.random() < rate:
                if self.current_strategy.mutation_type == "gaussian":
                    mutated[key] += random.gauss(0, abs(mutated[key]) * 0.1)
                elif self.current_strategy.mutation_type == "uniform":
                    mutated[key] += random.uniform(-abs(mutated[key]) * 0.2, abs(mutated[key]) * 0.2)
                elif self.current_strategy.mutation_type == "cauchy":
                    mutated[key] += np.random.standard_cauchy() * abs(mutated[key]) * 0.1
                else:  # adaptive
                    mutated[key] *= (1 + random.gauss(0, 0.05))

        return mutated

    def crossover(self, p1: Dict[str, float], p2: Dict[str, float]) -> Dict[str, float]:
        """Crossover using current strategy."""
        child = {}
        cross_type = self.current_strategy.crossover_type

        for key in p1:
            if key not in p2:
                child[key] = p1[key]
                continue

            if cross_type == "uniform":
                child[key] = p1[key] if random.random() < 0.5 else p2[key]
            elif cross_type == "one_point":
                # Simplified: first half from p1, second from p2
                child[key] = p1[key]  # One-point is tricky with dicts; use uniform for simplicity
            elif cross_type == "blend":
                alpha = random.random()
                child[key] = alpha * p1[key] + (1 - alpha) * p2[key]
            else:
                child[key] = p1[key] if random.random() < 0.5 else p2[key]

        return child

    def select_parents(self, population: List[Tuple[Dict, float]], k: int = 2) -> List[Tuple[Dict, float]]:
        """Select parents using current strategy."""
        if len(population) < 2:
            return population[:k]

        selected = []
        sel_type = self.current_strategy.selection_type

        for _ in range(k):
            if sel_type == "tournament":
                tournament_size = min(3, len(population))
                tournament = random.sample(population, tournament_size)
                selected.append(max(tournament, key=lambda x: x[1]))
            elif sel_type == "roulette":
                total = sum(max(0, f) for _, f in population)
                if total > 0:
                    pick = random.uniform(0, total)
                    current = 0
                    for g, f in population:
                        current += max(0, f)
                        if current >= pick:
                            selected.append((g, f))
                            break
                else:
                    selected.append(random.choice(population))
            else:  # rank
                sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
                idx = int(random.random() * len(sorted_pop))
                selected.append(sorted_pop[idx])

        return selected

    def breed_generation(self,
                         population: List[Tuple[Dict[str, float], float]],
                         task_fn: Callable[[Dict[str, float]], Any]) -> List[Tuple[Dict[str, float], float]]:
        """Run one generation with meta-learned strategy."""
        self.generation += 1
        self.adapt_strategy(population)

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)

        # Elitism
        n_elite = max(1, int(len(sorted_pop) * self.current_strategy.elitism_ratio))
        new_pop = sorted_pop[:n_elite]

        # Fill rest
        while len(new_pop) < self.population_size:
            parents = self.select_parents(sorted_pop, k=2)
            if len(parents) < 2:
                break
            child = self.crossover(parents[0][0], parents[1][0])
            child = self.mutate(child)
            result = task_fn(child)
            fitness = result.get("fitness", 0.0) if isinstance(result, dict) else float(result)
            new_pop.append((child, fitness))

        # Record performance
        avg_fitness = np.mean([f for _, f in new_pop])
        self.history.append((avg_fitness, self.current_strategy))

        return new_pop

    def record_final_performance(self, final_fitness: float,
                                    convergence_speed: float,
                                    success_rate: float):
        """Record final performance for meta-learning."""
        if not self.history:
            return

        # Extract problem features from last population
        # Use last strategy's features as proxy
        features = ProblemFeatures()

        perf = StrategyPerformance(
            strategy_config=self.current_strategy,
            problem_features=features,
            final_fitness=final_fitness,
            convergence_speed=convergence_speed,
            success_rate=success_rate,
        )
        self.meta_model.add_example(perf)

    def get_meta_summary(self) -> Dict:
        return {
            "generation": self.generation,
            "current_strategy": {
                "mutation_rate": self.current_strategy.mutation_rate,
                "crossover_type": self.current_strategy.crossover_type,
                "selection_type": self.current_strategy.selection_type,
            },
            "meta_model_examples": len(self.meta_model.examples),
            "history_length": len(self.history),
            "best_strategy_by_class": {
                k: v.mutation_rate
                for k, v in self.meta_model.get_best_strategy_per_class().items()
            },
        }
