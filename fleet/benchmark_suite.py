from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    problem: str
    breeder: str
    best_fitness: float
    generations: int
    time_seconds: float
    final_diversity: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "breeder": self.breeder,
            "best_fitness": self.best_fitness,
            "generations": self.generations,
            "time_seconds": self.time_seconds,
            "final_diversity": self.final_diversity,
            "success": self.success,
            "metadata": self.metadata,
        }


class BenchmarkSuite:
    """
    Standardized benchmark suite for fleet breeders.

    Tests breeders on standard optimization problems:
    - Sphere (Rosenbrock, Rastrigin, Ackley)
    - Combinatorial (OneMax, LeadingOnes)
    - Diversity (Coverage, Novelty)
    """

    def __init__(self, max_generations: int = 100):
        self.max_generations = max_generations
        self.results: List[BenchmarkResult] = []

    def _sphere(self, genome: np.ndarray) -> float:
        """Sphere function - simple unimodal."""
        return -np.sum(genome**2)

    def _rosenbrock(self, genome: np.ndarray) -> float:
        """Rosenbrock function - deceptive valley."""
        return -np.sum(
            100 * (genome[1:] - genome[:-1] ** 2) ** 2 + (1 - genome[:-1]) ** 2
        )

    def _rastrigin(self, genome: np.ndarray) -> float:
        """Rastrigin function - highly multimodal."""
        return -np.sum(genome**2 - 10 * np.cos(2 * np.pi * genome) + 10)

    def _ackley(self, genome: np.ndarray) -> float:
        """Ackley function - many local minima."""
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(genome)
        sum1 = np.sum(genome**2)
        sum2 = np.sum(np.cos(c * genome))
        return -(a + np.exp(1) - a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d))

    def _onemax(self, genome: np.ndarray) -> float:
        """OneMax - maximize number of 1s in binary vector."""
        return float(np.sum(genome > 0.5))

    def _leading_ones(self, genome: np.ndarray) -> float:
        """LeadingOnes - count consecutive 1s from start."""
        binary = genome > 0.5
        count = 0
        for bit in binary:
            if bit:
                count += 1
            else:
                break
        return float(count)

    def get_problems(self) -> Dict[str, Callable]:
        """Get all benchmark problems."""
        return {
            "sphere": self._sphere,
            "rosenbrock": self._rosenbrock,
            "rastrigin": self._rastrigin,
            "ackley": self._ackley,
            "onemax": self._onemax,
            "leading_ones": self._leading_ones,
        }

    def run_breeder(
        self,
        breeder_name: str,
        breeder_instance: Any,
        problem_name: str,
        dimensions: int = 10,
    ) -> BenchmarkResult:
        """
        Run a breeder on a specific problem.

        Args:
            breeder_name: Name identifier for the breeder
            breeder_instance: The breeder object (must have initialize, evolve, get_best)
            problem_name: Name of the problem to solve
            dimensions: Number of dimensions
        """
        problem = self.get_problems().get(problem_name)
        if not problem:
            raise ValueError(f"Unknown problem: {problem_name}")

        start_time = time.time()
        best_fitness = float("-inf")
        success = False
        final_diversity = 0.0

        try:
            # Initialize breeder
            if hasattr(breeder_instance, "initialize"):
                breeder_instance.initialize()

            # Evaluate initial population
            if hasattr(breeder_instance, "evaluate"):
                breeder_instance.evaluate(problem)

            # Evolve
            for gen in range(self.max_generations):
                if hasattr(breeder_instance, "evolve"):
                    breeder_instance.evolve(problem)
                else:
                    break

                # Check for improvement
                if hasattr(breeder_instance, "get_best"):
                    best = breeder_instance.get_best()
                    if best and hasattr(best, "fitness"):
                        best_fitness = best.fitness

            # Get diversity
            if hasattr(breeder_instance, "get_diversity"):
                final_diversity = breeder_instance.get_diversity()

            success = True

        except Exception as e:
            success = False
            best_fitness = float("-inf")

        elapsed = time.time() - start_time

        result = BenchmarkResult(
            problem=problem_name,
            breeder=breeder_name,
            best_fitness=best_fitness,
            generations=self.max_generations,
            time_seconds=elapsed,
            final_diversity=final_diversity,
            success=success,
        )
        self.results.append(result)
        return result

    def run_all(
        self, breeder_name: str, breeder_instance: Any
    ) -> List[BenchmarkResult]:
        """Run all benchmark problems on a breeder."""
        results = []
        for problem in self.get_problems().keys():
            result = self.run_breeder(breeder_name, breeder_instance, problem)
            results.append(result)
        return results

    def compare_breeders(
        self, breeder_results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, Any]:
        """
        Compare multiple breeders across all problems.

        Returns summary statistics per problem.
        """
        comparison = {}
        for problem in self.get_problems().keys():
            problem_results = []
            for breeder_name, results in breeder_results.items():
                for r in results:
                    if r.problem == problem:
                        problem_results.append(
                            {
                                "breeder": breeder_name,
                                "fitness": r.best_fitness,
                                "time": r.time_seconds,
                                "success": r.success,
                            }
                        )
            if problem_results:
                best = max(problem_results, key=lambda x: x["fitness"])
                comparison[problem] = {
                    "best_breeder": best["breeder"],
                    "best_fitness": best["fitness"],
                    "all": problem_results,
                }
        return comparison

    def get_leaderboard(self, problem: Optional[str] = None) -> List[BenchmarkResult]:
        """Get sorted leaderboard of results."""
        filtered = self.results
        if problem:
            filtered = [r for r in filtered if r.problem == problem]
        return sorted(filtered, key=lambda r: r.best_fitness, reverse=True)

    def export_json(self) -> str:
        """Export all results as JSON."""
        return json.dumps(
            {
                "max_generations": self.max_generations,
                "results": [r.to_dict() for r in self.results],
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problems": list(self.get_problems().keys()),
            "results": len(self.results),
            "leaderboard": [r.to_dict() for r in self.get_leaderboard()[:5]],
        }
