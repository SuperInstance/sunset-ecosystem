from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BreederWeight:
    """Weight for a breeder in the ensemble."""

    breeder_name: str
    weight: float
    performance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breeder_name": self.breeder_name,
            "weight": self.weight,
            "performance": self.performance,
        }


class EnsembleBreeder:
    """
    Ensemble breeder combining multiple breeding strategies.

    Uses weighted voting to combine outputs from multiple breeders.
    Weights are updated based on performance.
    """

    def __init__(self, population_size: int = 50, dimensions: int = 10):
        self.population_size = population_size
        self.dimensions = dimensions
        self.breeders: Dict[str, Any] = {}
        self.weights: Dict[str, BreederWeight] = {}
        self.history: List[Dict[str, Any]] = []
        self.generation = 0

    def add_breeder(self, name: str, breeder: Any, initial_weight: float = 1.0):
        """Add a breeder to the ensemble."""
        self.breeders[name] = breeder
        self.weights[name] = BreederWeight(
            breeder_name=name,
            weight=initial_weight,
            performance=0.0,
        )

    def remove_breeder(self, name: str) -> bool:
        """Remove a breeder from the ensemble."""
        if name not in self.breeders:
            return False
        del self.breeders[name]
        del self.weights[name]
        return True

    def initialize(self):
        """Initialize all breeders in the ensemble."""
        for breeder in self.breeders.values():
            if hasattr(breeder, "initialize"):
                breeder.initialize()

    def evaluate(self, fitness_fn: callable) -> Dict[str, float]:
        """Evaluate all breeders and update performance."""
        performances = {}
        for name, breeder in self.breeders.items():
            if hasattr(breeder, "evaluate"):
                perf = breeder.evaluate(fitness_fn)
                performances[name] = perf
                self.weights[name].performance = perf
            elif hasattr(breeder, "get_best") and breeder.get_best():
                perf = breeder.get_best().fitness
                performances[name] = perf
                self.weights[name].performance = perf
        return performances

    def evolve(self, fitness_fn: callable) -> Tuple[Any, float]:
        """
        Evolve all breeders and return the best combined result.
        Returns (best_genome, best_fitness).
        """
        self.generation += 1

        # Evolve each breeder
        for name, breeder in self.breeders.items():
            if hasattr(breeder, "evolve"):
                breeder.evolve(fitness_fn)

        # Evaluate and update weights
        performances = self.evaluate(fitness_fn)

        # Find best overall
        best_name = max(performances, key=performances.get)
        best_breeder = self.breeders[best_name]
        best_genome = (
            best_breeder.get_best().genome
            if hasattr(best_breeder, "get_best")
            else None
        )
        best_fitness = performances[best_name]

        # Update weights (softmax over performance)
        if performances:
            exp_perfs = {
                k: np.exp(v - max(performances.values()))
                for k, v in performances.items()
            }
            total = sum(exp_perfs.values())
            for name, exp_val in exp_perfs.items():
                self.weights[name].weight = (
                    exp_val / total if total > 0 else 1.0 / len(self.weights)
                )

        self.history.append(
            {
                "generation": self.generation,
                "best_breeder": best_name,
                "best_fitness": best_fitness,
                "weights": {k: w.weight for k, w in self.weights.items()},
            }
        )

        return best_genome, best_fitness

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "breeders": len(self.breeders),
            "weights": {k: w.to_dict() for k, w in self.weights.items()},
            "best_history": [h["best_fitness"] for h in self.history[-10:]],
        }

    def export_json(self) -> str:
        """Export ensemble state as JSON."""
        return json.dumps(
            {
                "breeders": list(self.breeders.keys()),
                "weights": {k: w.to_dict() for k, w in self.weights.items()},
                "history": self.history,
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_ensemble_stats(),
        }
