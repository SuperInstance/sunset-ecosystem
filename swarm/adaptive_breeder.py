from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class BreederStrategy:
    """A breeding strategy with performance metrics."""

    name: str
    breeder: Any
    success_rate: float = 0.0
    avg_improvement: float = 0.0
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "success_rate": self.success_rate,
            "avg_improvement": self.avg_improvement,
            "usage_count": self.usage_count,
        }


class AdaptiveBreeder:
    """
    Adaptive breeder that switches strategies based on landscape.

    Tracks performance of each strategy and selects the best one
    for the current problem landscape.
    """

    def __init__(self, population_size: int = 50, dimensions: int = 10):
        self.population_size = population_size
        self.dimensions = dimensions
        self._strategies: Dict[str, BreederStrategy] = {}
        self._current_strategy: Optional[str] = None
        self._history: List[Dict[str, Any]] = []
        self.generation = 0

    def add_strategy(self, name: str, breeder: Any, initial_weight: float = 1.0):
        """Add a breeding strategy."""
        self._strategies[name] = BreederStrategy(
            name=name,
            breeder=breeder,
        )
        if self._current_strategy is None:
            self._current_strategy = name

    def remove_strategy(self, name: str) -> bool:
        """Remove a breeding strategy."""
        if name not in self._strategies:
            return False
        del self._strategies[name]
        if self._current_strategy == name:
            self._current_strategy = next(iter(self._strategies), None)
        return True

    def select_strategy(
        self, landscape_features: Optional[Dict[str, float]] = None
    ) -> str:
        """Select the best strategy for the current landscape."""
        if not self._strategies:
            raise ValueError("No strategies available")

        # If only one strategy, use it
        if len(self._strategies) == 1:
            return next(iter(self._strategies))

        # Select based on success rate
        best_name = max(
            self._strategies.keys(),
            key=lambda n: self._strategies[n].success_rate,
        )
        self._current_strategy = best_name
        return best_name

    def evolve(self, fitness_fn: Callable) -> tuple:
        """
        Evolve using the current strategy.
        Returns (best_genome, best_fitness).
        """
        self.generation += 1
        strategy_name = self.select_strategy()
        strategy = self._strategies[strategy_name]

        # Run the breeder
        breeder = strategy.breeder
        if hasattr(breeder, "evolve"):
            breeder.evolve(fitness_fn)

        # Get best result
        best = breeder.get_best() if hasattr(breeder, "get_best") else None
        if best:
            best_genome = best.genome
            best_fitness = best.fitness
        else:
            best_genome = None
            best_fitness = 0.0

        # Update strategy metrics
        strategy.usage_count += 1
        # Track improvement (mock: assume improvement if fitness > 0)
        if best_fitness > 0:
            strategy.success_rate = (
                strategy.success_rate * (strategy.usage_count - 1) + 1.0
            ) / strategy.usage_count
        else:
            strategy.success_rate = (
                strategy.success_rate * (strategy.usage_count - 1) + 0.0
            ) / strategy.usage_count

        self._history.append(
            {
                "generation": self.generation,
                "strategy": strategy_name,
                "best_fitness": best_fitness,
            }
        )

        return best_genome, best_fitness

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics for all strategies."""
        return {
            "strategies": {name: s.to_dict() for name, s in self._strategies.items()},
            "current": self._current_strategy,
            "history": self._history[-10:],
        }

    def export_json(self) -> str:
        """Export adaptive breeder state as JSON."""
        return json.dumps(
            {
                "strategies": {k: v.to_dict() for k, v in self._strategies.items()},
                "current": self._current_strategy,
                "history": self._history,
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_strategy_stats(),
        }
