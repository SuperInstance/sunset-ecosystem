"""fleet/mercury_cellular.py — Mercury-style declarative cellular automata rules.

Expresses cellular automata rules as Mercury predicates (declarative logic)
and evaluates them in Python.  Provides a bridge to our Numba cellular engine.

Usage
-----
    from fleet.mercury_cellular import MercuryCellularEngine, rule_survival

    engine = MercuryCellularEngine(grid_size=(64, 64))
    engine.seed([(32, 32)], energy=1.0)
    engine.register_rule(rule_survival)
    stats = engine.tick()

Rules are written as Mercury predicates with explicit modes:

    :- pred survival(float::in, float::in, list(float)::in,
                      float::out, float::out) is det.

    survival(Energy, State, Neighbors, OutEnergy, OutState) :-
        Energy > 0.5,
        OutEnergy = Energy * 0.9,
        OutState = State.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Rule types ─────────────────────────────────────────────────────────

MercuryRule = Callable[
    [float, float, List[float]], Tuple[Optional[float], Optional[float]]
]


# ── Built-in rules ─────────────────────────────────────────────────────


def rule_survival(
    energy: float, state: float, neighbors: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Mercury-style survival rule: cells with energy > 0.5 survive, else die."""
    if energy > 0.5:
        return (energy * 0.9, state)
    return (0.0, 0.0)


def rule_reproduction(
    energy: float, state: float, neighbors: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Mercury-style reproduction rule: empty cell with 2+ energetic neighbors gets a child."""
    if energy == 0.0:
        active_neighbors = [n for n in neighbors if n > 0.5]
        if len(active_neighbors) >= 2:
            avg_energy = sum(active_neighbors) / len(active_neighbors)
            return (avg_energy * 0.8, state)
    return (None, None)  # No change


def rule_diffusion(
    energy: float, state: float, neighbors: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Mercury-style diffusion rule: energy spreads to neighbors."""
    if energy > 0.0:
        return (energy * 0.9, state)
    return (None, None)


def rule_mutation(
    energy: float, state: float, neighbors: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Mercury-style mutation rule: random perturbation."""
    if energy > 0.0:
        noise = np.random.normal(0, 0.05)
        return (max(0.0, energy + noise), state)
    return (None, None)


# ── Engine ──────────────────────────────────────────────────────────────


@dataclass
class MercuryCellularEngine:
    """Cellular automata engine with Mercury-style declarative rules."""

    grid_size: Tuple[int, int] = (64, 64)
    rules: List[MercuryRule] = field(default_factory=list)
    tick_count: int = 0
    total_energy_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._energies = np.zeros(self.grid_size, dtype=np.float32)
        self._states = np.zeros(self.grid_size, dtype=np.float32)

    @property
    def energies(self) -> np.ndarray:
        return self._energies

    @property
    def states(self) -> np.ndarray:
        return self._states

    def seed(
        self, positions: List[Tuple[int, int]], energy: float = 1.0, state: float = 1.0
    ) -> None:
        """Seed cells at specific positions."""
        for i, j in positions:
            if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                self._energies[i, j] = energy
                self._states[i, j] = state

    def seed_random(
        self, count: int, energy_range: Tuple[float, float] = (0.5, 1.0)
    ) -> None:
        """Randomly seed `count` cells."""
        rows, cols = self.grid_size
        indices = np.random.choice(rows * cols, size=count, replace=False)
        self._energies.flat[indices] = np.random.uniform(*energy_range, size=count)
        self._states.flat[indices] = 1.0

    def register_rule(self, rule: MercuryRule) -> None:
        """Add a Mercury-style rule to the engine."""
        self.rules.append(rule)

    def _get_neighbors(self, i: int, j: int) -> List[float]:
        """Get Moore neighborhood energies."""
        rows, cols = self.grid_size
        neighbors = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append(float(self._energies[ni, nj]))
        return neighbors

    def tick(self) -> Dict[str, Any]:
        """Evaluate one generation with Mercury-style rules."""
        new_energies = np.zeros(self.grid_size, dtype=np.float32)
        new_states = np.zeros(self.grid_size, dtype=np.float32)

        rows, cols = self.grid_size
        for i in range(rows):
            for j in range(cols):
                energy = float(self._energies[i, j])
                state = float(self._states[i, j])
                neighbors = self._get_neighbors(i, j)

                new_e, new_s = energy, state
                for rule in self.rules:
                    result = rule(new_e, new_s, neighbors)
                    if result[0] is not None:
                        new_e = result[0]
                    if result[1] is not None:
                        new_s = result[1]

                new_energies[i, j] = new_e
                new_states[i, j] = new_s

        self._energies = new_energies
        self._states = new_states
        self.tick_count += 1

        total_energy = float(np.sum(self._energies))
        active_cells = int(np.count_nonzero(self._energies > 0.0))
        self.total_energy_history.append(total_energy)

        return {
            "tick": self.tick_count,
            "total_energy": total_energy,
            "active_cells": active_cells,
            "grid_size": self.grid_size,
        }

    def run(self, ticks: int) -> List[Dict[str, Any]]:
        """Run multiple ticks, return history."""
        history = []
        for _ in range(ticks):
            history.append(self.tick())
        return history

    def benchmark(self, ticks: int = 100) -> Dict[str, float]:
        """Benchmark tick performance."""
        import time

        for _ in range(min(10, ticks)):
            self.tick()

        t0 = time.perf_counter()
        for _ in range(ticks):
            self.tick()
        dt = time.perf_counter() - t0

        return {
            "ticks": ticks,
            "total_seconds": dt,
            "ms_per_tick": (dt / ticks) * 1000.0 if ticks > 0 else 0.0,
            "ticks_per_second": ticks / dt if dt > 0 else float("inf"),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grid state."""
        return {
            "grid_size": self.grid_size,
            "tick_count": self.tick_count,
            "energies": self._energies.tolist(),
            "states": self._states.tolist(),
            "total_energy_history": self.total_energy_history,
        }

    # ── Bridge to Numba ───────────────────────────────────────────

    def to_numba_engine(self):
        """Convert to NumbaCellularEngine for JIT compilation."""
        from swarm.cellular_numba import NumbaCellularEngine

        engine = NumbaCellularEngine(grid_size=self.grid_size)
        # Copy state
        engine._energies[:] = self._energies[:]
        engine._states[:] = self._states[:]
        return engine
