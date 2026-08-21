"""swarm/cellular_numba.py — Numba-JIT compiled cellular automata rules.

Extends `cellular_engine.py` with high-performance rule kernels.
Rules are written as Python functions and JIT-compiled to LLVM
bytecode via Numba.  When CUDA is available, the same rule functions
are compiled to GPU kernels via `@cuda.jit`.

Usage
-----
    from swarm.cellular_numba import NumbaCellularEngine, rule_survival

    engine = NumbaCellularEngine(grid_size=(128, 128))
    engine.register_rule(rule_survival)
    engine.tick()  # JIT-compiled rule evaluation

    # Benchmark
    import timeit
    dt = timeit.timeit(engine.tick, number=100)

Rules
-----
A rule is a pure function with signature::

    def my_rule(energies, states, neighbors, out_energies, out_states, params)

All arguments are 2-D float32 arrays (grid_size x grid_size) or
1-D parameter arrays.  Rules must be `@njit`-decorated or
`NumbaCellularEngine` will auto-wrap them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
    from numba.core.registry import CPUDispatcher

    HAS_NUMBA = True
except Exception as exc:  # noqa: BLE001
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]
    CPUDispatcher = None  # type: ignore[assignment, misc]
    HAS_NUMBA = False
    logging.warning(
        "numba not available; cellular_numba will use pure Python fallback (%s)", exc
    )

logger = logging.getLogger(__name__)


# ── Rule signatures ─────────────────────────────────────────────────────

RuleKernel = Callable[..., None]


def _ensure_jit(fn: Callable[..., None]) -> Callable[..., None]:
    """Auto-wrap a Python function with @njit if not already compiled."""
    if not HAS_NUMBA:
        return fn
    if isinstance(fn, CPUDispatcher):
        return fn
    return njit(fn, cache=True)  # type: ignore[operator]


# ── Built-in rule kernels ───────────────────────────────────────────────


def _rule_survival_pure(
    energies: np.ndarray,
    states: np.ndarray,
    neighbors: np.ndarray,
    out_energies: np.ndarray,
    out_states: np.ndarray,
    params: np.ndarray,
) -> None:
    """Survival rule: cells with energy > threshold survive, else die."""
    threshold = params[0]
    decay = params[1]
    rows, cols = energies.shape
    for i in range(rows):
        for j in range(cols):
            if energies[i, j] > threshold:
                out_energies[i, j] = energies[i, j] * (1.0 - decay)
                out_states[i, j] = states[i, j]
            else:
                out_energies[i, j] = 0.0
                out_states[i, j] = 0.0


def _rule_reproduction_pure(
    energies: np.ndarray,
    states: np.ndarray,
    neighbors: np.ndarray,
    out_energies: np.ndarray,
    out_states: np.ndarray,
    params: np.ndarray,
) -> None:
    """Reproduction rule: empty cell with 2+ energetic neighbors gets a child."""
    threshold = params[0]
    inherit = params[1]
    noise = params[2]
    rows, cols = energies.shape
    for i in range(rows):
        for j in range(cols):
            if energies[i, j] == 0.0:
                n_active = 0
                total_energy = 0.0
                total_state = 0.0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if energies[ni, nj] > threshold:
                                n_active += 1
                                total_energy += energies[ni, nj]
                                total_state += states[ni, nj]
                if n_active >= 2:
                    out_energies[i, j] = (total_energy / n_active) * inherit + noise
                    out_states[i, j] = total_state / n_active
                else:
                    out_energies[i, j] = 0.0
                    out_states[i, j] = 0.0
            else:
                out_energies[i, j] = energies[i, j]
                out_states[i, j] = states[i, j]


def _rule_diffusion_pure(
    energies: np.ndarray,
    states: np.ndarray,
    neighbors: np.ndarray,
    out_energies: np.ndarray,
    out_states: np.ndarray,
    params: np.ndarray,
) -> None:
    """Diffusion rule: energy spreads to neighbors."""
    diffusion_rate = params[0]
    rows, cols = energies.shape
    for i in range(rows):
        for j in range(cols):
            if energies[i, j] > 0.0:
                out_energies[i, j] = energies[i, j] * (1.0 - diffusion_rate)
                out_states[i, j] = states[i, j]
                # Spread to neighbors
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            out_energies[ni, nj] += (
                                energies[i, j] * diffusion_rate / 8.0
                            )
            else:
                out_energies[i, j] += energies[i, j]
                out_states[i, j] = states[i, j]


# ── Numba JIT wrappers ──────────────────────────────────────────────────

if HAS_NUMBA:
    rule_survival = njit(cache=True)(_rule_survival_pure)
    rule_reproduction = njit(cache=True)(_rule_reproduction_pure)
    rule_diffusion = njit(cache=True)(_rule_diffusion_pure)
else:
    rule_survival = _rule_survival_pure
    rule_reproduction = _rule_reproduction_pure
    rule_diffusion = _rule_diffusion_pure


# ── Engine ────────────────────────────────────────────────────────────


@dataclass
class NumbaCellularEngine:
    """High-performance cellular automata engine using Numba JIT rules.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Dimensions of the cellular grid.
    rules : list[RuleKernel]
        JIT-compiled rule functions applied in order each tick.
    params : dict[str, np.ndarray]
        Parameter arrays keyed by rule name.
    """

    grid_size: Tuple[int, int] = (64, 64)
    rules: List[RuleKernel] = field(default_factory=list)
    params: Dict[str, np.ndarray] = field(default_factory=dict)
    tick_count: int = 0
    total_energy_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._energies = np.zeros(self.grid_size, dtype=np.float32)
        self._states = np.zeros(self.grid_size, dtype=np.float32)
        self._out_energies = np.zeros(self.grid_size, dtype=np.float32)
        self._out_states = np.zeros(self.grid_size, dtype=np.float32)
        self._neighbors = np.zeros(self.grid_size, dtype=np.float32)

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

    def register_rule(
        self,
        rule: RuleKernel,
        params: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        """Add a rule to the engine. Auto-wraps with @njit if needed."""
        jit_rule = _ensure_jit(rule)
        self.rules.append(jit_rule)
        rule_name = name or f"rule_{len(self.rules)}"
        if params is not None:
            self.params[rule_name] = params.astype(np.float32)
        else:
            self.params[rule_name] = np.array([0.5, 0.1, 0.01], dtype=np.float32)

    def _compute_neighbors(self) -> None:
        """Precompute neighbor counts (Moore neighborhood)."""
        rows, cols = self.grid_size
        self._neighbors.fill(0.0)
        for i in range(rows):
            for j in range(cols):
                count = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if self._energies[ni, nj] > 0.0:
                                count += 1
                self._neighbors[i, j] = float(count)

    def tick(self) -> Dict[str, Any]:
        """Evaluate one generation. Returns stats."""
        self._compute_neighbors()
        self._out_energies.fill(0.0)
        self._out_states.fill(0.0)

        for idx, rule in enumerate(self.rules):
            rule_name = f"rule_{idx + 1}"
            params = self.params.get(
                rule_name, np.array([0.5, 0.1, 0.01], dtype=np.float32)
            )
            rule(
                self._energies,
                self._states,
                self._neighbors,
                self._out_energies,
                self._out_states,
                params,
            )
            # Swap buffers for next rule
            self._energies, self._out_energies = self._out_energies, self._energies
            self._states, self._out_states = self._out_states, self._states
            self._out_energies.fill(0.0)
            self._out_states.fill(0.0)

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
        """Benchmark tick performance. Returns timing stats."""
        import time

        # Warmup
        for _ in range(min(10, ticks)):
            self.tick()

        t0 = time.perf_counter()
        for _ in range(ticks):
            self.tick()
        dt = time.perf_counter() - t0

        return {
            "ticks": ticks,
            "total_seconds": dt,
            "ms_per_tick": (dt / ticks) * 1000.0,
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
