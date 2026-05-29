# Cellular Engine
# GPU-ready cellular automata + LLM hybrid for agent behavior simulation

"""Cellular automata + LLM hybrid engine for massively parallel agent simulation.

Simulates agent populations as a cellular automata grid where cell states are
influenced by LLM-driven decision kernels. Enables 10k+ agent simulation on GPU
with LLM reasoning injected at boundaries and high-value clusters.

References:
- Conway's Game of Life: Gardner, M. (1970). Scientific American.
- Continuous CA: Wolfram, S. (2002). A New Kind of Science.
- Neural CA: Mordvintsev et al. (2020). Distill.
"""

from __future__ import annotations

import time
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from enum import Enum, auto

import numpy as np

# Optional GPU backends
_GPU = None
_GPU_ERROR = None
try:
    import torch
    _GPU = "torch"
except ImportError:
    try:
        import cupy as cp
        _GPU = "cupy"
    except ImportError:
        _GPU_ERROR = "No GPU backend available (torch/cupy)"


# ---------------------------------------------------------------------------
# Cell state
# ---------------------------------------------------------------------------

@dataclass
class CellState:
    """Structured state for a single cell."""
    energy: float = 0.0
    signal: float = 0.0
    identity_hash: int = 0
    last_llm_query: float = 0.0
    neighbor_influence: float = 0.0
    generation: int = 0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.energy,
            self.signal,
            float(self.identity_hash % 1000) / 1000.0,
            self.neighbor_influence,
            float(self.generation),
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "CellState":
        return cls(
            energy=float(v[0]),
            signal=float(v[1]),
            identity_hash=int(v[2] * 1000),
            neighbor_influence=float(v[3]),
            generation=int(v[4]),
        )


# ---------------------------------------------------------------------------
# Cellular grid
# ---------------------------------------------------------------------------

class CellularGrid:
    """n-dimensional grid of cells, numpy-backed, optional GPU."""

    def __init__(
        self,
        shape: Tuple[int, ...] = (64, 64),
        dtype: np.dtype = np.float32,
        device: str = "cpu",
    ) -> None:
        self._shape = shape
        self._ndim = len(shape)
        self._dtype = dtype
        self._device = device
        self._cells = np.zeros((*shape, 5), dtype=dtype)  # [energy, signal, identity, influence, generation]
        self._gpu = None
        if device != "cpu" and _GPU is not None:
            if _GPU == "torch":
                self._gpu = torch
                self._cells_torch = self._gpu.tensor(self._cells, device=device)
            elif _GPU == "cupy":
                self._gpu = cp
                self._cells = self._gpu.array(self._cells)
        self._llm_cache: Dict[Tuple[int, ...], CellState] = {}
        self._energy_total: float = 0.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return self._ndim

    def get(self, idx: Tuple[int, ...]) -> CellState:
        if len(idx) != self._ndim:
            raise ValueError(f"Index dimension mismatch: {len(idx)} != {self._ndim}")
        raw = self._cells[idx]
        return CellState(
            energy=float(raw[0]),
            signal=float(raw[1]),
            identity_hash=int(raw[2]),
            neighbor_influence=float(raw[3]),
            generation=int(raw[4]),
        )

    def set(self, idx: Tuple[int, ...], state: CellState) -> None:
        if len(idx) != self._ndim:
            raise ValueError(f"Index dimension mismatch: {len(idx)} != {self._ndim}")
        self._cells[idx] = [
            state.energy, state.signal,
            float(state.identity_hash), state.neighbor_influence,
            float(state.generation),
        ]

    def randomize(self, seed: Optional[int] = None, energy_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        if seed is not None:
            np.random.seed(seed)
        self._cells[..., 0] = np.random.uniform(*energy_range, size=self._shape)
        self._cells[..., 1] = np.random.uniform(0.0, 0.5, size=self._shape)
        self._cells[..., 2] = np.random.randint(0, 10000, size=self._shape).astype(self._dtype)
        self._cells[..., 3] = np.zeros(self._shape, dtype=self._dtype)
        self._cells[..., 4] = np.zeros(self._shape, dtype=self._dtype)
        self._energy_total = float(np.sum(self._cells[..., 0]))

    def energy(self) -> float:
        return float(np.sum(self._cells[..., 0]))

    def signal_density(self) -> float:
        return float(np.mean(self._cells[..., 1]))

    def high_value_cells(self, energy_threshold: float = 0.7, signal_threshold: float = 0.5) -> List[Tuple[int, ...]]:
        """Return indices of cells exceeding both thresholds."""
        mask = (self._cells[..., 0] > energy_threshold) & (self._cells[..., 1] > signal_threshold)
        indices = np.argwhere(mask)
        return [tuple(int(i) for i in idx) for idx in indices]

    def to_numpy(self) -> np.ndarray:
        if self._gpu is not None and _GPU == "cupy":
            return self._cells.get()
        return self._cells.copy()


# ---------------------------------------------------------------------------
# CA generation kernel
# ---------------------------------------------------------------------------

class CAGenerationKernel:
    """Pure CA rules: diffusion, energy decay, signal propagation, reproduction."""

    def __init__(
        self,
        diffusion_rate: float = 0.1,
        energy_decay: float = 0.01,
        signal_decay: float = 0.05,
        reproduction_threshold: float = 1.5,
        max_energy: float = 2.0,
    ) -> None:
        self._diffusion = diffusion_rate
        self._energy_decay = energy_decay
        self._signal_decay = signal_decay
        self._reproduction_threshold = reproduction_threshold
        self._max_energy = max_energy

    def step(self, grid: CellularGrid) -> CellularGrid:
        """Apply one CA generation step."""
        cells = grid._cells
        new_cells = cells.copy()

        # Diffusion (energy spreads to neighbors)
        if grid.ndim == 2:
            self._diffuse_2d(cells, new_cells)
        elif grid.ndim == 3:
            self._diffuse_3d(cells, new_cells)

        # Energy decay
        new_cells[..., 0] -= self._energy_decay
        new_cells[..., 0] = np.clip(new_cells[..., 0], 0.0, self._max_energy)

        # Signal decay
        new_cells[..., 1] -= self._signal_decay
        new_cells[..., 1] = np.clip(new_cells[..., 1], 0.0, 1.0)

        # Neighbor influence recalculation
        if grid.ndim == 2:
            self._calc_influence_2d(new_cells)
        elif grid.ndim == 3:
            self._calc_influence_3d(new_cells)

        # Reproduction: high-energy + high-signal cells spawn new generation
        self._reproduce(new_cells)

        grid._cells = new_cells
        return grid

    def _diffuse_2d(self, cells: np.ndarray, new_cells: np.ndarray) -> None:
        # 4-neighbor diffusion
        up = np.roll(cells[..., 0], 1, axis=0)
        down = np.roll(cells[..., 0], -1, axis=0)
        left = np.roll(cells[..., 0], 1, axis=1)
        right = np.roll(cells[..., 0], -1, axis=1)
        avg_neighbor = (up + down + left + right) / 4.0
        new_cells[..., 0] += self._diffusion * (avg_neighbor - cells[..., 0])
        # Signal propagation (similar)
        sig_up = np.roll(cells[..., 1], 1, axis=0)
        sig_down = np.roll(cells[..., 1], -1, axis=0)
        sig_left = np.roll(cells[..., 1], 1, axis=1)
        sig_right = np.roll(cells[..., 1], -1, axis=1)
        avg_sig = (sig_up + sig_down + sig_left + sig_right) / 4.0
        new_cells[..., 1] += self._diffusion * (avg_sig - cells[..., 1])

    def _diffuse_3d(self, cells: np.ndarray, new_cells: np.ndarray) -> None:
        # 6-neighbor diffusion for 3D
        for axis in range(3):
            fwd = np.roll(cells[..., 0], 1, axis=axis)
            bwd = np.roll(cells[..., 0], -1, axis=axis)
            new_cells[..., 0] += self._diffusion * (fwd + bwd - 2 * cells[..., 0]) / 6.0

    def _calc_influence_2d(self, cells: np.ndarray) -> None:
        up = np.roll(cells[..., 0], 1, axis=0)
        down = np.roll(cells[..., 0], -1, axis=0)
        left = np.roll(cells[..., 0], 1, axis=1)
        right = np.roll(cells[..., 0], -1, axis=1)
        cells[..., 3] = (up + down + left + right) / 4.0

    def _calc_influence_3d(self, cells: np.ndarray) -> None:
        for axis in range(3):
            fwd = np.roll(cells[..., 0], 1, axis=axis)
            bwd = np.roll(cells[..., 0], -1, axis=axis)
            cells[..., 3] += (fwd + bwd) / 6.0

    def _reproduce(self, cells: np.ndarray) -> None:
        mask = (cells[..., 0] > self._reproduction_threshold) & (cells[..., 1] > 0.5)
        cells[..., 4] += mask.astype(cells.dtype)
        # Energy cost for reproduction
        cells[..., 0] -= mask.astype(cells.dtype) * 0.3
        cells[..., 0] = np.clip(cells[..., 0], 0.0, self._max_energy)


# ---------------------------------------------------------------------------
# LLM injection kernel
# ---------------------------------------------------------------------------

class LLMInjectionKernel:
    """Identifies high-value cell clusters and queries LLM for state transitions."""

    def __init__(
        self,
        energy_threshold: float = 0.7,
        signal_threshold: float = 0.5,
        injection_interval: int = 10,
        query_cache: Optional[Dict] = None,
    ) -> None:
        self._energy_threshold = energy_threshold
        self._signal_threshold = signal_threshold
        self._injection_interval = injection_interval
        self._query_cache = query_cache or {}
        self._step_counter = 0

    def step(
        self,
        grid: CellularGrid,
        llm_query_fn: Optional[Callable[[List[CellState]], List[CellState]]] = None,
    ) -> CellularGrid:
        self._step_counter += 1
        if self._step_counter % self._injection_interval != 0:
            return grid
        if llm_query_fn is None:
            return grid

        high_value = grid.high_value_cells(self._energy_threshold, self._signal_threshold)
        if not high_value:
            return grid

        # Cluster cells into groups (simple: first 16 cells per query)
        cluster_size = 16
        for i in range(0, len(high_value), cluster_size):
            cluster_indices = high_value[i:i + cluster_size]
            states = [grid.get(idx) for idx in cluster_indices]
            cache_key = tuple(s.identity_hash for s in states)
            if cache_key in self._query_cache:
                new_states = self._query_cache[cache_key]
            else:
                new_states = llm_query_fn(states)
                self._query_cache[cache_key] = new_states
            for idx, new_state in zip(cluster_indices, new_states):
                grid.set(idx, new_state)
                grid._cells[idx][2] = float(time.time())  # mark last_llm_query

        return grid


# ---------------------------------------------------------------------------
# Cellular engine
# ---------------------------------------------------------------------------

class CellularEngine:
    """Orchestrates CA + LLM hybrid loop."""

    def __init__(
        self,
        grid: CellularGrid,
        ca_kernel: CAGenerationKernel,
        llm_kernel: LLMInjectionKernel,
        llm_query_fn: Optional[Callable[[List[CellState]], List[CellState]]] = None,
        target_fps: float = 60.0,
    ) -> None:
        self._grid = grid
        self._ca = ca_kernel
        self._llm = llm_kernel
        self._llm_query_fn = llm_query_fn
        self._target_fps = target_fps
        self._step_count = 0
        self._running = False
        self._history: List[Dict[str, float]] = []

    @property
    def grid(self) -> CellularGrid:
        return self._grid

    def step(self) -> None:
        """Run one CA + LLM hybrid step."""
        self._ca.step(self._grid)
        self._llm.step(self._grid, self._llm_query_fn)
        self._step_count += 1
        if self._step_count % 10 == 0:
            self._history.append({
                "step": self._step_count,
                "energy": self._grid.energy(),
                "signal_density": self._grid.signal_density(),
                "high_value_cells": len(self._grid.high_value_cells()),
            })

    def run_steps(self, n: int) -> None:
        for _ in range(n):
            self.step()

    def run(self, max_steps: Optional[int] = None) -> None:
        """Run continuously until stopped or max_steps reached."""
        self._running = True
        steps = 0
        while self._running:
            start = time.perf_counter()
            self.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            elapsed = time.perf_counter() - start
            sleep_time = max(0.0, (1.0 / self._target_fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False

    def stats(self) -> Dict[str, Any]:
        return {
            "steps": self._step_count,
            "energy": self._grid.energy(),
            "signal_density": self._grid.signal_density(),
            "high_value_cells": len(self._grid.high_value_cells()),
            "history": self._history,
        }


# ---------------------------------------------------------------------------
# Agent-cell mapper
# ---------------------------------------------------------------------------

class AgentCellMapper:
    """Bidirectional map between fleet agents and grid cells."""

    def __init__(self, grid: CellularGrid) -> None:
        self._grid = grid
        self._agent_to_cell: Dict[str, Tuple[int, ...]] = {}
        self._cell_to_agent: Dict[Tuple[int, ...], str] = {}
        self._next_slot = self._iter_slots()

    def _iter_slots(self):
        shape = self._grid.shape
        if self._grid.ndim == 2:
            for y in range(shape[0]):
                for x in range(shape[1]):
                    yield (y, x)
        elif self._grid.ndim == 3:
            for z in range(shape[0]):
                for y in range(shape[1]):
                    for x in range(shape[2]):
                        yield (z, y, x)

    def spawn_agent(self, agent_id: str, state: Optional[CellState] = None) -> Tuple[int, ...]:
        if agent_id in self._agent_to_cell:
            return self._agent_to_cell[agent_id]
        try:
            idx = next(self._next_slot)
        except StopIteration:
            # Grid full: overwrite oldest
            idx = random.choice(list(self._cell_to_agent.keys()))
            old_agent = self._cell_to_agent.pop(idx)
            del self._agent_to_cell[old_agent]
        self._agent_to_cell[agent_id] = idx
        self._cell_to_agent[idx] = agent_id
        if state is not None:
            self._grid.set(idx, state)
        return idx

    def kill_agent(self, agent_id: str) -> None:
        idx = self._agent_to_cell.pop(agent_id, None)
        if idx is not None:
            self._cell_to_agent.pop(idx, None)
            self._grid.set(idx, CellState())

    def get_agent_state(self, agent_id: str) -> Optional[CellState]:
        idx = self._agent_to_cell.get(agent_id)
        if idx is None:
            return None
        return self._grid.get(idx)

    def get_cell_agent(self, idx: Tuple[int, ...]) -> Optional[str]:
        return self._cell_to_agent.get(idx)

    def move_agent(self, agent_id: str, new_idx: Tuple[int, ...]) -> bool:
        old_idx = self._agent_to_cell.get(agent_id)
        if old_idx is None:
            return False
        if new_idx in self._cell_to_agent:
            return False
        del self._cell_to_agent[old_idx]
        self._agent_to_cell[agent_id] = new_idx
        self._cell_to_agent[new_idx] = agent_id
        # Swap cell states
        state = self._grid.get(old_idx)
        self._grid.set(old_idx, CellState())
        self._grid.set(new_idx, state)
        return True

    def agent_count(self) -> int:
        return len(self._agent_to_cell)

    def cell_count(self) -> int:
        return len(self._cell_to_agent)
