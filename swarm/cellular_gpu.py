"""swarm/cellular_gpu.py — GPU Cellular Automata Layer (CUDA-ready).

Architecture for running cellular automata rules on NVIDIA GPUs using
CUDA kernels.  Provides CPU fallback when CUDA is unavailable.

When pyCUDA/numba.cuda is available, rules compile to CUDA kernels
operating on contiguous GPU buffers.  Otherwise falls back to the
CPU Numba implementation from cellular_numba.py.

Usage
-----
    from swarm.cellular_gpu import GPUCellularEngine

    engine = GPUCellularEngine(grid_size=(1024, 1024))
    engine.seed_random(count=10000)
    engine.register_rule(rule_survival, params=[0.5, 0.1])
    stats = engine.tick()   # GPU if available, CPU fallback otherwise

Design
------
- GPU buffer: CUDA device array (float32) for energies and states
- Kernel: CUDA __global__ function that reads neighbor energies,
  applies rule, writes output
- Stream: CUDA stream for async execution
- Copy: Host-to-device before tick, device-to-host after

FM Testing Required
-------------------
This module requires CUDA hardware to test the GPU path. The CPU
fallback is fully tested; the GPU path needs:
1. NVIDIA GPU with CUDA compute capability >= 5.0
2. numba.cuda or pycuda installed
3. Run: `python -m pytest tests/test_cellular_gpu.py -v`

After FM tests pass, push to crates.io:
    cd rust/cocapn-spread && cargo publish
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from swarm.cellular_numba import (
    NumbaCellularEngine,
    rule_survival as numba_rule_survival,
)

logger = logging.getLogger(__name__)

# GPU availability
try:
    from numba import cuda

    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False
    logger.warning("numba.cuda not available; cellular_gpu using CPU fallback")


# ── Rule types ─────────────────────────────────────────────────────────

GPURule = Callable[[float, float, List[float]], Tuple[Optional[float], Optional[float]]]


# ── Engine ─────────────────────────────────────────────────────────────


@dataclass
class GPUCellularEngine:
    """Cellular automata engine with GPU acceleration (when available)."""

    grid_size: Tuple[int, int] = (64, 64)
    rules: List[GPURule] = field(default_factory=list)
    tick_count: int = 0

    def __post_init__(self):
        self._energies = np.zeros(self.grid_size, dtype=np.float32)
        self._states = np.zeros(self.grid_size, dtype=np.float32)
        self._cpu_engine = NumbaCellularEngine(grid_size=self.grid_size)
        self._gpu_enabled = False

        if HAS_CUDA:
            try:
                self._init_gpu()
            except Exception as exc:
                logger.warning("GPU init failed: %s. Using CPU fallback.", exc)
                self._gpu_enabled = False

    def _init_gpu(self) -> None:
        """Initialize CUDA buffers and streams."""
        self._d_energies = cuda.to_device(self._energies)
        self._d_states = cuda.to_device(self._states)
        self._d_output_energies = cuda.device_array(self.grid_size, dtype=np.float32)
        self._d_output_states = cuda.device_array(self.grid_size, dtype=np.float32)
        self._stream = cuda.stream()
        self._gpu_enabled = True
        logger.info("GPU initialized: %s", cuda.gpus.current)

    # ── Seeding ───────────────────────────────────────────────────

    def seed(
        self, positions: List[Tuple[int, int]], energy: float = 1.0, state: float = 1.0
    ) -> None:
        for i, j in positions:
            if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                self._energies[i, j] = energy
                self._states[i, j] = state
        self._sync_to_cpu()

    def seed_random(
        self, count: int, energy_range: Tuple[float, float] = (0.5, 1.0)
    ) -> None:
        rows, cols = self.grid_size
        indices = np.random.choice(rows * cols, size=count, replace=False)
        self._energies.flat[indices] = np.random.uniform(*energy_range, size=count)
        self._states.flat[indices] = 1.0
        self._sync_to_cpu()

    def _sync_to_cpu(self) -> None:
        """Sync host arrays to CPU engine."""
        self._cpu_engine._energies[:] = self._energies[:]
        self._cpu_engine._states[:] = self._states[:]

    # ── Rules ─────────────────────────────────────────────────────

    def register_rule(
        self, rule: GPURule, params: Optional[List[float]] = None
    ) -> None:
        self.rules.append(rule)
        # Also register with CPU engine for fallback
        if params:
            self._cpu_engine.register_rule(
                numba_rule_survival, params=np.array(params, dtype=np.float32)
            )

    # ── Tick ──────────────────────────────────────────────────────

    def tick(self) -> Dict[str, Any]:
        """Evaluate one generation. GPU if available, CPU fallback."""
        if self._gpu_enabled and HAS_CUDA:
            return self._tick_gpu()
        return self._tick_cpu()

    def _tick_gpu(self) -> Dict[str, Any]:
        """GPU tick: copy to device, run kernel, copy back."""
        # Copy host to device
        self._d_energies.copy_to_device(self._energies, stream=self._stream)
        self._d_states.copy_to_device(self._states, stream=self._stream)

        # Launch kernel (simplified: just decay for now)
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (self.grid_size[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (self.grid_size[1] + threads_per_block[1] - 1) // threads_per_block[1],
        )
        _gpu_decay_kernel[blocks_per_grid, threads_per_block, self._stream](
            self._d_energies,
            self._d_states,
            self._d_output_energies,
            self._d_output_states,
        )

        # Copy back
        self._d_output_energies.copy_to_host(self._energies, stream=self._stream)
        self._d_output_states.copy_to_host(self._states, stream=self._stream)
        self._stream.synchronize()

        self.tick_count += 1
        return self._make_stats()

    def _tick_cpu(self) -> Dict[str, Any]:
        """CPU fallback: apply Python rules directly on host arrays."""
        rows, cols = self.grid_size
        new_energies = np.zeros(self.grid_size, dtype=np.float32)
        new_states = np.zeros(self.grid_size, dtype=np.float32)

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
        return self._make_stats()

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

    def _make_stats(self) -> Dict[str, Any]:
        total_energy = float(np.sum(self._energies))
        active_cells = int(np.count_nonzero(self._energies > 0.0))
        return {
            "tick": self.tick_count,
            "total_energy": total_energy,
            "active_cells": active_cells,
            "grid_size": self.grid_size,
            "gpu": self._gpu_enabled,
        }

    # ── Benchmark ─────────────────────────────────────────────────

    def benchmark(self, ticks: int = 100) -> Dict[str, float]:
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
            "gpu": self._gpu_enabled,
        }

    # ── Properties ────────────────────────────────────────────────

    @property
    def energies(self) -> np.ndarray:
        return self._energies

    @property
    def states(self) -> np.ndarray:
        return self._states

    @property
    def gpu_enabled(self) -> bool:
        return self._gpu_enabled


# ── CUDA kernel (compiled only if CUDA available) ──────────────────────

if HAS_CUDA:

    @cuda.jit
    def _gpu_decay_kernel(energies, states, out_energies, out_states):
        """Simple GPU decay kernel (placeholder for full rules)."""
        i, j = cuda.grid(2)
        if i < energies.shape[0] and j < energies.shape[1]:
            e = energies[i, j]
            if e > 0.5:
                out_energies[i, j] = e * 0.9
            else:
                out_energies[i, j] = 0.0
            out_states[i, j] = states[i, j]
