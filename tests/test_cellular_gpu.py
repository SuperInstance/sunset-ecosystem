"""Tests for GPU Cellular Automata Layer.

GPU-specific tests are skipped when CUDA is unavailable.
CPU fallback is fully tested.
"""

import numpy as np
import pytest

from swarm.cellular_gpu import GPUCellularEngine, HAS_CUDA


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_default_size(self):
        engine = GPUCellularEngine()
        assert engine.grid_size == (64, 64)
        assert engine.gpu_enabled is False  # no CUDA on test node

    def test_custom_size(self):
        engine = GPUCellularEngine(grid_size=(32, 32))
        assert engine.grid_size == (32, 32)

    def test_energies_zero(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        assert np.all(engine.energies == 0.0)

    def test_states_zero(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        assert np.all(engine.states == 0.0)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_seed_single(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        assert engine.energies[5, 5] == 1.0
        assert engine.states[5, 5] == 1.0

    def test_seed_multiple(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed([(0, 0), (1, 1), (2, 2)], energy=0.5)
        assert engine.energies[0, 0] == 0.5
        assert engine.energies[1, 1] == 0.5
        assert engine.energies[2, 2] == 0.5

    def test_seed_out_of_bounds_ignored(self):
        engine = GPUCellularEngine(grid_size=(5, 5))
        engine.seed([(10, 10)], energy=1.0)
        assert np.all(engine.energies == 0.0)

    def test_seed_random(self):
        engine = GPUCellularEngine(grid_size=(20, 20))
        engine.seed_random(count=10)
        active = np.count_nonzero(engine.energies > 0.0)
        assert active == 10


# ---------------------------------------------------------------------------
# Tick (CPU fallback)
# ---------------------------------------------------------------------------


class TestTickCPU:
    def test_tick_empty(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        stats = engine.tick()
        assert stats["active_cells"] == 0
        assert stats["total_energy"] == 0.0
        assert stats["gpu"] is False

    def test_tick_with_survival_rule(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(lambda e, s, n: (e * 0.9, s) if e > 0.5 else (0.0, 0.0))
        stats = engine.tick()
        assert stats["active_cells"] == 1
        assert engine.energies[5, 5] == pytest.approx(0.9, abs=0.01)

    def test_tick_kills_weak(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=0.3, state=1.0)
        engine.register_rule(lambda e, s, n: (e * 0.9, s) if e > 0.5 else (0.0, 0.0))
        engine.tick()
        assert engine.energies[5, 5] == 0.0

    def test_multiple_ticks(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(lambda e, s, n: (e * 0.9, s) if e > 0.5 else (0.0, 0.0))
        for _ in range(5):
            engine.tick()
        assert engine.tick_count == 5
        assert engine.energies[5, 5] == pytest.approx(0.59, abs=0.01)

    def test_energy_never_negative(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed_random(count=20)
        engine.register_rule(lambda e, s, n: (max(0.0, e - 0.1), s))
        for _ in range(10):
            engine.tick()
        assert np.all(engine.energies >= 0.0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_runs(self):
        engine = GPUCellularEngine(grid_size=(32, 32))
        engine.seed_random(count=50)
        engine.register_rule(lambda e, s, n: (e * 0.9, s) if e > 0.5 else (0.0, 0.0))
        result = engine.benchmark(ticks=20)
        assert result["ticks"] == 20
        assert result["ms_per_tick"] > 0.0
        assert result["gpu"] is False

    def test_benchmark_with_rules(self):
        engine = GPUCellularEngine(grid_size=(64, 64))
        engine.seed_random(count=100)
        engine.register_rule(lambda e, s, n: (e * 0.95, s) if e > 0.5 else (0.0, 0.0))
        result = engine.benchmark(ticks=50)
        assert result["ticks_per_second"] > 0.0


# ---------------------------------------------------------------------------
# GPU-specific (skipped if no CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestGPU:
    def test_gpu_enabled(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        assert engine.gpu_enabled is True

    def test_gpu_tick(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0)
        stats = engine.tick()
        assert stats["gpu"] is True
        assert stats["active_cells"] == 1

    def test_gpu_benchmark(self):
        engine = GPUCellularEngine(grid_size=(256, 256))
        engine.seed_random(count=1000)
        engine.register_rule(lambda e, s, n: (e * 0.9, s) if e > 0.5 else (0.0, 0.0))
        result = engine.benchmark(ticks=100)
        assert result["gpu"] is True
        assert result["ticks_per_second"] > 1000  # GPU should be fast


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_cell_grid(self):
        engine = GPUCellularEngine(grid_size=(1, 1))
        engine.seed([(0, 0)], energy=1.0)
        engine.register_rule(lambda e, s, n: (e * 0.9, s) if e > 0.5 else (0.0, 0.0))
        stats = engine.tick()
        assert stats["active_cells"] == 1

    def test_rectangular_grid(self):
        engine = GPUCellularEngine(grid_size=(8, 16))
        engine.seed([(4, 8)], energy=1.0)
        engine.register_rule(lambda e, s, n: (e * 0.9, s) if e > 0.5 else (0.0, 0.0))
        stats = engine.tick()
        assert stats["active_cells"] == 1

    def test_no_rules(self):
        engine = GPUCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0)
        stats = engine.tick()
        assert stats["active_cells"] == 1
        assert engine.energies[5, 5] == 1.0  # no rules, no change
