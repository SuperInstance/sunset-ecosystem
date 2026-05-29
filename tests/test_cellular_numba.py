"""Tests for Numba-JIT cellular automata engine.

Covers engine initialization, rule registration, seeding, tick
evaluation, benchmarking, and energy conservation.  Numba-specific
benchmarks are skipped if numba is not installed.
"""

import numpy as np
import pytest

from swarm.cellular_numba import (
    NumbaCellularEngine,
    rule_survival,
    rule_reproduction,
    rule_diffusion,
    _ensure_jit,
    HAS_NUMBA,
)


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_default_size(self):
        engine = NumbaCellularEngine()
        assert engine.grid_size == (64, 64)

    def test_custom_size(self):
        engine = NumbaCellularEngine(grid_size=(32, 32))
        assert engine.grid_size == (32, 32)

    def test_energies_zero_initially(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        assert np.all(engine.energies == 0.0)

    def test_states_zero_initially(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        assert np.all(engine.states == 0.0)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

class TestSeeding:
    def test_seed_single(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        assert engine.energies[5, 5] == 1.0
        assert engine.states[5, 5] == 1.0

    def test_seed_multiple(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed([(0, 0), (1, 1), (2, 2)], energy=0.5)
        assert engine.energies[0, 0] == 0.5
        assert engine.energies[1, 1] == 0.5
        assert engine.energies[2, 2] == 0.5

    def test_seed_out_of_bounds_ignored(self):
        engine = NumbaCellularEngine(grid_size=(5, 5))
        engine.seed([(10, 10)], energy=1.0)
        assert np.all(engine.energies == 0.0)

    def test_seed_random(self):
        engine = NumbaCellularEngine(grid_size=(20, 20))
        engine.seed_random(count=10)
        active = np.count_nonzero(engine.energies > 0.0)
        assert active == 10


# ---------------------------------------------------------------------------
# Rule registration
# ---------------------------------------------------------------------------

class TestRuleRegistration:
    def test_register_survival(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival, params=np.array([0.5, 0.1], dtype=np.float32))
        assert len(engine.rules) == 1

    def test_register_multiple(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival)
        engine.register_rule(rule_reproduction)
        assert len(engine.rules) == 2

    def test_auto_wrap_python_rule(self):
        def my_rule(energies, states, neighbors, out_e, out_s, params):
            out_e[:] = energies * 0.9
            out_s[:] = states

        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.register_rule(my_rule)
        assert len(engine.rules) == 1
        # Should be callable
        engine.rules[0](
            engine.energies, engine.states,
            np.zeros((10, 10), dtype=np.float32),
            np.zeros((10, 10), dtype=np.float32),
            np.zeros((10, 10), dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        )

    def test_params_stored(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        p = np.array([0.3, 0.2, 0.1], dtype=np.float32)
        engine.register_rule(rule_survival, params=p, name="survival")
        assert "survival" in engine.params
        assert np.allclose(engine.params["survival"], p)


# ---------------------------------------------------------------------------
# Tick evaluation
# ---------------------------------------------------------------------------

class TestTickEvaluation:
    def test_tick_empty_grid(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival, params=np.array([0.5, 0.1], dtype=np.float32))
        stats = engine.tick()
        assert stats["active_cells"] == 0
        assert stats["total_energy"] == 0.0

    def test_tick_survival(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_survival, params=np.array([0.5, 0.1], dtype=np.float32))
        stats = engine.tick()
        assert stats["active_cells"] == 1
        # Energy decayed by 10%
        assert engine.energies[5, 5] == pytest.approx(0.9, abs=0.01)

    def test_tick_survival_kills_weak(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=0.3, state=1.0)
        engine.register_rule(rule_survival, params=np.array([0.5, 0.1], dtype=np.float32))
        engine.tick()
        assert engine.energies[5, 5] == 0.0  # killed (energy < 0.5)

    def test_tick_reproduction(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        # Two energetic neighbors around (5,5)
        engine.seed([(4, 5), (6, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_reproduction, params=np.array([0.5, 0.8, 0.01], dtype=np.float32))
        stats = engine.tick()
        # (5,5) should be born because it has 2 energetic neighbors
        assert engine.energies[5, 5] > 0.0

    def test_multiple_ticks(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_survival, params=np.array([0.5, 0.1], dtype=np.float32))
        history = engine.run(ticks=5)
        assert len(history) == 5
        assert history[4]["tick"] == 5

    def test_tick_count_increments(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival)
        engine.tick()
        engine.tick()
        assert engine.tick_count == 2


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    def test_energy_never_negative(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed_random(count=20)
        engine.register_rule(rule_survival)
        engine.register_rule(rule_diffusion)
        for _ in range(10):
            engine.tick()
        assert np.all(engine.energies >= 0.0)

    def test_total_energy_history_recorded(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0)
        engine.register_rule(rule_survival)
        engine.run(ticks=5)
        assert len(engine.total_energy_history) == 5


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestBenchmark:
    def test_benchmark_runs(self):
        engine = NumbaCellularEngine(grid_size=(64, 64))
        engine.seed_random(count=100)
        engine.register_rule(rule_survival)
        result = engine.benchmark(ticks=50)
        assert result["ticks"] == 50
        assert result["ms_per_tick"] > 0.0

    def test_benchmark_with_warmup(self):
        engine = NumbaCellularEngine(grid_size=(32, 32))
        engine.seed_random(count=50)
        engine.register_rule(rule_survival)
        engine.register_rule(rule_reproduction)
        result = engine.benchmark(ticks=20)
        assert result["ticks_per_second"] > 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_basic(self):
        engine = NumbaCellularEngine(grid_size=(5, 5))
        engine.seed([(2, 2)], energy=1.0)
        d = engine.to_dict()
        assert d["grid_size"] == (5, 5)
        assert d["tick_count"] == 0
        assert len(d["energies"]) == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_rules_tick(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        stats = engine.tick()
        assert stats["active_cells"] == 0

    def test_single_cell_grid(self):
        engine = NumbaCellularEngine(grid_size=(1, 1))
        engine.seed([(0, 0)], energy=1.0)
        engine.register_rule(rule_survival)
        stats = engine.tick()
        assert stats["active_cells"] == 1

    def test_rectangular_grid(self):
        engine = NumbaCellularEngine(grid_size=(8, 16))
        engine.seed([(4, 8)], energy=1.0)
        engine.register_rule(rule_survival)
        stats = engine.tick()
        assert stats["active_cells"] == 1

    def test_all_cells_dead_after_survival(self):
        engine = NumbaCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=0.1)
        engine.register_rule(rule_survival, params=np.array([0.5, 0.1], dtype=np.float32))
        stats = engine.tick()
        assert stats["active_cells"] == 0

    def test_ensure_jit_idempotent(self):
        jit1 = _ensure_jit(rule_survival)
        jit2 = _ensure_jit(jit1)
        assert jit1 is jit2
