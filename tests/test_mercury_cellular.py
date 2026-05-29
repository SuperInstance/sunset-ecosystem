"""Tests for Mercury-style declarative cellular automata engine.

Covers engine initialization, rule registration, seeding, tick evaluation,
energy conservation, and bridge to Numba engine.
"""

import numpy as np
import pytest

from fleet.mercury_cellular import (
    MercuryCellularEngine,
    rule_survival,
    rule_reproduction,
    rule_diffusion,
    rule_mutation,
)


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_default_size(self):
        engine = MercuryCellularEngine()
        assert engine.grid_size == (64, 64)

    def test_custom_size(self):
        engine = MercuryCellularEngine(grid_size=(32, 32))
        assert engine.grid_size == (32, 32)

    def test_energies_zero_initially(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        assert np.all(engine.energies == 0.0)

    def test_states_zero_initially(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        assert np.all(engine.states == 0.0)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

class TestSeeding:
    def test_seed_single(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        assert engine.energies[5, 5] == 1.0
        assert engine.states[5, 5] == 1.0

    def test_seed_multiple(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(0, 0), (1, 1), (2, 2)], energy=0.5)
        assert engine.energies[0, 0] == 0.5
        assert engine.energies[1, 1] == 0.5
        assert engine.energies[2, 2] == 0.5

    def test_seed_out_of_bounds_ignored(self):
        engine = MercuryCellularEngine(grid_size=(5, 5))
        engine.seed([(10, 10)], energy=1.0)
        assert np.all(engine.energies == 0.0)

    def test_seed_random(self):
        engine = MercuryCellularEngine(grid_size=(20, 20))
        engine.seed_random(count=10)
        active = np.count_nonzero(engine.energies > 0.0)
        assert active == 10


# ---------------------------------------------------------------------------
# Rule registration
# ---------------------------------------------------------------------------

class TestRuleRegistration:
    def test_register_survival(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival)
        assert len(engine.rules) == 1

    def test_register_multiple(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival)
        engine.register_rule(rule_reproduction)
        engine.register_rule(rule_diffusion)
        assert len(engine.rules) == 3

    def test_custom_rule(self):
        def my_rule(energy, state, neighbors):
            if energy > 0.5:
                return (energy * 0.5, state)
            return (0.0, 0.0)

        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.register_rule(my_rule)
        assert len(engine.rules) == 1


# ---------------------------------------------------------------------------
# Tick evaluation
# ---------------------------------------------------------------------------

class TestTickEvaluation:
    def test_tick_empty_grid(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival)
        stats = engine.tick()
        assert stats["active_cells"] == 0
        assert stats["total_energy"] == 0.0

    def test_tick_survival(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_survival)
        stats = engine.tick()
        assert stats["active_cells"] == 1
        # Energy decayed by 10%
        assert engine.energies[5, 5] == pytest.approx(0.9, abs=0.01)

    def test_tick_survival_kills_weak(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=0.3, state=1.0)
        engine.register_rule(rule_survival)
        engine.tick()
        assert engine.energies[5, 5] == 0.0  # killed (energy < 0.5)

    def test_tick_reproduction(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        # Two energetic neighbors around (5,5)
        engine.seed([(4, 5), (6, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_reproduction)
        stats = engine.tick()
        # (5,5) should be born because it has 2 energetic neighbors
        assert engine.energies[5, 5] > 0.0

    def test_multiple_ticks(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_survival)
        history = engine.run(ticks=5)
        assert len(history) == 5
        assert history[4]["tick"] == 5

    def test_tick_count_increments(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.register_rule(rule_survival)
        engine.tick()
        engine.tick()
        assert engine.tick_count == 2

    def test_combined_rules(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_survival)
        engine.register_rule(rule_diffusion)
        stats = engine.tick()
        assert stats["active_cells"] >= 1


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    def test_energy_never_negative(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed_random(count=20)
        engine.register_rule(rule_survival)
        engine.register_rule(rule_diffusion)
        engine.register_rule(rule_mutation)
        for _ in range(10):
            engine.tick()
        assert np.all(engine.energies >= 0.0)

    def test_total_energy_history_recorded(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0)
        engine.register_rule(rule_survival)
        engine.run(ticks=5)
        assert len(engine.total_energy_history) == 5


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_runs(self):
        engine = MercuryCellularEngine(grid_size=(64, 64))
        engine.seed_random(count=100)
        engine.register_rule(rule_survival)
        result = engine.benchmark(ticks=50)
        assert result["ticks"] == 50
        assert result["ms_per_tick"] > 0.0

    def test_benchmark_with_warmup(self):
        engine = MercuryCellularEngine(grid_size=(32, 32))
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
        engine = MercuryCellularEngine(grid_size=(5, 5))
        engine.seed([(2, 2)], energy=1.0)
        d = engine.to_dict()
        assert d["grid_size"] == (5, 5)
        assert d["tick_count"] == 0
        assert len(d["energies"]) == 5


# ---------------------------------------------------------------------------
# Bridge to Numba
# ---------------------------------------------------------------------------

class TestNumbaBridge:
    def test_to_numba_engine(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0, state=1.0)
        engine.register_rule(rule_survival)
        engine.tick()

        numba_engine = engine.to_numba_engine()
        assert numba_engine.grid_size == (10, 10)
        assert numba_engine.energies[5, 5] == pytest.approx(0.9, abs=0.01)

    def test_numba_engine_runs(self):
        from swarm.cellular_numba import rule_survival as numba_rule

        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0)
        engine.tick()  # Apply Mercury rule first

        numba_engine = engine.to_numba_engine()
        numba_engine.register_rule(numba_rule, params=np.array([0.5, 0.1], dtype=np.float32))
        stats = numba_engine.tick()
        assert stats["active_cells"] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_rules_tick(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        stats = engine.tick()
        assert stats["active_cells"] == 0

    def test_single_cell_grid(self):
        engine = MercuryCellularEngine(grid_size=(1, 1))
        engine.seed([(0, 0)], energy=1.0)
        engine.register_rule(rule_survival)
        stats = engine.tick()
        assert stats["active_cells"] == 1

    def test_rectangular_grid(self):
        engine = MercuryCellularEngine(grid_size=(8, 16))
        engine.seed([(4, 8)], energy=1.0)
        engine.register_rule(rule_survival)
        stats = engine.tick()
        assert stats["active_cells"] == 1

    def test_all_cells_dead_after_survival(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=0.1)
        engine.register_rule(rule_survival)
        stats = engine.tick()
        assert stats["active_cells"] == 0

    def test_rule_returns_none_no_change(self):
        engine = MercuryCellularEngine(grid_size=(10, 10))
        engine.seed([(5, 5)], energy=1.0)
        engine.register_rule(rule_reproduction)
        engine.tick()
        # (5,5) should not be affected by reproduction rule (has no empty neighbors)
        assert engine.energies[5, 5] == 1.0
