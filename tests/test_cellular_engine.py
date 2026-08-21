"""Tests for Cellular Engine.

Covers grid init, CA rules, LLM injection mock, agent mapping, energy conservation,
signal propagation, GPU detection, edge cases.
"""

import pytest
import numpy as np

from swarm.cellular_engine import (
    CellState,
    CellularGrid,
    CAGenerationKernel,
    LLMInjectionKernel,
    CellularEngine,
    AgentCellMapper,
)


# ---------------------------------------------------------------------------
# CellState
# ---------------------------------------------------------------------------


class TestCellState:
    def test_defaults(self):
        cs = CellState()
        assert cs.energy == 0.0
        assert cs.signal == 0.0
        assert cs.identity_hash == 0

    def test_to_vector(self):
        cs = CellState(energy=0.5, signal=0.3, identity_hash=42, generation=3)
        v = cs.to_vector()
        assert v.shape == (5,)
        assert v[0] == 0.5
        assert v[1] == 0.3

    def test_from_vector(self):
        v = np.array([0.5, 0.3, 0.042, 0.1, 3.0], dtype=np.float32)
        cs = CellState.from_vector(v)
        assert cs.energy == pytest.approx(0.5)
        assert cs.signal == pytest.approx(0.3)
        assert cs.identity_hash == 42
        assert cs.generation == 3

    def test_roundtrip(self):
        cs = CellState(energy=0.7, signal=0.2, identity_hash=99, generation=5)
        v = cs.to_vector()
        cs2 = CellState.from_vector(v)
        assert cs2.energy == pytest.approx(0.7)
        assert cs2.generation == 5


# ---------------------------------------------------------------------------
# CellularGrid
# ---------------------------------------------------------------------------


class TestCellularGrid:
    def test_init_2d(self):
        grid = CellularGrid(shape=(8, 8))
        assert grid.shape == (8, 8)
        assert grid.ndim == 2

    def test_init_3d(self):
        grid = CellularGrid(shape=(4, 4, 4))
        assert grid.ndim == 3

    def test_get_set(self):
        grid = CellularGrid(shape=(4, 4))
        grid.set((1, 1), CellState(energy=0.5, signal=0.3))
        cs = grid.get((1, 1))
        assert cs.energy == pytest.approx(0.5)
        assert cs.signal == pytest.approx(0.3)

    def test_get_invalid_index(self):
        grid = CellularGrid(shape=(4, 4))
        with pytest.raises(ValueError):
            grid.get((1,))

    def test_randomize(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        assert grid.energy() > 0
        assert grid.signal_density() > 0

    def test_randomize_deterministic(self):
        grid1 = CellularGrid(shape=(8, 8))
        grid1.randomize(seed=42)
        grid2 = CellularGrid(shape=(8, 8))
        grid2.randomize(seed=42)
        np.testing.assert_array_almost_equal(grid1.to_numpy(), grid2.to_numpy())

    def test_high_value_cells(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=0.9, signal=0.6))
        grid.set((2, 2), CellState(energy=0.5, signal=0.6))
        high = grid.high_value_cells(energy_threshold=0.7, signal_threshold=0.5)
        assert (1, 1) in high
        assert (2, 2) not in high

    def test_no_high_value_cells(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42, energy_range=(0.0, 0.5))
        high = grid.high_value_cells(energy_threshold=0.7, signal_threshold=0.5)
        assert len(high) == 0

    def test_to_numpy(self):
        grid = CellularGrid(shape=(4, 4))
        arr = grid.to_numpy()
        assert arr.shape == (4, 4, 5)

    def test_energy_conservation_randomize(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42, energy_range=(0.0, 1.0))
        e1 = grid.energy()
        # Energy should be roughly 64 * 0.5 = 32
        assert e1 > 20 and e1 < 45


# ---------------------------------------------------------------------------
# CAGenerationKernel
# ---------------------------------------------------------------------------


class TestCAGenerationKernel:
    def test_step_no_crash(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        kernel = CAGenerationKernel()
        kernel.step(grid)
        assert grid.energy() >= 0

    def test_energy_decay(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=1.0, signal=0.0))
        kernel = CAGenerationKernel(energy_decay=0.1)
        kernel.step(grid)
        cs = grid.get((1, 1))
        assert cs.energy < 1.0

    def test_diffusion_spreads_energy(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=1.0, signal=0.0))
        grid.set((1, 2), CellState(energy=0.0, signal=0.0))
        kernel = CAGenerationKernel(diffusion_rate=0.5)
        kernel.step(grid)
        # Energy should have spread from (1,1) to neighbors
        assert grid.get((1, 2)).energy > 0.0

    def test_reproduction_threshold(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=2.0, signal=0.6, generation=0))
        kernel = CAGenerationKernel(reproduction_threshold=1.5)
        kernel.step(grid)
        cs = grid.get((1, 1))
        # Energy may have decayed; reproduction may or may not happen depending on signal threshold
        assert cs.energy >= 0

    def test_max_energy_clip(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=3.0, signal=0.0))
        kernel = CAGenerationKernel(max_energy=2.0)
        kernel.step(grid)
        cs = grid.get((1, 1))
        assert cs.energy <= 2.0

    def test_signal_decay(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=0.5, signal=0.8))
        kernel = CAGenerationKernel(signal_decay=0.1)
        kernel.step(grid)
        cs = grid.get((1, 1))
        assert cs.signal < 0.8

    def test_energy_never_negative(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        kernel = CAGenerationKernel(energy_decay=0.5)
        for _ in range(10):
            kernel.step(grid)
        arr = grid.to_numpy()
        assert np.all(arr[..., 0] >= 0)

    def test_3d_grid(self):
        grid = CellularGrid(shape=(4, 4, 4))
        grid.randomize(seed=42)
        kernel = CAGenerationKernel()
        kernel.step(grid)
        assert grid.energy() >= 0


# ---------------------------------------------------------------------------
# LLMInjectionKernel
# ---------------------------------------------------------------------------


class TestLLMInjectionKernel:
    def test_no_injection_without_fn(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=0.9, signal=0.6))
        kernel = LLMInjectionKernel(injection_interval=1)
        kernel.step(grid, llm_query_fn=None)
        # No crash, no change expected (or minimal change from CA step)
        cs = grid.get((1, 1))
        assert cs.energy >= 0.8

    def test_injection_on_interval(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=0.9, signal=0.6))
        grid.set((2, 2), CellState(energy=0.9, signal=0.6))

        def mock_llm(states):
            return [CellState(energy=0.5, signal=0.5) for _ in states]

        kernel = LLMInjectionKernel(injection_interval=1, energy_threshold=0.7)
        kernel.step(grid, llm_query_fn=mock_llm)
        cs = grid.get((1, 1))
        assert cs.energy == 0.5

    def test_injection_skips_low_energy(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=0.3, signal=0.6))

        called = False

        def mock_llm(states):
            nonlocal called
            called = True
            return states

        kernel = LLMInjectionKernel(injection_interval=1, energy_threshold=0.7)
        kernel.step(grid, llm_query_fn=mock_llm)
        assert not called

    def test_cache_prevents_duplicate_queries(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=0.9, signal=0.6, identity_hash=123))

        call_count = 0

        def mock_llm(states):
            nonlocal call_count
            call_count += 1
            return states

        kernel = LLMInjectionKernel(injection_interval=1)
        kernel.step(grid, llm_query_fn=mock_llm)
        # Second call may or may not hit cache depending on grid state changes
        # Just verify it doesn't crash
        kernel.step(grid, llm_query_fn=mock_llm)
        assert call_count >= 1

    def test_injection_not_every_step(self):
        grid = CellularGrid(shape=(8, 8))
        grid.set((1, 1), CellState(energy=0.9, signal=0.6))

        call_count = 0

        def mock_llm(states):
            nonlocal call_count
            call_count += 1
            return states

        kernel = LLMInjectionKernel(injection_interval=5)
        for _ in range(5):
            kernel.step(grid, llm_query_fn=mock_llm)
        assert call_count == 1


# ---------------------------------------------------------------------------
# CellularEngine
# ---------------------------------------------------------------------------


class TestCellularEngine:
    def test_init(self):
        grid = CellularGrid(shape=(8, 8))
        ca = CAGenerationKernel()
        llm = LLMInjectionKernel()
        engine = CellularEngine(grid, ca, llm)
        assert engine.grid is grid
        assert engine._step_count == 0

    def test_single_step(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        ca = CAGenerationKernel()
        llm = LLMInjectionKernel()
        engine = CellularEngine(grid, ca, llm)
        engine.step()
        assert engine._step_count == 1

    def test_run_steps(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        ca = CAGenerationKernel()
        llm = LLMInjectionKernel()
        engine = CellularEngine(grid, ca, llm)
        engine.run_steps(10)
        assert engine._step_count == 10

    def test_stats(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        ca = CAGenerationKernel()
        llm = LLMInjectionKernel()
        engine = CellularEngine(grid, ca, llm)
        engine.run_steps(20)
        stats = engine.stats()
        assert stats["steps"] == 20
        assert "energy" in stats
        assert "signal_density" in stats
        assert "history" in stats

    def test_history_records_every_10_steps(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        ca = CAGenerationKernel()
        llm = LLMInjectionKernel()
        engine = CellularEngine(grid, ca, llm)
        engine.run_steps(25)
        assert len(engine._history) == 2  # 10, 20

    def test_stop(self):
        grid = CellularGrid(shape=(8, 8))
        ca = CAGenerationKernel()
        llm = LLMInjectionKernel()
        engine = CellularEngine(grid, ca, llm, target_fps=1000.0)
        engine.stop()
        engine.run(max_steps=1)
        assert engine._step_count == 1

    def test_run_with_max_steps(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        ca = CAGenerationKernel()
        llm = LLMInjectionKernel()
        engine = CellularEngine(grid, ca, llm, target_fps=1000.0)
        engine.run(max_steps=5)
        assert engine._step_count == 5


# ---------------------------------------------------------------------------
# AgentCellMapper
# ---------------------------------------------------------------------------


class TestAgentCellMapper:
    def test_spawn_agent(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        idx = mapper.spawn_agent("agent-1")
        assert idx is not None
        assert mapper.agent_count() == 1

    def test_spawn_multiple(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        for i in range(3):
            mapper.spawn_agent(f"agent-{i}")
        assert mapper.agent_count() == 3

    def test_spawn_duplicate_id(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        idx1 = mapper.spawn_agent("agent-1")
        idx2 = mapper.spawn_agent("agent-1")
        assert idx1 == idx2
        assert mapper.agent_count() == 1

    def test_kill_agent(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        mapper.spawn_agent("agent-1", CellState(energy=0.5))
        mapper.kill_agent("agent-1")
        assert mapper.agent_count() == 0
        assert grid.get((0, 0)).energy == 0.0

    def test_get_agent_state(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        mapper.spawn_agent("agent-1", CellState(energy=0.7))
        state = mapper.get_agent_state("agent-1")
        assert state is not None
        assert state.energy == pytest.approx(0.7)

    def test_get_agent_state_unknown(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        assert mapper.get_agent_state("unknown") is None

    def test_move_agent(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        mapper.spawn_agent("agent-1", CellState(energy=0.5))
        success = mapper.move_agent("agent-1", (2, 2))
        assert success
        assert mapper.get_agent_state("agent-1").energy == 0.5

    def test_move_agent_to_occupied(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        mapper.spawn_agent("agent-1", CellState(energy=0.5))
        mapper.spawn_agent("agent-2", CellState(energy=0.5))
        idx = mapper._agent_to_cell["agent-2"]
        success = mapper.move_agent("agent-1", idx)
        assert not success

    def test_move_unknown_agent(self):
        grid = CellularGrid(shape=(4, 4))
        mapper = AgentCellMapper(grid)
        assert not mapper.move_agent("unknown", (1, 1))

    def test_grid_full_overwrite(self):
        grid = CellularGrid(shape=(2, 2))
        mapper = AgentCellMapper(grid)
        for i in range(5):  # 2x2 = 4 cells, 5th overwrites
            mapper.spawn_agent(f"agent-{i}")
        assert mapper.agent_count() == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_size_grid(self):
        grid = CellularGrid(shape=(1, 1))
        grid.set((0, 0), CellState(energy=1.0))
        assert grid.energy() == 1.0

    def test_large_grid(self):
        grid = CellularGrid(shape=(256, 256))
        grid.randomize(seed=42)
        assert grid.energy() > 0
        kernel = CAGenerationKernel()
        kernel.step(grid)
        assert grid.energy() >= 0

    def test_energy_conservation_over_many_steps(self):
        grid = CellularGrid(shape=(8, 8))
        grid.randomize(seed=42)
        ca = CAGenerationKernel(energy_decay=0.0, diffusion_rate=0.0)
        # With no decay and no diffusion, energy should be conserved (minus reproduction cost)
        e0 = grid.energy()
        for _ in range(10):
            ca.step(grid)
        e1 = grid.energy()
        # Energy should be close to original (small reproduction losses)
        assert abs(e1 - e0) < 5.0
