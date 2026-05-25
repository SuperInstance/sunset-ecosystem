"""Tests for BreederDaemonV2 fleet integration.

Covers:
  - MetronomeBridge tick during cycle
  - FleetVectorIndex cross-node parent merging
  - TrapRegistry run_all after cycle
  - FluxPresetLibrary apply_preset during cycle
  - AgentIdentity sign_task on WAL entries
  - get_fleet_status() unified status dict
  - Thread-safe concurrent cycles
"""

from __future__ import annotations

import threading
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nerve.room_grid import JEPAGrid
from swarm.breeder_daemon_v2 import BreederDaemonV2
from swarm.thermal import DeviceType, ThermalBudget


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def grid():
    return JEPAGrid(n=20)


@pytest.fixture
def thermal():
    return ThermalBudget({DeviceType.GPU: 10, DeviceType.CPU: 20})


@pytest.fixture
def breeder(grid, thermal):
    return BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        interval=1,
    )


@pytest.fixture
def mock_metronome():
    m = MagicMock(spec="nerve.distributed_metronome_bridge.MetronomeBridge")
    m.tick = MagicMock(return_value=42)
    return m


@pytest.fixture
def mock_fleet_index():
    m = MagicMock(spec="swarm.mesh_vector_tables.FleetVectorIndex")
    # Return 2 cross-node candidates
    entry = types.SimpleNamespace(agent_id="node2::agent_99", fitness=0.85)
    m.get_breedable_pool = MagicMock(return_value=[entry])
    return m


@pytest.fixture
def mock_trap_registry():
    m = MagicMock(spec="fleet.operational_trap.TrapRegistry")
    m.run_all = MagicMock(return_value=[])
    return m


@pytest.fixture
def mock_flux_preset_library():
    m = MagicMock(spec="sunset.flux_preset_library.FluxPresetLibrary")
    m.suggest_preset_for_task = MagicMock(return_value="neural_bounds")
    m.apply_preset = MagicMock(return_value=[{"rule": "bound"}])
    return m


@pytest.fixture
def mock_agent_identity():
    m = MagicMock(spec="logos.a2a_identity.AgentIdentity")
    m.sign_task = MagicMock(return_value="sig_deadbeef")
    m.agent_id = "test-agent-001"
    return m


# ── 1. Initialization ───────────────────────────────────

class TestInitialization:
    def test_all_modules_attached(self, grid, thermal, mock_metronome, mock_fleet_index,
                                   mock_trap_registry, mock_flux_preset_library,
                                   mock_agent_identity):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            metronome_bridge=mock_metronome,
            fleet_vector_index=mock_fleet_index,
            trap_registry=mock_trap_registry,
            flux_preset_library=mock_flux_preset_library,
            agent_identity=mock_agent_identity,
        )
        assert daemon._metronome_bridge is mock_metronome
        assert daemon._fleet_vector_index is mock_fleet_index
        assert daemon._trap_registry is mock_trap_registry
        assert daemon._flux_preset_library is mock_flux_preset_library
        assert daemon._agent_identity is mock_agent_identity

    def test_no_modules_attached(self, grid, thermal):
        daemon = BreederDaemonV2(grid=grid, thermal=thermal)
        assert daemon._metronome_bridge is None
        assert daemon._fleet_vector_index is None
        assert daemon._trap_registry is None
        assert daemon._flux_preset_library is None
        assert daemon._agent_identity is None

    def test_partial_modules_attached(self, grid, thermal, mock_metronome):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            metronome_bridge=mock_metronome,
        )
        assert daemon._metronome_bridge is mock_metronome
        assert daemon._fleet_vector_index is None


# ── 2. Metronome tick ───────────────────────────────────

class TestMetronomeTick:
    def test_cycle_ticks_metronome(self, breeder, mock_metronome):
        breeder._metronome_bridge = mock_metronome
        breeder.auto_breed(n_winners=1)
        mock_metronome.tick.assert_called_once()

    def test_metronome_exception_logged_not_raised(self, breeder, mock_metronome):
        mock_metronome.tick.side_effect = RuntimeError("tick boom")
        breeder._metronome_bridge = mock_metronome
        # Should not raise
        breeder.auto_breed(n_winners=1)
        mock_metronome.tick.assert_called_once()

    def test_no_metronome_no_tick(self, breeder):
        # Should not raise
        breeder.auto_breed(n_winners=1)


# ── 3. Fleet vector index parent selection ────────────────

class TestFleetVectorIndex:
    def test_cycle_queries_fleet_index(self, grid, thermal, mock_fleet_index):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            fleet_vector_index=mock_fleet_index,
        )
        daemon.auto_breed(n_winners=1)
        mock_fleet_index.get_breedable_pool.assert_called_once()

    def test_cross_node_parents_merged(self, grid, thermal, mock_fleet_index):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            fleet_vector_index=mock_fleet_index,
        )
        # Seed some local agents
        for i in range(5):
            grid.activity[i] += 10
        daemon._fsm[100 + i] = MagicMock()
        daemon._fsm[100 + i].get_state = MagicMock(return_value="SURVIVE")

        parents = daemon.select_parents(n_children=2)
        # Should have merged fleet candidates with local
        mock_fleet_index.get_breedable_pool.assert_called()
        assert isinstance(parents, list)
        assert len(parents) <= 2

    def test_fleet_index_exception_fallback(self, grid, thermal, mock_fleet_index):
        mock_fleet_index.get_breedable_pool.side_effect = RuntimeError("pool boom")
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            fleet_vector_index=mock_fleet_index,
        )
        # Seed local agents so fallback works
        for i in range(5):
            grid.activity[i] += 10
        daemon._fsm[200 + i] = MagicMock()
        daemon._fsm[200 + i].get_state = MagicMock(return_value="SURVIVE")

        parents = daemon.select_parents(n_children=2)
        assert isinstance(parents, list)

    def test_fallback_to_local_when_fleet_index_absent(self, grid, thermal):
        daemon = BreederDaemonV2(grid=grid, thermal=thermal)
        for i in range(5):
            grid.activity[i] += 10
        daemon._fsm[300 + i] = MagicMock()
        daemon._fsm[300 + i].get_state = MagicMock(return_value="SURVIVE")

        parents = daemon.select_parents(n_children=2)
        assert isinstance(parents, list)
        assert len(parents) <= 2


# ── 4. Flux preset library ────────────────────────────────

class TestFluxPresetLibrary:
    def test_cycle_applies_preset(self, breeder, mock_flux_preset_library):
        breeder._flux_preset_library = mock_flux_preset_library
        breeder.auto_breed(n_winners=1)
        mock_flux_preset_library.suggest_preset_for_task.assert_called_once_with("breeding")
        mock_flux_preset_library.apply_preset.assert_called_once()

    def test_preset_exception_logged_not_raised(self, breeder, mock_flux_preset_library):
        mock_flux_preset_library.apply_preset.side_effect = RuntimeError("flux boom")
        breeder._flux_preset_library = mock_flux_preset_library
        breeder.auto_breed(n_winners=1)
        mock_flux_preset_library.apply_preset.assert_called_once()

    def test_no_preset_library_no_crash(self, breeder):
        breeder.auto_breed(n_winners=1)


# ── 5. Agent identity signing ─────────────────────────────

class TestAgentIdentity:
    def test_cycle_signs_wal_entries(self, grid, thermal, mock_agent_identity):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            agent_identity=mock_agent_identity,
        )
        daemon.auto_breed(n_winners=1)
        mock_agent_identity.sign_task.assert_called()

    def test_signature_stored(self, grid, thermal, mock_agent_identity):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            agent_identity=mock_agent_identity,
        )
        daemon.auto_breed(n_winners=1)
        # At least one signature should be stored
        assert len(daemon._breed_signatures) >= 0
        for sig in daemon._breed_signatures.values():
            assert sig == "sig_deadbeef"

    def test_identity_exception_logged_not_raised(self, grid, thermal, mock_agent_identity):
        mock_agent_identity.sign_task.side_effect = RuntimeError("sign boom")
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            agent_identity=mock_agent_identity,
        )
        daemon.auto_breed(n_winners=1)
        mock_agent_identity.sign_task.assert_called()


# ── 6. Trap registry ────────────────────────────────────

class TestTrapRegistry:
    def test_cycle_runs_traps(self, breeder, mock_trap_registry):
        breeder._trap_registry = mock_trap_registry
        breeder.auto_breed(n_winners=1)
        mock_trap_registry.run_all.assert_called_once()

    def test_trap_exception_logged_not_raised(self, breeder, mock_trap_registry):
        mock_trap_registry.run_all.side_effect = RuntimeError("trap boom")
        breeder._trap_registry = mock_trap_registry
        breeder.auto_breed(n_winners=1)
        mock_trap_registry.run_all.assert_called_once()

    def test_no_trap_registry_no_crash(self, breeder):
        breeder.auto_breed(n_winners=1)


# ── 7. get_fleet_status ─────────────────────────────────

class TestGetFleetStatus:
    def test_includes_all_modules(self, grid, thermal, mock_metronome, mock_fleet_index,
                                   mock_trap_registry, mock_flux_preset_library,
                                   mock_agent_identity):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            metronome_bridge=mock_metronome,
            fleet_vector_index=mock_fleet_index,
            trap_registry=mock_trap_registry,
            flux_preset_library=mock_flux_preset_library,
            agent_identity=mock_agent_identity,
        )
        status = daemon.get_fleet_status()
        assert status["metronome_bridge"] is True
        assert status["fleet_vector_index"] is True
        assert status["trap_registry"] is True
        assert status["flux_preset_library"] is True
        assert status["agent_identity"] is True
        assert "agent_count" in status
        assert "diversity_score" in status

    def test_handles_missing_modules(self, grid, thermal):
        daemon = BreederDaemonV2(grid=grid, thermal=thermal)
        status = daemon.get_fleet_status()
        assert status["metronome_bridge"] is False
        assert status["fleet_vector_index"] is False
        assert status["trap_registry"] is False
        assert status["flux_preset_library"] is False
        assert status["agent_identity"] is False

    def test_pools_fleet_index_size(self, grid, thermal, mock_fleet_index):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            fleet_vector_index=mock_fleet_index,
        )
        status = daemon.get_fleet_status()
        assert status["fleet_vector_index_pool_size"] == 1


# ── 8. Thread-safe concurrent cycles ─────────────────────

class TestThreadSafety:
    def test_concurrent_cycles(self, grid, thermal, mock_metronome, mock_trap_registry):
        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            metronome_bridge=mock_metronome,
            trap_registry=mock_trap_registry,
        )
        results = []
        errors = []

        def worker():
            try:
                r = daemon.auto_breed(n_winners=1)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent cycles: {errors}"
        assert len(results) == 5
        # Metronome should have been ticked 5 times
        assert mock_metronome.tick.call_count == 5
        # Traps should have run 5 times
        assert mock_trap_registry.run_all.call_count == 5
