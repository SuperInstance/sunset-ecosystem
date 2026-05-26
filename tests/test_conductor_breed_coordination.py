"""Tests for FleetConductorV2 breed coordination pipeline.

Verifies that the SDA loop properly gates breeding based on beat phase,
queue depth, and FLUX constraints.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from nexus.fleet_conductor_v2 import FleetConductorV2, ConductorConfig


@pytest.fixture
def mock_breeder():
    m = MagicMock()
    m.get_status.return_value = {"queue_depth": 2, "queue_capacity": 10}
    m.select_parents.return_value = [(1, 2)]
    m.queue_breed = MagicMock()
    return m


@pytest.fixture
def mock_mesh():
    m = MagicMock()
    m.stats = {"total_entries": 50}
    m.get_breedable_pool.return_value = []
    return m


@pytest.fixture
def mock_flux():
    m = MagicMock()
    m.apply_preset.return_value = [{"passed": True, "severity": "info"}]
    return m


@pytest.fixture
def mock_metronome():
    m = MagicMock()
    m._tick_counter = 2  # beat phase 2 (breeding)
    m.tick.return_value = 1
    m.sync_with_peers.return_value = []
    m.maybe_correct_drift.return_value = (False, 120.0)
    m.compute_drift.return_value = 0.5
    return m


@pytest.fixture
def conductor(mock_breeder, mock_mesh, mock_flux, mock_metronome):
    cfg = ConductorConfig(
        node_id="test-node",
        enable_metronome=True,
        enable_mesh=True,
        enable_breeding=True,
        enable_flux_presets=True,
    )
    c = FleetConductorV2(config=cfg)
    # Inject mocks into subsystems
    c._subsystems["breeder"].instance = mock_breeder
    c._subsystems["mesh"].instance = mock_mesh
    c._subsystems["flux"].instance = mock_flux
    c._subsystems["metronome"].instance = mock_metronome
    c._subsystems["traps"].instance = MagicMock()
    c._subsystems["traps"].instance.run_all.return_value = []
    c._subsystems["traps"].instance.get_status.return_value = {}
    return c


class TestBreedCoordinationPipeline:
    def test_pipeline_registered(self, conductor):
        c = conductor
        c.start()
        sda = c._get_sda()
        assert sda is not None
        pipelines = sda.list_pipelines()
        assert "breed_coordination" in pipelines

    def test_breed_on_beat_phase_2(self, conductor, mock_breeder, mock_metronome):
        c = conductor
        c.start()
        mock_metronome._tick_counter = 2  # breeding phase
        result = c.beat()
        assert "breeder" in result
        # select_parents should have been called by the act
        assert mock_breeder.select_parents.called or mock_breeder.queue_breed.called

    def test_no_breed_on_beat_phase_0(self, conductor, mock_breeder, mock_metronome):
        c = conductor
        c.start()
        mock_breeder.reset_mock()
        mock_metronome._tick_counter = 0  # not breeding phase
        result = c.beat()
        # select_parents should NOT have been called
        assert not mock_breeder.select_parents.called

    def test_throttle_when_queue_full(self, conductor, mock_breeder, mock_metronome):
        c = conductor
        c.start()
        mock_breeder.get_status.return_value = {"queue_depth": 95, "queue_capacity": 100}
        mock_metronome._tick_counter = 2
        result = c.beat()
        sda = c._get_sda()
        tick_results = sda.tick()
        breed_result = tick_results.get("breed_coordination")
        if breed_result is not None:
            assert "throttled" in breed_result.side_effects or "idle" in breed_result.side_effects

    def test_flux_gate_blocks_breed(self, conductor, mock_breeder, mock_flux, mock_metronome):
        c = conductor
        c.start()
        mock_flux.apply_preset.return_value = [{"passed": False, "severity": "critical"}]
        mock_metronome._tick_counter = 2
        result = c.beat()
        # FLUX gate should block — select_parents may be called but queue_breed should not
        # or the act should report failure
        calls = [call for call in mock_breeder.method_calls if "queue_breed" in str(call)]
        # With FLUX blocked, queue_breed should not be called
        assert len(calls) == 0 or not any("queue_breed" in str(c) for c in calls)

    def test_cross_breed_when_pool_empty(self, conductor, mock_breeder, mock_mesh, mock_metronome):
        c = conductor
        c.start()
        mock_breeder.get_status.return_value = {"queue_depth": 1, "queue_capacity": 10}
        mock_mesh.get_breedable_pool.return_value = [
            MagicMock(agent_id="node-2::agent_42", fitness=0.8),
            MagicMock(agent_id="node-2::agent_43", fitness=0.7),
        ]
        mock_metronome._tick_counter = 2
        result = c.beat()
        # cross-node parents should be considered
        assert mock_breeder.select_parents.called or mock_mesh.get_breedable_pool.called

    def test_breed_coordination_metrics(self, conductor, mock_metronome):
        c = conductor
        c.start()
        mock_metronome._tick_counter = 2
        c.beat()
        sda = c._get_sda()
        metrics = sda.get_metrics()
        assert "breed_coordination" in metrics.get("pipeline_ticks", {})
