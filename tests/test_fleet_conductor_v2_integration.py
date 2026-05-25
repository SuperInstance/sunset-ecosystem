"""Integration tests for FleetConductorV2 SDA pipelines + beat() extensions.

Covers the 3 new SDA pipelines (identity_monitor, mesh_diversity_monitor,
opcode_safety_monitor) and the 2 new beat() subsystems (breeder tick,
SSE dashboard publish).
"""

from __future__ import annotations

import sys
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from nexus.fleet_conductor_v2 import (
    ConductorConfig,
    FleetConductorV2,
    _IdentitySense,
    _IdentityDecide,
    _MeshDiversitySense,
    _MeshDiversityDecide,
    _OpcodeSafetySense,
    _OpcodeSafetyDecide,
)


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_integration_subsystems(monkeypatch):
    """Mock all heavy subsystem imports for integration tests."""
    # MetronomeBridge
    mock_nerve = MagicMock()
    mock_nerve.MetronomeBridge = MagicMock()
    mock_nerve.MetronomeBridge.return_value.tick.return_value = 1
    mock_nerve.MetronomeBridge.return_value.sync_with_peers.return_value = []
    mock_nerve.MetronomeBridge.return_value.maybe_correct_drift.return_value = (False, 120.0)
    mock_nerve.MetronomeBridge.return_value.compute_drift.return_value = 0.0
    mock_nerve.MetronomeBridge.return_value.peers = []
    monkeypatch.setitem(sys.modules, "nerve.distributed_metronome_bridge", mock_nerve)

    # FleetVectorIndex
    mock_swarm = MagicMock()
    mock_swarm.FleetVectorIndex = MagicMock()
    mock_table = MagicMock()
    mock_table.insert_signed.return_value = None
    mock_swarm.FleetVectorIndex.return_value.get_gen_table.return_value = mock_table
    mock_swarm.FleetVectorIndex.return_value.get_fleet_sync_payload.return_value = b"{}"
    mock_swarm.FleetVectorIndex.return_value.stats = {"total_entries": 42}
    monkeypatch.setitem(sys.modules, "swarm.mesh_vector_tables", mock_swarm)

    # TrapRegistry + ThermalTrap
    mock_fleet_trap = MagicMock()
    mock_fleet_trap.TrapRegistry = MagicMock()
    mock_fleet_trap.TrapRegistry.return_value.run_all.return_value = []
    mock_fleet_trap.TrapRegistry.return_value.get_status.return_value = {}
    mock_fleet_trap.ThermalTrap = MagicMock()
    monkeypatch.setitem(sys.modules, "fleet.operational_trap", mock_fleet_trap)

    # FluxPresetLibrary
    mock_sunset = MagicMock()
    mock_sunset.FluxPresetLibrary = MagicMock()
    monkeypatch.setitem(sys.modules, "sunset.flux_preset_library", mock_sunset)

    # AgentRegistry + AgentIdentity + AgentCard
    mock_logos = MagicMock()
    mock_logos.AgentRegistry = MagicMock()
    mock_logos.AgentRegistry.return_value.list_agents.return_value = ["agent-a", "agent-b"]
    mock_logos.AgentIdentity = MagicMock()
    mock_logos.AgentCard = MagicMock()
    mock_logos.AgentCard.from_dict = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "logos.a2a_identity", mock_logos)

    # GatewayPacing
    class MockGatewayPacing:
        def __init__(self):
            self._state = "OPEN"
        def can_dispatch(self):
            return (True, "OPEN")
        def record_timeout(self):
            pass
        def record_success(self):
            pass
        def record_failure(self):
            pass
        def get_status(self):
            return {"state": self._state}

    mock_pacing = MagicMock()
    mock_pacing.GatewayPacing = MockGatewayPacing
    monkeypatch.setitem(sys.modules, "fleet.gateway_pacing", mock_pacing)

    # SDALoop + related classes  — keep real dataclasses so helper tests work
    import fleet.sense_decide_act as real_sda
    mock_sda = MagicMock()
    mock_sda.SDALoop = MagicMock()
    mock_sda.SDALoop.return_value.tick.return_value = {}
    mock_sda.TrapSense = MagicMock()
    mock_sda.GatewayDispatchDecide = MagicMock()
    mock_sda.FluxPresetDecide = MagicMock()
    mock_sda.Policy = MagicMock()
    mock_sda.ActResult = real_sda.ActResult
    mock_sda.Observation = real_sda.Observation
    mock_sda.Decision = real_sda.Decision
    monkeypatch.setitem(sys.modules, "fleet.sense_decide_act", mock_sda)

    # DispatchRouter
    mock_dispatch = MagicMock()
    mock_dispatch.DispatchRouter = MagicMock()
    mock_dispatch.DispatchRouter.return_value.route.return_value = {"mode": "direct"}
    monkeypatch.setitem(sys.modules, "fleet.dispatch_router", mock_dispatch)

    # ThermalBudget + DeviceType
    mock_thermal = MagicMock()
    mock_thermal.ThermalBudget = MagicMock()
    mock_thermal.DeviceType = MagicMock()
    monkeypatch.setitem(sys.modules, "swarm.thermal", mock_thermal)

    # BreederDaemonV2
    mock_breeder = MagicMock()
    mock_breeder.BreederDaemonV2 = MagicMock()
    mock_breeder.BreederDaemonV2.return_value.step = MagicMock()
    mock_breeder.BreederDaemonV2.return_value.get_status.return_value = {"queue_depth": 3}
    monkeypatch.setitem(sys.modules, "swarm.breeder_daemon_v2", mock_breeder)

    # SSEStreamDashboard
    mock_sse = MagicMock()
    mock_sse.SSEStreamDashboard = MagicMock()
    mock_sse.SSEStreamDashboard.return_value.publish = MagicMock()
    monkeypatch.setitem(sys.modules, "fleet.sse_stream_dashboard", mock_sse)

    # OpcodeCapabilityIndex
    mock_opcode = MagicMock()
    mock_opcode.OpcodeCapabilityIndex = MagicMock()
    mock_opcode.OpcodeCapabilityIndex.return_value.get_summary.return_value = {"untested_count": 2}
    monkeypatch.setitem(sys.modules, "logos.opcode_capability_index", mock_opcode)

    # Decision journal (used by _LoggingAct)
    mock_journal = MagicMock()
    mock_journal.log_human_command = MagicMock()
    monkeypatch.setitem(sys.modules, "logos.decision_journal", mock_journal)


@pytest.fixture(autouse=True)
def mock_identity(monkeypatch):
    """Patch _get_identity to avoid ed25519 key generation hangs."""
    monkeypatch.setattr(FleetConductorV2, "_get_identity", lambda self: None)


@pytest.fixture
def full_config() -> ConductorConfig:
    """Config with ALL optional subsystems enabled."""
    return ConductorConfig(
        node_id="integration-node",
        bpm=120.0,
        peers=["peer-a"],
        enable_traps=True,
        enable_mesh=True,
        enable_metronome=True,
        enable_flux_presets=True,
        enable_identity=True,
        enable_gateway_pacing=True,
        enable_sda_loop=True,
        enable_breeding=True,
        enable_sse_dashboard=True,
        enable_opcode_index=True,
    )


@pytest.fixture
def full_conductor(full_config: ConductorConfig) -> FleetConductorV2:
    c = FleetConductorV2(config=full_config)
    yield c
    c.shutdown()


# ── 1. New SDA pipelines registered ───────────────────────


def test_identity_pipeline_registered(full_conductor: FleetConductorV2):
    full_conductor.start()
    sda = full_conductor._get_sda()
    assert sda is not None
    # The real SDALoop is mocked, but _wire_default_pipelines still calls
    # loop.register() on the mock. Verify register was called with the right name.
    register_calls = [call for call in sda.register.call_args_list
                      if call.kwargs.get("name") == "identity_monitor"]
    assert len(register_calls) == 1


def test_mesh_diversity_pipeline_registered(full_conductor: FleetConductorV2):
    full_conductor.start()
    sda = full_conductor._get_sda()
    register_calls = [call for call in sda.register.call_args_list
                      if call.kwargs.get("name") == "mesh_diversity_monitor"]
    assert len(register_calls) == 1


def test_opcode_safety_pipeline_registered(full_conductor: FleetConductorV2):
    full_conductor.start()
    sda = full_conductor._get_sda()
    register_calls = [call for call in sda.register.call_args_list
                      if call.kwargs.get("name") == "opcode_safety_monitor"]
    assert len(register_calls) == 1


def test_opcode_pipeline_not_registered_when_disabled():
    cfg = ConductorConfig(
        node_id="no-opcode-node",
        enable_opcode_index=False,
        enable_traps=False,
        enable_mesh=False,
        enable_metronome=False,
        enable_flux_presets=False,
        enable_identity=False,
        enable_gateway_pacing=False,
        enable_sda_loop=True,
    )
    c = FleetConductorV2(config=cfg)
    c.start()
    sda = c._get_sda()
    register_calls = [call for call in sda.register.call_args_list
                      if call.kwargs.get("name") == "opcode_safety_monitor"]
    assert len(register_calls) == 0


# ── 2. Pipeline count with all subsystems enabled ─────────


def test_all_six_pipelines_registered(full_conductor: FleetConductorV2):
    full_conductor.start()
    sda = full_conductor._get_sda()
    expected_names = {
        "trap_monitor",
        "gateway_monitor",
        "flux_monitor",
        "identity_monitor",
        "mesh_diversity_monitor",
        "opcode_safety_monitor",
    }
    actual_names = {call.kwargs.get("name") for call in sda.register.call_args_list}
    assert expected_names.issubset(actual_names)


# ── 3. beat() breeder tick ──────────────────────────────


def test_beat_ticks_breeder_when_enabled(full_conductor: FleetConductorV2):
    full_conductor.start()
    result = full_conductor.beat()
    assert "breeder" in result
    assert result["breeder"]["stepped"] is True
    breeder = full_conductor._get_breeder()
    breeder.step.assert_called_once()


def test_beat_no_breeder_when_disabled():
    cfg = ConductorConfig(
        node_id="no-breeder-node",
        enable_breeding=False,
        enable_traps=False,
        enable_mesh=False,
        enable_metronome=False,
        enable_flux_presets=False,
        enable_identity=False,
        enable_gateway_pacing=False,
        enable_sda_loop=False,
    )
    c = FleetConductorV2(config=cfg)
    c.start()
    result = c.beat()
    assert "breeder" not in result


def test_beat_breeder_failure_handled(full_conductor: FleetConductorV2):
    full_conductor.start()
    breeder = full_conductor._get_breeder()
    breeder.step.side_effect = RuntimeError("breeder boom")
    result = full_conductor.beat()
    assert "breeder" in result
    assert "error" in result["breeder"]
    assert "breeder boom" in result["breeder"]["error"]


# ── 4. beat() SSE dashboard publish ───────────────────────


def test_beat_publishes_sse_when_enabled(full_conductor: FleetConductorV2):
    full_conductor.start()
    result = full_conductor.beat()
    assert "sse_dashboard" in result
    assert result["sse_dashboard"]["published"] is True
    sse = full_conductor._get_sse_dashboard()
    sse.publish.assert_called_once()


def test_beat_no_sse_when_disabled():
    cfg = ConductorConfig(
        node_id="no-sse-node",
        enable_sse_dashboard=False,
        enable_breeding=False,
        enable_traps=False,
        enable_mesh=False,
        enable_metronome=False,
        enable_flux_presets=False,
        enable_identity=False,
        enable_gateway_pacing=False,
        enable_sda_loop=False,
    )
    c = FleetConductorV2(config=cfg)
    c.start()
    result = c.beat()
    assert "sse_dashboard" not in result


def test_beat_sse_failure_handled(full_conductor: FleetConductorV2):
    full_conductor.start()
    sse = full_conductor._get_sse_dashboard()
    sse.publish.side_effect = RuntimeError("sse boom")
    result = full_conductor.beat()
    assert "sse_dashboard" in result
    assert "error" in result["sse_dashboard"]
    assert "sse boom" in result["sse_dashboard"]["error"]


# ── 5. Beat ordering ─────────────────────────────────────


def test_beat_ordering_new_subsystems(full_conductor: FleetConductorV2):
    """Breeder and SSE appear after traps in the tick results."""
    full_conductor.start()
    result = full_conductor.beat()
    keys = list(result.keys())
    assert keys.index("traps") < keys.index("breeder")
    assert keys.index("breeder") < keys.index("sse_dashboard")


# ── 6. SDA helper class unit tests ────────────────────────


def test_identity_sense_healthy():
    registry = MagicMock()
    registry.list_agents.return_value = ["a", "b", "c"]
    sense = _IdentitySense(registry)
    obs = sense.observe()
    assert obs.source == "identity_sense"
    assert obs.metrics["agent_count"] == 3
    assert obs.metrics["healthy"] is True


def test_identity_decide_re_register_when_empty():
    decider = _IdentityDecide()
    obs = MagicMock()
    obs.metrics = {"agent_count": 0}
    decision = decider.decide(obs)
    assert decision.action_type == "re_register"
    assert decision.confidence == 0.7


def test_identity_decide_noop_when_healthy():
    decider = _IdentityDecide()
    obs = MagicMock()
    obs.metrics = {"agent_count": 5}
    decision = decider.decide(obs)
    assert decision.action_type == "noop"
    assert decision.confidence == 1.0


def test_mesh_diversity_decide_cross_breed_when_low():
    decider = _MeshDiversityDecide()
    obs = MagicMock()
    obs.metrics = {"diversity": 2}
    decision = decider.decide(obs)
    assert decision.action_type == "cross_breed"
    assert decision.confidence == 0.8


def test_mesh_diversity_decide_noop_when_adequate():
    decider = _MeshDiversityDecide()
    obs = MagicMock()
    obs.metrics = {"diversity": 10}
    decision = decider.decide(obs)
    assert decision.action_type == "noop"
    assert decision.confidence == 1.0


def test_opcode_safety_decide_flag_unsafe_when_untested():
    decider = _OpcodeSafetyDecide()
    obs = MagicMock()
    obs.metrics = {"untested_count": 5}
    decision = decider.decide(obs)
    assert decision.action_type == "flag_unsafe"
    assert decision.confidence == 0.8


def test_opcode_safety_decide_noop_when_safe():
    decider = _OpcodeSafetyDecide()
    obs = MagicMock()
    obs.metrics = {"untested_count": 0}
    decision = decider.decide(obs)
    assert decision.action_type == "noop"
    assert decision.confidence == 1.0


# ── 7. Conductor status snapshot in SSE publish ────────────


def test_sse_publish_receives_status_snapshot(full_conductor: FleetConductorV2):
    full_conductor.start()
    full_conductor.beat()
    sse = full_conductor._get_sse_dashboard()
    assert sse.publish.call_count == 1
    published_data = sse.publish.call_args[0][0]
    assert "node_id" in published_data
    assert published_data["node_id"] == "integration-node"
    assert "health" in published_data
    assert "subsystems" in published_data
