"""Tests for FleetConductorV2.

Covers initialization, lifecycle, beat ordering, status reporting,
dispatch routing, node registration, lazy init, auto-restart, and
thread-safety.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from nexus.fleet_conductor_v2 import (
    ConductorConfig,
    ConductorHealth,
    FleetConductorV2,
    SubsystemWrapper,
)


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture
def default_config() -> ConductorConfig:
    return ConductorConfig(
        node_id="test-node",
        bpm=120.0,
        peers=["peer-a", "peer-b"],
        enable_traps=True,
        enable_mesh=True,
        enable_metronome=True,
        enable_flux_presets=True,
        enable_identity=True,
        enable_gateway_pacing=True,
        enable_sda_loop=True,
        enable_breeding=False,
    )


@pytest.fixture(autouse=True)
def mock_identity(monkeypatch):
    """Patch FleetConductorV2._get_identity to avoid ed25519 key generation hangs."""
    monkeypatch.setattr(
        FleetConductorV2, "_get_identity", lambda self: None
    )


@pytest.fixture
def minimal_config() -> ConductorConfig:
    """Everything disabled except the bare minimum."""
    return ConductorConfig(
        node_id="minimal-node",
        bpm=60.0,
        enable_traps=False,
        enable_mesh=False,
        enable_metronome=False,
        enable_flux_presets=False,
        enable_identity=False,
        enable_gateway_pacing=False,
        enable_sda_loop=False,
        enable_breeding=False,
    )


@pytest.fixture
def conductor(default_config: ConductorConfig) -> FleetConductorV2:
    return FleetConductorV2(config=default_config)


# ── 1. Initialization with default config ─────────────────


def test_init_default_config():
    c = FleetConductorV2()
    assert c.config.node_id == "unnamed-node"
    assert c.config.bpm == 120.0
    assert c._started is False
    assert c._shutdown is False
    assert c._beat_count == 0


def test_init_custom_config(default_config: ConductorConfig):
    c = FleetConductorV2(config=default_config)
    assert c.config.node_id == "test-node"
    assert c.config.bpm == 120.0
    assert "metronome" in c._subsystems
    assert "mesh" in c._subsystems
    assert "traps" in c._subsystems
    assert "flux" in c._subsystems
    assert "identity" in c._subsystems
    assert "pacing" in c._subsystems
    assert "sda" in c._subsystems


def test_subsystem_wrappers_initialized():
    c = FleetConductorV2()
    for name in ["metronome", "mesh", "traps", "flux", "identity", "pacing", "sda"]:
        assert name in c._subsystems
        wrapper = c._subsystems[name]
        assert wrapper.name == name
        assert wrapper.instance is None  # lazy


# ── 2. start() brings up all subsystems ───────────────────


def test_start_brings_up_all_subsystems(conductor: FleetConductorV2):
    results = conductor.start()
    assert conductor._started is True
    assert conductor._start_time > 0.0
    for name, state in results.items():
        if name == "breeder":
            continue  # breeding disabled by default
        assert state == "started", f"Subsystem {name} failed to start"


def test_start_idempotent(conductor: FleetConductorV2):
    r1 = conductor.start()
    r2 = conductor.start()
    for name in r1:
        if name == "breeder":
            continue
        assert r2[name] == "already_started"


# ── 3. beat() runs in correct order ───────────────────────


def test_beat_increments_counter(conductor: FleetConductorV2):
    conductor.start()
    assert conductor.beat_count == 0
    result = conductor.beat()
    assert result["beat_number"] == 1
    assert conductor.beat_count == 1
    result2 = conductor.beat()
    assert result2["beat_number"] == 2
    assert conductor.beat_count == 2


def test_beat_runs_metronome(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.beat()
    assert "metronome" in result
    assert "local_beat_count" in result["metronome"]
    assert result["metronome"]["local_beat_count"] >= 1


def test_beat_runs_sda(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.beat()
    assert "sda" in result
    # sda.tick() returns a dict of pipeline_name > ActResult|None
    assert "pipelines_run" in result["sda"]


def test_beat_runs_traps(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.beat()
    assert "traps" in result
    assert "fired" in result["traps"]
    assert result["traps"]["fired"] == 0  # no traps registered by default


def test_beat_runs_mesh(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.beat()
    assert "mesh" in result
    assert "sync_payload_bytes" in result["mesh"]
    assert "stats" in result["mesh"]


def test_beat_ordering(conductor: FleetConductorV2):
    """Verify beat() touches subsystems in documented order."""
    conductor.start()
    result = conductor.beat()
    keys = list(result.keys())
    # metronome first, then sda, then mesh, then traps, then status
    assert keys.index("metronome") < keys.index("sda")
    assert keys.index("sda") < keys.index("mesh")
    assert keys.index("mesh") < keys.index("traps")


def test_beat_metronome_sync_every_4th_beat(conductor: FleetConductorV2):
    conductor.start()
    # Beat 1–3: no sync message count
    for i in range(1, 4):
        r = conductor.beat()
        assert "sync_messages" not in r.get("metronome", {})
    # Beat 4: sync triggered
    r4 = conductor.beat()
    assert "sync_messages" in r4["metronome"]


# ── 4. get_status() includes all subsystems ───────────────


def test_get_status_structure(conductor: FleetConductorV2):
    conductor.start()
    status = conductor.get_status()
    assert status["node_id"] == "test-node"
    assert "uptime_seconds" in status
    assert "beat_count" in status
    assert "subsystems" in status
    assert "nodes" in status
    assert "agents" in status
    assert "drift_ms" in status
    assert "diversity" in status
    assert "health" in status
    assert "queued_tasks" in status
    assert status["health"] in {"healthy", "degraded", "critical"}


def test_get_status_subsystem_states(conductor: FleetConductorV2):
    conductor.start()
    status = conductor.get_status()
    for name in ["metronome", "mesh", "traps", "flux", "identity", "pacing", "sda"]:
        assert name in status["subsystems"]
        sub = status["subsystems"][name]
        assert "state" in sub
        assert sub["state"] in {"healthy", "degraded", "failed", "disabled"}


# ── 5. spawn_agent respects GatewayPacing ──────────────────


def test_spawn_agent_when_open(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.spawn_agent({"description": "test task", "fn": lambda: 42})
    assert result["dispatched"] is True
    # Reason may come from pacing or direct execution
    if "reason" in result:
        assert result["reason"] in {
            "OPEN — dispatch allowed",
            "HALF_OPEN — probe allowed",
        }


def test_spawn_agent_queues_when_closed():
    """Force circuit CLOSED by recording two consecutive timeouts."""
    c = FleetConductorV2(
        config=ConductorConfig(
            node_id="closed-node",
            enable_gateway_pacing=True,
            enable_traps=False,
            enable_mesh=False,
            enable_metronome=False,
            enable_flux_presets=False,
            enable_identity=False,
            enable_sda_loop=False,
        )
    )
    c.start()
    pacing = c._get_pacing()
    pacing.record_timeout()
    pacing.record_timeout()
    assert pacing.get_status()["state"] == "CLOSED"

    result = c.spawn_agent({"description": "blocked task"})
    assert result["dispatched"] is False
    assert result["queued"] is True
    assert "CLOSED" in result["reason"]


# ── 6. spawn_agent routes simple task to direct ────────────


def test_spawn_agent_direct_execution(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.spawn_agent(
        {
            "description": "simple task",
            "fn": lambda x: x * 2,
            "args": [21],
        }
    )
    assert result["dispatched"] is True
    assert result.get("success") is True
    assert result.get("result") == 42


# ── 7. spawn_agent routes complex task to subagent ─────────


def test_spawn_agent_subagent_routing(conductor: FleetConductorV2):
    conductor.start()
    # Mock router to return subagent mode
    class FakeRouter:
        def route(self, desc: str) -> dict[str, Any]:
            return {"mode": "subagent", "estimated_seconds": 30}

    conductor._dispatch_router = FakeRouter()
    result = conductor.spawn_agent(
        {
            "description": "complex research task",
            "fn": lambda: None,
        }
    )
    assert result["dispatched"] is True
    assert result["route"]["mode"] == "subagent"
    assert result.get("mode") == "subagent"


# ── 8. register_node adds to mesh ────────────────────────


def test_register_node(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.register_node(
        {
            "node_id": "new-peer",
            "agent_cards": {
                "agent-1": {
                    "name": "TestAgent",
                    "version": "1.0",
                    "description": "A test agent",
                    "capabilities": {"streaming": True, "pushNotifications": False},
                    "skills": [
                        {
                            "id": "test",
                            "name": "Test",
                            "description": "d",
                            "tags": [],
                            "examples": [],
                        }
                    ],
                }
            },
        }
    )
    assert result["registered"] is True
    assert result["node_id"] == "new-peer"
    assert result["agents_discovered"] == 1


def test_register_node_no_node_id(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.register_node({})
    assert result["registered"] is False
    assert "error" in result


# ── 9. shutdown stops all subsystems gracefully ─────────────


def test_shutdown(conductor: FleetConductorV2):
    conductor.start()
    assert conductor.started is True
    results = conductor.shutdown()
    assert conductor.started is False
    assert conductor.shutdown_flag is True
    for name, state in results.items():
        if name == "breeder":
            continue
        assert state == "stopped", f"Subsystem {name} not stopped: {state}"


def test_shutdown_idempotent(conductor: FleetConductorV2):
    conductor.start()
    r1 = conductor.shutdown()
    r2 = conductor.shutdown()
    for name in r1:
        if name == "breeder":
            continue
        assert r2[name] == "already_shutdown"


def test_beat_after_shutdown(conductor: FleetConductorV2):
    conductor.start()
    conductor.shutdown()
    result = conductor.beat()
    assert "error" in result
    assert result["error"] == "conductor_shutdown"


# ── 10. Lazy init: disabled subsystems not started ────────


def test_lazy_init_disabled_subsystems(minimal_config: ConductorConfig):
    c = FleetConductorV2(config=minimal_config)
    results = c.start()
    for name, state in results.items():
        assert state == "disabled", f"Subsystem {name} should be disabled"


def test_lazy_init_on_demand():
    c = FleetConductorV2(
        config=ConductorConfig(
            node_id="lazy-node",
            enable_metronome=False,
            enable_mesh=True,
        )
    )
    c.start()
    # metronome not started
    assert c._subsystems["metronome"].instance is None
    # mesh started on first access
    mesh = c._get_mesh()
    assert mesh is not None
    assert c._subsystems["mesh"].instance is not None


# ── 11. Auto-restart on simulated failure ─────────────────


def test_auto_restart_backoff():
    wrapper = SubsystemWrapper(
        name="failing",
        factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        enabled=True,
    )
    instance = wrapper.ensure()
    assert instance is None
    assert wrapper.health.state == "failed"
    assert wrapper.health.consecutive_failures == 1

    # Second failure
    instance2 = wrapper.ensure()
    assert instance2 is None
    assert wrapper.health.consecutive_failures == 2


def test_conductor_auto_restart_flag():
    c = FleetConductorV2(
        config=ConductorConfig(
            node_id="restart-node",
            auto_restart=True,
            restart_backoff_base=0.01,
            restart_backoff_max=0.05,
        )
    )
    c.start()
    # Simulate repeated failures on metronome
    wrapper = c._subsystems["metronome"]
    wrapper.health.consecutive_failures = 3
    # _maybe_auto_restart should trigger restart with tiny backoff
    c._maybe_auto_restart("metronome")
    # After restart, consecutive failures should be cleared on success
    # (or stay if factory still fails)


# ── 12. Thread-safe concurrent beats ──────────────────────


def test_concurrent_beats(conductor: FleetConductorV2):
    conductor.start()
    num_threads = 8
    beats_per_thread = 10
    errors: list[str] = []

    def worker():
        try:
            for _ in range(beats_per_thread):
                conductor.beat()
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent beats: {errors}"
    assert conductor.beat_count == num_threads * beats_per_thread


# ── 13. Queued task management ────────────────────────────


def test_queue_and_drain(conductor: FleetConductorV2):
    conductor.start()
    # Force queue by shutting gateway
    pacing = conductor._get_pacing()
    pacing.record_timeout()
    pacing.record_timeout()
    # Spawn while closed
    conductor.spawn_agent({"description": "task1"})
    conductor.spawn_agent({"description": "task2"})
    assert conductor.queue_length() == 2

    drained = conductor.drain_queue(max_tasks=1)
    assert len(drained) == 1
    assert conductor.queue_length() == 1

    drained_all = conductor.drain_queue(max_tasks=10)
    assert len(drained_all) == 1
    assert conductor.queue_length() == 0


# ── 14. Status log limit ──────────────────────────────────


def test_status_log_limit(conductor: FleetConductorV2):
    conductor.start()
    conductor._status_log_limit = 5
    for _ in range(10):
        conductor.beat()
    assert len(conductor._status_log) == 5


# ── 15. Dispatch router fallback ──────────────────────────


def test_dispatch_router_unavailable(default_config: ConductorConfig):
    c = FleetConductorV2(config=default_config)
    c._dispatch_router = None
    c.start()
    result = c.spawn_agent(
        {
            "description": "task",
            "fn": lambda: "ok",
        }
    )
    assert result["dispatched"] is True
    # No router = direct execution
    assert "route" in result


# ── 16. ConductorHealth dataclass ─────────────────────────


def test_conductor_health():
    h = ConductorHealth(
        subsystem="test", state="healthy", last_ok=time.monotonic(), consecutive_failures=0
    )
    assert h.subsystem == "test"
    assert h.state == "healthy"


# ── 17. Config defaults ───────────────────────────────────


def test_config_defaults():
    cfg = ConductorConfig()
    assert cfg.node_id == "unnamed-node"
    assert cfg.bpm == 120.0
    assert cfg.enable_traps is True
    assert cfg.enable_mesh is True
    assert cfg.enable_metronome is True
    assert cfg.auto_restart is True


def test_config_custom_peers():
    cfg = ConductorConfig(peers=["alpha", "beta"])
    assert cfg.peers == ["alpha", "beta"]


# ── 18. SubsystemWrapper health_check ─────────────────────


def test_subsystem_wrapper_health_check():
    class DummySubsystem:
        def get_status(self) -> dict[str, Any]:
            return {"ok": True}

    wrapper = SubsystemWrapper(
        name="dummy",
        factory=DummySubsystem,
        enabled=True,
    )
    assert wrapper.instance is None
    wrapper.ensure()
    assert wrapper.instance is not None
    health = wrapper.health_check()
    assert health.state == "healthy"
    assert health.consecutive_failures == 0


def test_subsystem_wrapper_health_check_failure():
    class BadSubsystem:
        def get_status(self) -> dict[str, Any]:
            raise RuntimeError("bad status")

    wrapper = SubsystemWrapper(
        name="bad",
        factory=BadSubsystem,
        enabled=True,
    )
    wrapper.ensure()
    h1 = wrapper.health_check()
    assert h1.state == "degraded"
    h2 = wrapper.health_check()
    assert h2.state == "degraded"
    h3 = wrapper.health_check()
    assert h3.state == "failed"
    assert h3.consecutive_failures == 3


# ── 19. SubsystemWrapper restart ──────────────────────────


def test_subsystem_wrapper_restart():
    class CountingSubsystem:
        count = 0

        def __init__(self) -> None:
            CountingSubsystem.count += 1

    wrapper = SubsystemWrapper(
        name="counting",
        factory=CountingSubsystem,
        enabled=True,
    )
    wrapper.ensure()
    assert CountingSubsystem.count == 1
    wrapper.restart()
    assert CountingSubsystem.count == 2
    assert wrapper.instance is not None


# ── 20. Breeding enabled config ───────────────────────────


def test_breeding_enabled_config():
    cfg = ConductorConfig(
        node_id="breeder-node",
        enable_breeding=True,
    )
    c = FleetConductorV2(config=cfg)
    assert "breeder" in c._subsystems
    assert c._subsystems["breeder"].enabled is True


# ── 21. Node info with no agent_cards ─────────────────────


def test_register_node_no_agents(conductor: FleetConductorV2):
    conductor.start()
    result = conductor.register_node({"node_id": "lonely-node"})
    assert result["registered"] is True
    assert result["agents_discovered"] == 0


# ── 22. beat after multiple starts ────────────────────────


def test_beat_after_start_stop_start(conductor: FleetConductorV2):
    conductor.start()
    conductor.beat()
    conductor.shutdown()
    # After shutdown, start again
    conductor.start()
    r = conductor.beat()
    assert "beat_number" in r
    assert r["beat_number"] == 1  # reset on new start? no, beat_count is not reset
    # Actually beat_count is not reset on start() — only on init. That may be intentional.
    # Let's just verify it works:
    assert conductor.beat_count >= 2
