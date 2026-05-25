"""Fleet Conductor V2 — Central nervous system for the Cocapn Fleet.

Orchestrates all subsystems built this session:
  • MetronomeBridge — cross-node beat sync
  • FleetVectorIndex — federated mesh tables
  • TrapRegistry — fleet health monitoring
  • FluxPresetLibrary — FLUX constraint presets
  • AgentRegistry — A2A identity + discovery
  • GatewayPacing — circuit breaker for dispatch
  • SDALoop — Sense→Decide→Act pipelines

Replaces ``nexus/fleet_conductor.py`` as the default conductor
but keeps it available for backward compatibility.

Reference: docs/FLEET_CONDUCTOR_V2.md
"""

from __future__ import annotations

__all__ = [
    "FleetConductorV2",
    "ConductorConfig",
    "ConductorHealth",
    "SubsystemWrapper",
]

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── data structures ───────────────────────────────────────────

@dataclass(frozen=True)
class ConductorConfig:
    """Configuration for FleetConductorV2.

    All enable_* flags control lazy initialization — if False, the
    subsystem is never instantiated and is skipped on start().
    """

    node_id: str = "unnamed-node"
    bpm: float = 120.0
    peers: List[str] = field(default_factory=list)
    thermal_limits: Dict[str, float] = field(default_factory=dict)
    flux_preset_name: str = "FleetHealth"
    enable_traps: bool = True
    enable_mesh: bool = True
    enable_metronome: bool = True
    enable_flux_presets: bool = True
    enable_identity: bool = True
    enable_gateway_pacing: bool = True
    enable_sda_loop: bool = True
    enable_breeding: bool = False
    enable_sse_dashboard: bool = False
    enable_metronome_gossip: bool = False
    enable_opcode_index: bool = False
    enable_hebbian_mesh: bool = False
    enable_decision_journal: bool = False
    sda_interval_ms: float = 1000.0
    max_drift_ms: float = 10.0
    auto_restart: bool = True
    restart_backoff_base: float = 1.0
    restart_backoff_max: float = 60.0
    decision_journal_path: str = ""


@dataclass
class ConductorHealth:
    """Health snapshot for a single subsystem."""

    subsystem: str
    state: str  # "healthy", "degraded", "failed", "disabled"
    last_ok: float
    consecutive_failures: int
    last_error: str = ""


@dataclass
class SubsystemWrapper:
    """Wrapper around a subsystem instance with health tracking."""

    name: str
    factory: Callable[[], Any]
    instance: Any | None = None
    enabled: bool = True
    health: ConductorHealth = field(
        default_factory=lambda: ConductorHealth(
            subsystem="", state="disabled", last_ok=0.0, consecutive_failures=0
        )
    )
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def ensure(self) -> Any | None:
        """Lazy-init the subsystem if enabled and not yet created."""
        with self._lock:
            if not self.enabled:
                return None
            if self.instance is not None:
                return self.instance
            try:
                self.instance = self.factory()
                self.health = ConductorHealth(
                    subsystem=self.name,
                    state="healthy",
                    last_ok=time.monotonic(),
                    consecutive_failures=0,
                )
                logger.info("Subsystem '%s' initialized", self.name)
                return self.instance
            except Exception as exc:
                self.health = ConductorHealth(
                    subsystem=self.name,
                    state="failed",
                    last_ok=0.0,
                    consecutive_failures=self.health.consecutive_failures + 1,
                    last_error=str(exc),
                )
                logger.error("Subsystem '%s' init failed: %s", self.name, exc)
                return None

    def health_check(self) -> ConductorHealth:
        """Ping the subsystem; mark degraded on exception."""
        with self._lock:
            if not self.enabled or self.instance is None:
                self.health.state = "disabled"
                return self.health
            try:
                # Most subsystems expose get_status()
                if hasattr(self.instance, "get_status"):
                    self.instance.get_status()
                self.health.last_ok = time.monotonic()
                self.health.consecutive_failures = 0
                self.health.state = "healthy"
            except Exception as exc:
                self.health.consecutive_failures += 1
                self.health.last_error = str(exc)
                self.health.state = (
                    "failed"
                    if self.health.consecutive_failures >= 3
                    else "degraded"
                )
                logger.warning(
                    "Health check failed for '%s' (%d consecutive): %s",
                    self.name,
                    self.health.consecutive_failures,
                    exc,
                )
            return self.health

    def destroy(self) -> None:
        """Gracefully tear down the subsystem."""
        with self._lock:
            if self.instance is None:
                return
            try:
                if hasattr(self.instance, "shutdown"):
                    self.instance.shutdown()
                elif hasattr(self.instance, "close"):
                    self.instance.close()
                elif hasattr(self.instance, "stop"):
                    self.instance.stop()
            except Exception as exc:
                logger.warning("Error tearing down '%s': %s", self.name, exc)
            finally:
                self.instance = None
                self.health.state = "disabled"

    def restart(self) -> Any | None:
        """Destroy and re-initialize."""
        with self._lock:
            self.destroy()
        return self.ensure()


# ── FleetConductorV2 ──────────────────────────────────────────

class FleetConductorV2:
    """Fleet central nervous system.

    Lifecycle
    ---------
    1. ``__init__(config)`` — prepares subsystem wrappers (no I/O).
    2. ``start()`` — lazy-initializes enabled subsystems.
    3. ``beat()`` — one conductor tick:
       a. tick metronome
       b. run SDA pipelines
       c. sync mesh tables with peers
       d. run operational traps
       e. log fleet status
    4. ``get_status()`` — full fleet snapshot.
    5. ``spawn_agent(task_spec)`` — dispatch with pacing + routing.
    6. ``register_node(node_info)`` — add peer to mesh.
    7. ``shutdown()`` — graceful stop of all subsystems.
    """

    def __init__(self, config: ConductorConfig | None = None) -> None:
        self.config = config or ConductorConfig()
        self._node_id = self.config.node_id

        # Internal state
        self._started: bool = False
        self._shutdown: bool = False
        self._beat_count: int = 0
        self._start_time: float = 0.0
        self._status_log: list[dict[str, Any]] = []
        self._status_log_limit: int = 100
        self._queued_tasks: list[dict[str, Any]] = []
        self._queued_tasks_limit: int = 500
        self._lock = threading.RLock()

        # Subsystem wrappers (lazy init)
        self._subsystems: Dict[str, SubsystemWrapper] = {}
        self._build_subsystems()

        # Dispatch router (lightweight, created eagerly)
        self._dispatch_router: Any | None = None
        self._init_dispatch_router()

    # ── subsystem wiring ────────────────────────────────────

    def _build_subsystems(self) -> None:
        cfg = self.config

        # 1. MetronomeBridge
        def _make_metronome() -> Any:
            from nerve.distributed_metronome_bridge import MetronomeBridge

            return MetronomeBridge(
                local_bpm=cfg.bpm,
                node_id=cfg.node_id,
                peers=list(cfg.peers),
                identity=self._get_identity(),
                drift_threshold_ms=cfg.max_drift_ms,
            )

        self._subsystems["metronome"] = SubsystemWrapper(
            name="metronome",
            factory=_make_metronome,
            enabled=cfg.enable_metronome,
        )

        # 2. FleetVectorIndex (mesh tables)
        def _make_mesh() -> Any:
            from swarm.mesh_vector_tables import FleetVectorIndex

            return FleetVectorIndex(
                node_id=cfg.node_id,
                identity=self._get_identity(),
            )

        self._subsystems["mesh"] = SubsystemWrapper(
            name="mesh",
            factory=_make_mesh,
            enabled=cfg.enable_mesh,
        )

        # 3. TrapRegistry
        def _make_traps() -> Any:
            from fleet.operational_trap import TrapRegistry

            registry = TrapRegistry()
            # Wire thermal limits if provided
            if cfg.thermal_limits:
                from fleet.operational_trap import ThermalTrap
                from swarm.thermal import ThermalBudget, DeviceType

                budget = ThermalBudget()
                for dev_str, limit in cfg.thermal_limits.items():
                    dt = DeviceType(dev_str)
                    budget.allocate(dt, max_agents=int(limit))
                registry.register(ThermalTrap(budget=budget))
            return registry

        self._subsystems["traps"] = SubsystemWrapper(
            name="traps",
            factory=_make_traps,
            enabled=cfg.enable_traps,
        )

        # 4. FluxPresetLibrary
        def _make_flux() -> Any:
            from sunset.flux_preset_library import FluxPresetLibrary

            return FluxPresetLibrary()

        self._subsystems["flux"] = SubsystemWrapper(
            name="flux",
            factory=_make_flux,
            enabled=cfg.enable_flux_presets,
        )

        # 5. AgentRegistry
        def _make_registry() -> Any:
            from logos.a2a_identity import AgentRegistry

            return AgentRegistry(identity=self._get_identity())

        self._subsystems["identity"] = SubsystemWrapper(
            name="identity",
            factory=_make_registry,
            enabled=cfg.enable_identity,
        )

        # 6. GatewayPacing
        def _make_pacing() -> Any:
            from fleet.gateway_pacing import GatewayPacing

            return GatewayPacing()

        self._subsystems["pacing"] = SubsystemWrapper(
            name="pacing",
            factory=_make_pacing,
            enabled=cfg.enable_gateway_pacing,
        )

        # 7. SDALoop
        def _make_sda() -> Any:
            from fleet.sense_decide_act import SDALoop

            loop = SDALoop()
            # Auto-wire pipelines if subsystems available
            self._wire_default_pipelines(loop)
            return loop

        self._subsystems["sda"] = SubsystemWrapper(
            name="sda",
            factory=_make_sda,
            enabled=cfg.enable_sda_loop,
        )

        # 8. BreederDaemonV2 (optional)
        if cfg.enable_breeding:
            def _make_breeder() -> Any:
                from swarm.breeder_daemon_v2 import BreederDaemonV2

                return BreederDaemonV2(
                    node_id=cfg.node_id,
                    vector_index=self._get_mesh(),
                    flux_preset=self.config.flux_preset_name,
                )

            self._subsystems["breeder"] = SubsystemWrapper(
                name="breeder",
                factory=_make_breeder,
                enabled=True,
            )

        # 9. SSEStreamDashboard (optional)
        if cfg.enable_sse_dashboard:
            def _make_sse() -> Any:
                from fleet.sse_stream_dashboard import SSEStreamDashboard

                return SSEStreamDashboard()

            self._subsystems["sse_dashboard"] = SubsystemWrapper(
                name="sse_dashboard",
                factory=_make_sse,
                enabled=True,
            )

        # 10. MetronomeGossipBridge (optional)
        if cfg.enable_metronome_gossip:
            def _make_gossip() -> Any:
                from nerve.metronome_mesh_bridge import MetronomeGossipBridge

                return MetronomeGossipBridge(node_id=cfg.node_id, peers=list(cfg.peers))

            self._subsystems["metronome_gossip"] = SubsystemWrapper(
                name="metronome_gossip",
                factory=_make_gossip,
                enabled=True,
            )

        # 11. OpcodeCapabilityIndex (optional)
        if cfg.enable_opcode_index:
            def _make_opcode() -> Any:
                from logos.opcode_capability_index import OpcodeCapabilityIndex

                return OpcodeCapabilityIndex()

            self._subsystems["opcode_index"] = SubsystemWrapper(
                name="opcode_index",
                factory=_make_opcode,
                enabled=True,
            )

        # 12. HebbianMeshLayer (optional)
        if cfg.enable_hebbian_mesh:
            def _make_hebbian() -> Any:
                from swarm.hebbian_mesh import HebbianMeshLayer

                return HebbianMeshLayer(node_id=cfg.node_id)

            self._subsystems["hebbian_mesh"] = SubsystemWrapper(
                name="hebbian_mesh",
                factory=_make_hebbian,
                enabled=True,
            )

    def _init_dispatch_router(self) -> None:
        """Create a lightweight DispatchRouter if available."""
        try:
            from fleet.dispatch_router import DispatchRouter

            self._dispatch_router = DispatchRouter()
        except Exception as exc:
            logger.debug("DispatchRouter not available: %s", exc)
            self._dispatch_router = None

    def _wire_default_pipelines(self, loop: Any) -> None:
        """Wire built-in SDA pipelines using available subsystems."""
        from fleet.sense_decide_act import (
            TrapSense,
            GatewayDispatchDecide,
            FluxPresetDecide,
            Policy,
        )

        # Trap pipeline
        traps = self._get_traps()
        if traps is not None:
            policy = Policy()
            policy.add_rule(
                lambda obs: obs.metrics.get("critical", 0) > 0,
                "escalate",
                1.0,
                "Critical trap fired — escalate immediately",
            )
            policy.add_rule(
                lambda obs: obs.metrics.get("warning", 0) > 0,
                "warn",
                0.8,
                "Warning trap fired — log and monitor",
            )
            loop.register(
                sense=TrapSense(traps),
                decide=policy,
                act=_NoopAct(),
                name="trap_monitor",
                interval_ms=5000.0,
            )

        # Gateway pacing pipeline
        pacing = self._get_pacing()
        if pacing is not None:
            loop.register(
                sense=_GatewaySense(pacing),
                decide=GatewayDispatchDecide(pacing),
                act=_NoopAct(),
                name="gateway_monitor",
                interval_ms=3000.0,
            )

        # FLUX preset pipeline
        flux = self._get_flux()
        if flux is not None:
            loop.register(
                sense=_FluxSense(),
                decide=FluxPresetDecide(flux, preset_name=self.config.flux_preset_name),
                act=_NoopAct(),
                name="flux_monitor",
                interval_ms=10000.0,
            )

        # 4. Identity monitor pipeline
        identity = self._get_identity_registry()
        if identity is not None:
            loop.register(
                sense=_IdentitySense(identity),
                decide=_IdentityDecide(),
                act=_NoopAct(),
                name="identity_monitor",
                interval_ms=5000.0,
            )

        # 5. Mesh diversity monitor pipeline
        mesh_diversity = self._get_mesh()
        if mesh_diversity is not None:
            loop.register(
                sense=_MeshDiversitySense(mesh_diversity),
                decide=_MeshDiversityDecide(),
                act=_NoopAct(),
                name="mesh_diversity_monitor",
                interval_ms=8000.0,
            )

        # 6. Opcode safety monitor pipeline
        opcode_index = self._get_opcode_index()
        if opcode_index is not None:
            loop.register(
                sense=_OpcodeSafetySense(opcode_index),
                decide=_OpcodeSafetyDecide(),
                act=_NoopAct(),
                name="opcode_safety_monitor",
                interval_ms=10000.0,
            )

    def _get_subsystem(self, name: str) -> Any | None:
        wrapper = self._subsystems.get(name)
        if wrapper is None:
            return None
        return wrapper.ensure()

    def _get_metronome(self) -> Any | None:
        return self._get_subsystem("metronome")

    def _get_mesh(self) -> Any | None:
        return self._get_subsystem("mesh")

    def _get_traps(self) -> Any | None:
        return self._get_subsystem("traps")

    def _get_flux(self) -> Any | None:
        return self._get_subsystem("flux")

    def _get_identity_registry(self) -> Any | None:
        return self._get_subsystem("identity")

    def _get_pacing(self) -> Any | None:
        return self._get_subsystem("pacing")

    def _get_sda(self) -> Any | None:
        return self._get_subsystem("sda")

    def _get_breeder(self) -> Any | None:
        return self._get_subsystem("breeder")

    def _get_sse_dashboard(self) -> Any | None:
        return self._get_subsystem("sse_dashboard")

    def _get_metronome_gossip(self) -> Any | None:
        return self._get_subsystem("metronome_gossip")

    def _get_opcode_index(self) -> Any | None:
        return self._get_subsystem("opcode_index")

    def _get_hebbian_mesh(self) -> Any | None:
        return self._get_subsystem("hebbian_mesh")

    def _get_identity(self) -> Any | None:
        """Return an AgentIdentity if available, else None."""
        try:
            from logos.a2a_identity import AgentIdentity

            return AgentIdentity(agent_id=self._node_id)
        except Exception as exc:
            logger.debug("AgentIdentity unavailable: %s", exc)
            return None

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> dict[str, str]:
        """Bring up all enabled subsystems.

        Returns a dict mapping subsystem_name → "started" | "disabled" | "failed".
        """
        with self._lock:
            if self._started:
                return {name: "already_started" for name in self._subsystems}
            self._started = True
            self._start_time = time.monotonic()
            self._shutdown = False

        results: dict[str, str] = {}
        for name, wrapper in self._subsystems.items():
            if not wrapper.enabled:
                results[name] = "disabled"
                continue
            instance = wrapper.ensure()
            results[name] = "started" if instance is not None else "failed"

        logger.info("FleetConductorV2 started: %s", results)
        return results

    def beat(self) -> dict[str, Any]:
        """One conductor tick.

        Order:
        1. Tick metronome
        2. Run SDA pipelines
        3. Sync mesh tables
        4. Run operational traps
        5. Log fleet status
        """
        with self._lock:
            if self._shutdown:
                return {"error": "conductor_shutdown"}
            self._beat_count += 1
            beat_number = self._beat_count

        tick_results: dict[str, Any] = {"beat_number": beat_number}

        # 1. Metronome tick
        metronome = self._get_metronome()
        if metronome is not None:
            try:
                local_count = metronome.tick()
                tick_results["metronome"] = {"local_beat_count": local_count}
                # Sync with peers every 4 beats (quarter-note)
                if beat_number % 4 == 0:
                    sync_msgs = metronome.sync_with_peers()
                    tick_results["metronome"]["sync_messages"] = len(sync_msgs)
                    # Auto-correct drift
                    did_adjust, new_bpm = metronome.maybe_correct_drift()
                    tick_results["metronome"]["drift_corrected"] = did_adjust
                    tick_results["metronome"]["bpm"] = round(new_bpm, 3)
            except Exception as exc:
                logger.warning("Metronome tick failed: %s", exc)
                tick_results["metronome"] = {"error": str(exc)}
                self._maybe_auto_restart("metronome")

        # 2. SDA loop
        sda = self._get_sda()
        if sda is not None:
            try:
                sda_results = sda.tick()
                tick_results["sda"] = {
                    "pipelines_run": len(sda_results),
                    "results": sda_results,
                }
            except Exception as exc:
                logger.warning("SDA tick failed: %s", exc)
                tick_results["sda"] = {"error": str(exc)}
                self._maybe_auto_restart("sda")

        # 3. Sync mesh tables
        mesh = self._get_mesh()
        if mesh is not None:
            try:
                payload = mesh.get_fleet_sync_payload()
                tick_results["mesh"] = {
                    "sync_payload_bytes": len(payload),
                    "stats": mesh.stats,
                }
            except Exception as exc:
                logger.warning("Mesh sync failed: %s", exc)
                tick_results["mesh"] = {"error": str(exc)}
                self._maybe_auto_restart("mesh")

        # 4. Operational traps
        traps = self._get_traps()
        if traps is not None:
            try:
                fired = traps.run_all()
                tick_results["traps"] = {
                    "fired": len(fired),
                    "conditions": [r.condition for r in fired],
                    "status": traps.get_status(),
                }
            except Exception as exc:
                logger.warning("Trap run failed: %s", exc)
                tick_results["traps"] = {"error": str(exc)}
                self._maybe_auto_restart("traps")

        # 5. Log fleet status
        status = self._snapshot_status(tick_results)
        with self._lock:
            self._status_log.append(status)
            if len(self._status_log) > self._status_log_limit:
                self._status_log = self._status_log[-self._status_log_limit :]

        # 6. Tick breeder if enabled
        breeder = self._get_breeder()
        if breeder is not None:
            try:
                breeder.step()
                tick_results["breeder"] = {"stepped": True}
            except Exception as exc:
                logger.warning("Breeder step failed: %s", exc)
                tick_results["breeder"] = {"error": str(exc)}
                self._maybe_auto_restart("breeder")

        # 7. Publish conductor status to SSEStreamDashboard
        sse = self._get_sse_dashboard()
        if sse is not None and hasattr(sse, "publish"):
            try:
                status_snapshot = self.get_status()
                sse.publish(status_snapshot)
                tick_results["sse_dashboard"] = {"published": True}
            except Exception as exc:
                logger.warning("SSE dashboard publish failed: %s", exc)
                tick_results["sse_dashboard"] = {"error": str(exc)}

        tick_results["status_logged_at"] = status.get("timestamp")
        return tick_results

    def get_status(self) -> dict[str, Any]:
        """Return full fleet status snapshot.

        Keys
        ----
        node_id : str
        uptime_seconds : float
        beat_count : int
        subsystems : dict[str, dict]
            Per-subsystem health + status.
        nodes : list[str]
            Known peer node IDs.
        agents : list[str]
            Known agent IDs from identity registry.
        drift_ms : float | None
            Current metronome drift.
        diversity : float | None
            Fleet-wide diversity score from mesh.
        health : str
            Overall health: "healthy", "degraded", "critical".
        queued_tasks : int
        """
        with self._lock:
            uptime = time.monotonic() - self._start_time if self._started else 0.0
            beat_count = self._beat_count
            queued = len(self._queued_tasks)

        # Subsystem health
        subsystems: dict[str, dict] = {}
        critical_count = 0
        degraded_count = 0
        for name, wrapper in self._subsystems.items():
            health = wrapper.health_check()
            subsystems[name] = {
                "state": health.state,
                "consecutive_failures": health.consecutive_failures,
                "last_error": health.last_error,
            }
            if health.state == "failed":
                critical_count += 1
            elif health.state == "degraded":
                degraded_count += 1

            # Add subsystem-specific status if available
            if wrapper.instance is not None and hasattr(wrapper.instance, "get_status"):
                try:
                    subsystems[name]["detail"] = wrapper.instance.get_status()
                except Exception as exc:
                    subsystems[name]["detail_error"] = str(exc)

        # Metronome drift
        drift_ms: float | None = None
        metronome = self._get_metronome()
        if metronome is not None:
            try:
                drift_ms = metronome.compute_drift()
            except Exception:
                pass

        # Diversity
        diversity: float | None = None
        mesh = self._get_mesh()
        if mesh is not None:
            try:
                diversity = mesh.stats.get("total_entries", 0)
            except Exception:
                pass

        # Nodes / agents
        nodes: list[str] = list(self.config.peers)
        agents: list[str] = []
        registry = self._get_identity_registry()
        if registry is not None:
            try:
                agents = registry.list_agents()
            except Exception:
                pass

        # Overall health
        if critical_count > 0:
            overall_health = "critical"
        elif degraded_count > 0:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        return {
            "node_id": self._node_id,
            "uptime_seconds": round(uptime, 1),
            "beat_count": beat_count,
            "subsystems": subsystems,
            "nodes": nodes,
            "agents": agents,
            "drift_ms": round(drift_ms, 3) if drift_ms is not None else None,
            "diversity": diversity,
            "health": overall_health,
            "queued_tasks": queued,
            "timestamp": time.time(),
        }

    def spawn_agent(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an agent task through GatewayPacing + optional router.

        Returns a dict with keys:
        - dispatched: bool
        - reason: str
        - route: dict (if routed)
        - queued: bool (if pacing blocked)
        """
        pacing = self._get_pacing()
        if pacing is not None:
            ok, reason = pacing.can_dispatch()
            if not ok:
                # Queue the task for later
                with self._lock:
                    if len(self._queued_tasks) < self._queued_tasks_limit:
                        self._queued_tasks.append(
                            {
                                "task_spec": task_spec,
                                "queued_at": time.time(),
                                "reason": reason,
                            }
                        )
                return {
                    "dispatched": False,
                    "reason": reason,
                    "queued": True,
                }

        # Routing decision
        route: dict[str, Any] = {"mode": "direct"}
        if self._dispatch_router is not None:
            try:
                desc = task_spec.get("description", "default task")
                route = self._dispatch_router.route(desc)
            except Exception as exc:
                route = {"mode": "direct", "router_error": str(exc)}

        mode = route.get("mode", "direct")

        # Execute or delegate
        if mode == "direct":
            result = self._execute_direct(task_spec)
        elif mode == "subagent":
            result = self._delegate_subagent(task_spec)
        elif mode == "queue":
            with self._lock:
                if len(self._queued_tasks) < self._queued_tasks_limit:
                    self._queued_tasks.append(
                        {
                            "task_spec": task_spec,
                            "queued_at": time.time(),
                            "reason": "router_queue",
                        }
                    )
            result = {"dispatched": False, "reason": "router_queue", "queued": True}
        else:
            result = {"dispatched": False, "reason": f"unknown_mode:{mode}", "queued": False}

        # Record pacing outcome
        if pacing is not None:
            if result.get("dispatched") or result.get("success"):
                pacing.record_success()
            elif "timeout" in result.get("reason", "").lower():
                pacing.record_timeout()
            else:
                pacing.record_failure()

        result["route"] = route
        return result

    def register_node(self, node_info: dict[str, Any]) -> dict[str, Any]:
        """Add a peer node to the mesh and discover its agents.

        Returns a dict with:
        - node_id: str
        - registered: bool
        - agents_discovered: int
        - mesh_stats: dict
        """
        node_id = node_info.get("node_id", "")
        if not node_id:
            return {"node_id": "", "registered": False, "error": "missing node_id"}

        # Add to metronome peers
        metronome = self._get_metronome()
        if metronome is not None:
            try:
                if node_id not in metronome.peers:
                    metronome.peers.append(node_id)
            except Exception:
                pass

        # Register in mesh
        mesh = self._get_mesh()
        if mesh is not None:
            try:
                # Insert a placeholder entry for the node
                import numpy as np

                placeholder = mesh.get_gen_table(0).insert_signed(
                    agent_id=f"{node_id}::node_placeholder",
                    vector=np.zeros(64, dtype=np.float32),
                    node_id=node_id,
                    generation=0,
                    fitness=0.5,
                )
            except Exception as exc:
                logger.debug("Mesh node registration placeholder failed: %s", exc)

        # Discover agents via registry
        agents_discovered = 0
        registry = self._get_identity_registry()
        if registry is not None:
            try:
                for agent_id, card in node_info.get("agent_cards", {}).items():
                    from logos.a2a_identity import AgentCard

                    if isinstance(card, dict):
                        card = AgentCard.from_dict(card)
                    registry.register(agent_id, card)
                    agents_discovered += 1
            except Exception as exc:
                logger.debug("Agent discovery failed for %s: %s", node_id, exc)

        mesh_stats = {}
        if mesh is not None:
            try:
                mesh_stats = mesh.stats
            except Exception:
                pass

        return {
            "node_id": node_id,
            "registered": True,
            "agents_discovered": agents_discovered,
            "mesh_stats": mesh_stats,
        }

    def shutdown(self) -> dict[str, str]:
        """Gracefully stop all subsystems.

        Returns a dict mapping subsystem_name → "stopped" | "not_started".
        """
        with self._lock:
            if self._shutdown:
                return {name: "already_shutdown" for name in self._subsystems}
            self._shutdown = True
            self._started = False

        results: dict[str, str] = {}
        for name, wrapper in self._subsystems.items():
            if wrapper.instance is None:
                results[name] = "not_started"
                continue
            wrapper.destroy()
            results[name] = "stopped"

        logger.info("FleetConductorV2 shutdown: %s", results)
        return results

    # ── internal helpers ────────────────────────────────────

    def _maybe_auto_restart(self, name: str) -> None:
        """Auto-restart a subsystem if configured and health is poor."""
        if not self.config.auto_restart:
            return
        wrapper = self._subsystems.get(name)
        if wrapper is None:
            return
        if wrapper.health.consecutive_failures >= 3:
            backoff = min(
                self.config.restart_backoff_base * (2 ** wrapper.health.consecutive_failures),
                self.config.restart_backoff_max,
            )
            logger.warning(
                "Auto-restarting subsystem '%s' after %d failures (backoff %.1fs)",
                name,
                wrapper.health.consecutive_failures,
                backoff,
            )
            if backoff > 0:
                time.sleep(backoff)
            wrapper.restart()

    def _snapshot_status(self, tick_results: dict[str, Any]) -> dict[str, Any]:
        """Build a lightweight status dict for the log."""
        return {
            "beat_number": tick_results.get("beat_number"),
            "timestamp": time.time(),
            "subsystem_ticks": {
                k: "ok" if "error" not in v else "err"
                for k, v in tick_results.items()
                if isinstance(v, dict)
            },
        }

    def _execute_direct(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Execute a task directly (in-process)."""
        try:
            fn = task_spec.get("fn")
            args = task_spec.get("args", [])
            kwargs = task_spec.get("kwargs", {})
            if callable(fn):
                result = fn(*args, **kwargs)
                return {"dispatched": True, "success": True, "result": result}
            return {
                "dispatched": False,
                "success": False,
                "reason": "no callable fn in task_spec",
            }
        except Exception as exc:
            return {"dispatched": False, "success": False, "reason": str(exc)}

    def _delegate_subagent(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        """Delegate to a subagent (placeholder — real dispatch hooks into OpenClaw)."""
        return {
            "dispatched": True,
            "success": True,
            "mode": "subagent",
            "task_spec": task_spec,
            "reason": "subagent delegation placeholder",
        }

    # ── queued task management ──────────────────────────────

    def drain_queue(self, max_tasks: int = 10) -> list[dict[str, Any]]:
        """Pull up to *max_tasks* from the queued task list."""
        with self._lock:
            drained = self._queued_tasks[:max_tasks]
            self._queued_tasks = self._queued_tasks[max_tasks:]
            return drained

    def queue_length(self) -> int:
        """Current number of queued tasks."""
        with self._lock:
            return len(self._queued_tasks)

    # ── properties ──────────────────────────────────────────

    @property
    def beat_count(self) -> int:
        with self._lock:
            return self._beat_count

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def shutdown_flag(self) -> bool:
        with self._lock:
            return self._shutdown


# ── lightweight SDA act/noop helpers ──────────────────────────

class _NoopAct:
    """Act that does nothing (for monitoring pipelines)."""

    from fleet.sense_decide_act import ActResult

    def execute(self, decision: Any) -> "ActResult":
        from fleet.sense_decide_act import ActResult

        return ActResult(success=True, latency_ms=0.0, side_effects=["noop"])


class _IdentitySense:
    """Sense adapter that reads AgentRegistry health."""

    from fleet.sense_decide_act import Observation

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def observe(self) -> "Observation":
        from fleet.sense_decide_act import Observation

        try:
            agents = self.registry.list_agents() if hasattr(self.registry, "list_agents") else []
            return Observation(
                timestamp=time.time(),
                source="identity_sense",
                metrics={"agent_count": len(agents), "healthy": True},
                severity_hint="info",
            )
        except Exception as exc:
            return Observation(
                timestamp=time.time(),
                source="identity_sense",
                metrics={"error": str(exc)},
                severity_hint="warning",
            )


class _IdentityDecide:
    """Decide if agent re-registration is needed."""

    from fleet.sense_decide_act import Decision

    def decide(self, observation: Any) -> "Decision":
        from fleet.sense_decide_act import Decision

        metrics = getattr(observation, "metrics", {})
        if "error" in metrics:
            return Decision(
                action_type="re_register",
                confidence=0.9,
                reasoning="Registry unhealthy — re-register agents",
            )
        agent_count = metrics.get("agent_count", 0)
        if agent_count < 1:
            return Decision(
                action_type="re_register",
                confidence=0.7,
                reasoning="No agents registered",
            )
        return Decision(action_type="noop", confidence=1.0, reasoning="Registry healthy")


class _MeshDiversitySense:
    """Sense adapter that reads mesh diversity score."""

    from fleet.sense_decide_act import Observation

    def __init__(self, mesh: Any) -> None:
        self.mesh = mesh

    def observe(self) -> "Observation":
        from fleet.sense_decide_act import Observation

        try:
            stats = self.mesh.stats if hasattr(self.mesh, "stats") else {}
            diversity = stats.get("total_entries", 0)
            return Observation(
                timestamp=time.time(),
                source="mesh_diversity_sense",
                metrics={"diversity": diversity, "total_entries": diversity},
                severity_hint="info",
            )
        except Exception as exc:
            return Observation(
                timestamp=time.time(),
                source="mesh_diversity_sense",
                metrics={"error": str(exc)},
                severity_hint="warning",
            )


class _MeshDiversityDecide:
    """Decide if cross-node breeding is needed based on diversity."""

    from fleet.sense_decide_act import Decision

    def decide(self, observation: Any) -> "Decision":
        from fleet.sense_decide_act import Decision

        metrics = getattr(observation, "metrics", {})
        diversity = metrics.get("diversity", 0)
        if "error" in metrics:
            return Decision(
                action_type="cross_breed",
                confidence=0.6,
                reasoning="Mesh diversity unreadable — attempt cross-node breed",
            )
        if diversity < 5:
            return Decision(
                action_type="cross_breed",
                confidence=0.8,
                reasoning=f"Low diversity ({diversity}) — cross-node breed recommended",
            )
        return Decision(action_type="noop", confidence=1.0, reasoning="Diversity adequate")


class _OpcodeSafetySense:
    """Sense adapter that reads opcode capability index."""

    from fleet.sense_decide_act import Observation

    def __init__(self, opcode_index: Any) -> None:
        self.opcode_index = opcode_index

    def observe(self) -> "Observation":
        from fleet.sense_decide_act import Observation

        try:
            summary = self.opcode_index.get_summary() if hasattr(self.opcode_index, "get_summary") else {}
            return Observation(
                timestamp=time.time(),
                source="opcode_safety_sense",
                metrics=summary,
                severity_hint="info",
            )
        except Exception as exc:
            return Observation(
                timestamp=time.time(),
                source="opcode_safety_sense",
                metrics={"error": str(exc)},
                severity_hint="warning",
            )


class _OpcodeSafetyDecide:
    """Decide if compile requests are safe based on opcode index."""

    from fleet.sense_decide_act import Decision

    def decide(self, observation: Any) -> "Decision":
        from fleet.sense_decide_act import Decision

        metrics = getattr(observation, "metrics", {})
        if "error" in metrics:
            return Decision(
                action_type="flag_unsafe",
                confidence=0.7,
                reasoning="Opcode index unreadable — assume unsafe",
            )
        untested = metrics.get("untested_count", 0)
        if untested > 0:
            return Decision(
                action_type="flag_unsafe",
                confidence=0.8,
                reasoning=f"{untested} untested opcodes — gate compile tasks",
            )
        return Decision(action_type="noop", confidence=1.0, reasoning="All opcodes safe")


class _BreederThermalSense:
    """Sense adapter that reads breeder queue + thermal budget."""

    from fleet.sense_decide_act import Observation

    def __init__(self, breeder: Any, thermal_limits: dict[str, float]) -> None:
        self.breeder = breeder
        self.thermal_limits = thermal_limits

    def observe(self) -> "Observation":
        from fleet.sense_decide_act import Observation

        try:
            breeder_status = self.breeder.get_status() if hasattr(self.breeder, "get_status") else {}
            queue_depth = breeder_status.get("queue_depth", 0)
            queue_capacity = breeder_status.get("queue_capacity", 0)
            thermal = self.thermal_limits
            return Observation(
                timestamp=time.time(),
                source="breeder_thermal_sense",
                metrics={
                    "queue_depth": queue_depth,
                    "queue_capacity": queue_capacity,
                    "thermal_limits": thermal,
                    "queue_ratio": queue_depth / max(queue_capacity, 1),
                },
                severity_hint="info",
            )
        except Exception as exc:
            return Observation(
                timestamp=time.time(),
                source="breeder_thermal_sense",
                metrics={"error": str(exc)},
                severity_hint="warning",
            )


class _BreederThermalDecide:
    """Decide spawn rate based on breeder queue and thermal budget."""

    from fleet.sense_decide_act import Decision

    def decide(self, observation: Any) -> "Decision":
        from fleet.sense_decide_act import Decision

        metrics = getattr(observation, "metrics", {})
        if "error" in metrics:
            return Decision(
                action_type="throttle",
                confidence=0.8,
                reasoning="Cannot read breeder/thermal state — throttle spawning",
            )
        queue_ratio = metrics.get("queue_ratio", 0.0)
        if queue_ratio > 0.8:
            return Decision(
                action_type="throttle",
                confidence=0.9,
                reasoning=f"Queue near full ({queue_ratio:.0%}) — throttle spawning",
            )
        if queue_ratio < 0.2:
            return Decision(
                action_type="accelerate",
                confidence=0.8,
                reasoning=f"Queue near empty ({queue_ratio:.0%}) — accelerate spawning",
            )
        return Decision(action_type="noop", confidence=1.0, reasoning="Queue/thermal balanced")


class _HebbianRoutingSense:
    """Sense adapter that reads peer diversity from HebbianMeshLayer."""

    from fleet.sense_decide_act import Observation

    def __init__(self, hebbian: Any) -> None:
        self.hebbian = hebbian

    def observe(self) -> "Observation":
        from fleet.sense_decide_act import Observation

        try:
            diversity = self.hebbian.get_diversity_score() if hasattr(self.hebbian, "get_diversity_score") else 0.5
            chaos = self.hebbian.chaos_factor if hasattr(self.hebbian, "chaos_factor") else 0.3
            return Observation(
                timestamp=time.time(),
                source="hebbian_routing_sense",
                metrics={"peer_diversity": diversity, "chaos_factor": chaos},
                severity_hint="info",
            )
        except Exception as exc:
            return Observation(
                timestamp=time.time(),
                source="hebbian_routing_sense",
                metrics={"error": str(exc)},
                severity_hint="warning",
            )


class _HebbianRoutingDecide:
    """Decide routing weights based on peer diversity."""

    from fleet.sense_decide_act import Decision

    def decide(self, observation: Any) -> "Decision":
        from fleet.sense_decide_act import Decision

        metrics = getattr(observation, "metrics", {})
        if "error" in metrics:
            return Decision(
                action_type="reweight_uniform",
                confidence=0.6,
                reasoning="Hebbian state unreadable — uniform routing weights",
            )
        diversity = metrics.get("peer_diversity", 0.5)
        chaos = metrics.get("chaos_factor", 0.3)
        if diversity < 0.3:
            return Decision(
                action_type="reweight_explore",
                confidence=0.8,
                reasoning=f"Low diversity ({diversity:.2f}) — increase exploration weights",
                payload={"chaos_boost": chaos + 0.1},
            )
        if diversity > 0.7:
            return Decision(
                action_type="reweight_exploit",
                confidence=0.7,
                reasoning=f"High diversity ({diversity:.2f}) — increase exploitation weights",
                payload={"chaos_reduce": max(0.1, chaos - 0.1)},
            )
        return Decision(action_type="noop", confidence=1.0, reasoning="Diversity balanced")


class _LoggingAct:
    """Act that logs decisions to the decision journal."""

    from fleet.sense_decide_act import ActResult

    def __init__(self, pipeline_name: str, journal_path: str = "") -> None:
        self.pipeline_name = pipeline_name
        self.journal_path = journal_path

    def execute(self, decision: Any) -> "ActResult":
        from fleet.sense_decide_act import ActResult

        try:
            from logos.decision_journal import log_human_command
            log_human_command(
                intent=type("Intent", (), {
                    "action": getattr(decision, "action_type", "noop"),
                    "raw_command": getattr(decision, "reasoning", ""),
                    "is_destructive": lambda: False,
                })(),
                confirmed=True,
                scope=self.pipeline_name,
                journal_path=self.journal_path or None,
            )
        except Exception as exc:
            logger.debug("Decision journal log failed: %s", exc)

        return ActResult(success=True, latency_ms=0.0, side_effects=["logged"])


class _GatewaySense:
    """Sense adapter that reads GatewayPacing status."""

    from fleet.sense_decide_act import Observation

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway

    def observe(self) -> "Observation":
        from fleet.sense_decide_act import Observation

        try:
            status = self.gateway.get_status()
            return Observation(
                timestamp=time.time(),
                source="gateway_sense",
                metrics=status,
                severity_hint="info",
            )
        except Exception as exc:
            return Observation(
                timestamp=time.time(),
                source="gateway_sense",
                metrics={"error": str(exc)},
                severity_hint="warning",
            )


class _FluxSense:
    """Sense adapter that produces empty FLUX context."""

    from fleet.sense_decide_act import Observation

    def observe(self) -> "Observation":
        from fleet.sense_decide_act import Observation

        return Observation(
            timestamp=time.time(),
            source="flux_sense",
            metrics={"preset": " FleetHealth"},
            severity_hint="info",
        )
