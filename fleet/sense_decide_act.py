"""Sense → Decide → Act Framework — unifying interface for fleet operations.

Every operational module in the sunset ecosystem is a variation of the same
distributed loop: SENSE collects state, DECIDE applies policy, ACT performs
the action. This module provides the common interface and built-in pipeline
adapters for all 20 cross-pollination patterns.

Reference: docs/SENSE_DECIDE_ACT.md
"""

from __future__ import annotations

__all__ = [
    "Observation",
    "Decision",
    "ActResult",
    "SDAPipeline",
    "Sense",
    "Decide",
    "Act",
    "Policy",
    "SDALoop",
    "TrapSense",
    "GatewayDispatchDecide",
    "HebbianMeshSense",
    "HebbianMeshAct",
    "FluxPresetDecide",
    "BreedCoordinationSense",
    "BreedCoordinationAct",
]

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── data structures ───────────────────────────────────────


@dataclass(frozen=True)
class Observation:
    """Raw state collected by a Sense node.

    Attributes
    ----------
    timestamp : float
        Unix time when the observation was produced.
    source : str
        Identifier for the sensing component (e.g. "thermal_trap").
    metrics : dict[str, Any]
        Arbitrary key/value measurements.
    severity_hint : str
        One of "info", "warning", "critical" — advisory only; the Decide
        stage may override based on policy.
    """

    timestamp: float
    source: str
    metrics: dict[str, Any] = field(default_factory=dict)
    severity_hint: str = "info"


@dataclass(frozen=True)
class Decision:
    """Result of evaluating an Observation against policy.

    Attributes
    ----------
    action_type : str
        Symbolic action name (e.g. "escalate", "route", "gossip", "noop").
    confidence : float
        0.0–1.0 certainty that this decision is correct.
    payload : dict[str, Any]
        Action-specific parameters.
    reasoning : str
        Human-readable justification (useful for logs / audits).
    """

    action_type: str
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass(frozen=True)
class ActResult:
    """Outcome of executing a Decision.

    Attributes
    ----------
    success : bool
        Whether the action completed without error.
    latency_ms : float
        Wall-clock milliseconds spent in execution.
    side_effects : list[str]
        Names of state mutations or external effects produced.
    new_observations : list[Observation]
        Additional observations triggered by the action (feedback loop).
    """

    success: bool
    latency_ms: float
    side_effects: list[str] = field(default_factory=list)
    new_observations: list[Observation] = field(default_factory=list)


@dataclass
class SDAPipeline:
    """A complete Sense→Decide→Act wiring.

    Attributes
    ----------
    name : str
        Unique pipeline identifier.
    sense : Sense
        The sensing component.
    decide : Decide
        The decision component.
    act : Act
        The action component.
    interval_ms : float
        Minimum milliseconds between ticks for this pipeline.
    last_run : float
        Unix timestamp of the most recent tick (0.0 = never).
    enabled : bool
        If False, the pipeline is skipped on tick().
    """

    name: str
    sense: "Sense"
    decide: "Decide"
    act: "Act"
    interval_ms: float = 0.0
    last_run: float = 0.0
    enabled: bool = True


# ── abstract bases ────────────────────────────────────────


class Sense(ABC):
    """Collect raw state and package it as an Observation."""

    @abstractmethod
    def observe(self) -> Observation:
        """Return the current Observation.

        Implementations should be fast (< 50 ms) and side-effect-free.
        Heavy I/O should be cached or delegated to background threads.
        """
        raise NotImplementedError


class Decide(ABC):
    """Evaluate an Observation and produce a Decision."""

    @abstractmethod
    def evaluate(self, observation: Observation) -> Decision:
        """Apply policy/rules to *observation* and return a Decision.

        Confidence must be in [0.0, 1.0]. Values < 0.5 cause the Act stage
        to skip execution (configurable per pipeline).
        """
        raise NotImplementedError


class Act(ABC):
    """Execute a Decision and report the result."""

    @abstractmethod
    def execute(self, decision: Decision) -> ActResult:
        """Perform the action described by *decision*.

        Should catch its own exceptions and return ``success=False``
        rather than raising, so the loop stays stable.
        """
        raise NotImplementedError


# ── Policy engine ─────────────────────────────────────────


class Policy:
    """Rule-based decision engine.

    Each rule is a ``(condition, action_type, confidence, reasoning)``
    tuple. The first matching rule wins. If no rule matches, a fallback
    ``noop`` decision is returned with confidence 1.0.

    Conditions are callables: ``(observation: Observation) -> bool``.
    """

    def __init__(self) -> None:
        self._rules: list[
            tuple[
                Callable[[Observation], bool],
                str,  # action_type
                float,  # confidence
                str,  # reasoning
            ]
        ] = []
        self._lock = threading.Lock()

    def add_rule(
        self,
        condition: Callable[[Observation], bool],
        action_type: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Append a rule. Rules are evaluated in insertion order."""
        with self._lock:
            self._rules.append((condition, action_type, confidence, reasoning))

    def evaluate(self, observation: Observation) -> Decision:
        """Evaluate *observation* against registered rules."""
        with self._lock:
            rules = list(self._rules)

        for condition, action_type, confidence, reasoning in rules:
            try:
                if condition(observation):
                    return Decision(
                        action_type=action_type,
                        confidence=max(0.0, min(1.0, confidence)),
                        payload={"matched_rule": len(rules)},
                        reasoning=reasoning,
                    )
            except Exception as exc:
                logger.warning("Policy rule failed for %s: %s", observation.source, exc)

        return Decision(
            action_type="noop",
            confidence=1.0,
            payload={},
            reasoning="No policy rule matched; default noop",
        )


# ── SDALoop orchestrator ──────────────────────────────────


class SDALoop:
    """Orchestrates one or more Sense→Decide→Act pipelines.

    Typical lifecycle::

        loop = SDALoop()
        loop.register(TrapSense(registry), policy, escalation_act, name="thermal")
        loop.tick()  # sense all → decide all → act all
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        metrics_window: int = 100,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._metrics_window = metrics_window
        self._pipelines: dict[str, SDAPipeline] = {}
        self._lock = threading.Lock()

        # Metrics
        self._tick_count: int = 0
        self._pipeline_ticks: dict[str, int] = {}
        self._latencies_ms: list[float] = []
        self._decision_counts: dict[str, int] = {}
        self._act_success_count: int = 0
        self._act_failure_count: int = 0

    # ── public API ──────────────────────────────────────────

    def register(
        self,
        sense: Sense,
        decide: Decide,
        act: Act,
        name: str,
        interval_ms: float = 0.0,
        enabled: bool = True,
    ) -> SDAPipeline:
        """Wire a complete SDA pipeline and return the handle."""
        pipeline = SDAPipeline(
            name=name,
            sense=sense,
            decide=decide,
            act=act,
            interval_ms=interval_ms,
            last_run=0.0,
            enabled=enabled,
        )
        with self._lock:
            self._pipelines[name] = pipeline
            self._pipeline_ticks[name] = 0
        logger.debug("Registered SDA pipeline: %s", name)
        return pipeline

    def get_pipeline(self, name: str) -> SDAPipeline:
        """Retrieve a pipeline by name. Raises KeyError if missing."""
        with self._lock:
            if name not in self._pipelines:
                raise KeyError(f"No pipeline named '{name}'")
            return self._pipelines[name]

    def list_pipelines(self) -> list[str]:
        """Return all registered pipeline names."""
        with self._lock:
            return list(self._pipelines.keys())

    def enable(self, name: str) -> None:
        """Enable a pipeline."""
        with self._lock:
            self._pipelines[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a pipeline (skipped on tick)."""
        with self._lock:
            self._pipelines[name].enabled = False

    def tick(self) -> dict[str, ActResult | None]:
        """Run one full SDA cycle across all enabled pipelines.

        Returns a mapping ``pipeline_name → ActResult | None``.
        ``None`` means the pipeline was skipped (disabled, throttled by
        interval, or confidence too low).
        """
        now = time.monotonic()
        results: dict[str, ActResult | None] = {}

        with self._lock:
            pipelines = list(self._pipelines.values())

        for pipe in pipelines:
            if not pipe.enabled:
                results[pipe.name] = None
                continue

            if pipe.interval_ms > 0.0 and (now - pipe.last_run) * 1000.0 < pipe.interval_ms:
                results[pipe.name] = None
                continue

            result = self._tick_pipeline(pipe, now)
            results[pipe.name] = result

        with self._lock:
            self._tick_count += 1

        return results

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregate loop metrics.

        Keys
        ----
        tick_count : int
            Total number of tick() calls.
        pipeline_ticks : dict[str, int]
            Per-pipeline tick counts.
        mean_latency_ms : float | None
            Average tick latency across all pipelines.
        max_latency_ms : float | None
            Worst-case tick latency.
        decision_counts : dict[str, int]
            How many times each action_type was decided.
        act_success_rate : float
            Fraction of Act executions that returned success=True.
        """
        with self._lock:
            latencies = list(self._latencies_ms)
            decision_counts = dict(self._decision_counts)
            total_acts = self._act_success_count + self._act_failure_count

        mean_lat = None
        max_lat = None
        if latencies:
            mean_lat = round(sum(latencies) / len(latencies), 3)
            max_lat = round(max(latencies), 3)

        success_rate = 0.0
        if total_acts > 0:
            success_rate = round(self._act_success_count / total_acts, 3)

        return {
            "tick_count": self._tick_count,
            "pipeline_ticks": dict(self._pipeline_ticks),
            "mean_latency_ms": mean_lat,
            "max_latency_ms": max_lat,
            "decision_counts": decision_counts,
            "act_success_rate": success_rate,
            "total_pipelines": len(self._pipelines),
        }

    # ── internal tick ───────────────────────────────────────

    def _tick_pipeline(
        self, pipe: SDAPipeline, now: float
    ) -> ActResult | None:
        """Run sense → decide → act for a single pipeline."""
        start = time.perf_counter()

        # ── SENSE ──
        try:
            observation = pipe.sense.observe()
        except Exception as exc:
            logger.error("Sense failed in pipeline '%s': %s", pipe.name, exc)
            return ActResult(
                success=False,
                latency_ms=0.0,
                side_effects=["sense_error"],
                new_observations=[],
            )

        # ── DECIDE ──
        try:
            decision = pipe.decide.evaluate(observation)
        except Exception as exc:
            logger.error("Decide failed in pipeline '%s': %s", pipe.name, exc)
            return ActResult(
                success=False,
                latency_ms=0.0,
                side_effects=["decide_error"],
                new_observations=[],
            )

        # Confidence threshold gate
        if decision.confidence < self._confidence_threshold:
            logger.debug(
                "Pipeline '%s' skipped: confidence %.2f < threshold %.2f",
                pipe.name,
                decision.confidence,
                self._confidence_threshold,
            )
            return None

        # ── ACT ──
        try:
            act_result = pipe.act.execute(decision)
        except Exception as exc:
            logger.error("Act failed in pipeline '%s': %s", pipe.name, exc)
            act_result = ActResult(
                success=False,
                latency_ms=0.0,
                side_effects=["act_error"],
                new_observations=[],
            )

        latency = (time.perf_counter() - start) * 1000.0
        act_result = ActResult(
            success=act_result.success,
            latency_ms=latency,
            side_effects=act_result.side_effects,
            new_observations=act_result.new_observations,
        )

        # Update metrics
        with self._lock:
            self._pipeline_ticks[pipe.name] = self._pipeline_ticks.get(pipe.name, 0) + 1
            self._latencies_ms.append(latency)
            if len(self._latencies_ms) > self._metrics_window:
                self._latencies_ms = self._latencies_ms[-self._metrics_window :]
            self._decision_counts[decision.action_type] = (
                self._decision_counts.get(decision.action_type, 0) + 1
            )
            if act_result.success:
                self._act_success_count += 1
            else:
                self._act_failure_count += 1
            pipe.last_run = now

        return act_result


# ═══════════════════════════════════════════════════════════
# Built-in pipeline adapters — fleet modules wearing SDA hats
# ═══════════════════════════════════════════════════════════


class TrapSense(Sense):
    """Sense adapter for OperationalTrap / TrapRegistry.

    Observes the fleet health by running all registered traps.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def observe(self) -> Observation:
        from fleet.operational_trap import TrapRegistry, TrapSeverity

        if not isinstance(self.registry, TrapRegistry):
            return Observation(
                timestamp=time.time(),
                source="trap_sense",
                metrics={"error": "invalid registry type"},
                severity_hint="critical",
            )

        results = self.registry.run_all()
        critical_count = sum(
            1 for r in results if r.severity == TrapSeverity.CRITICAL
        )
        warning_count = sum(
            1 for r in results if r.severity == TrapSeverity.WARNING
        )

        severity = "info"
        if critical_count > 0:
            severity = "critical"
        elif warning_count > 0:
            severity = "warning"

        return Observation(
            timestamp=time.time(),
            source="trap_sense",
            metrics={
                "trap_count": len(self.registry._traps),
                "fired": len(results),
                "critical": critical_count,
                "warning": warning_count,
                "conditions": [r.condition for r in results],
            },
            severity_hint=severity,
        )


class GatewayDispatchDecide(Decide):
    """Decide adapter for GatewayPacing + DispatchRouter.

    Observes gateway state and task context, decides whether to
    dispatch directly, delegate to subagent, or defer.
    """

    def __init__(
        self,
        gateway: Any,
        router: Any | None = None,
        task_description: str = "default task",
    ) -> None:
        self.gateway = gateway
        self.router = router
        self.task_description = task_description

    def evaluate(self, observation: Observation) -> Decision:
        from fleet.gateway_pacing import GatewayPacing, State

        if not isinstance(self.gateway, GatewayPacing):
            return Decision(
                action_type="noop",
                confidence=1.0,
                payload={"error": "invalid gateway type"},
                reasoning="GatewayPacing instance required",
            )

        status = self.gateway.get_status()
        state = status.get("state", "UNKNOWN")
        consecutive_timeouts = status.get("consecutive_timeouts", 0)

        # Build confidence from gateway health
        if state == State.OPEN.name:
            confidence = 1.0
            action = "dispatch"
            reasoning = "Gateway OPEN — normal dispatch allowed"
        elif state == State.HALF_OPEN.name:
            confidence = 0.6
            action = "probe_dispatch"
            reasoning = "Gateway HALF_OPEN — cautious probe allowed"
        elif state == State.CLOSED.name:
            confidence = 0.2
            action = "defer"
            reasoning = f"Gateway CLOSED — backoff remaining {status.get('backoff_remaining', 0):.1f}s"
        else:
            confidence = 0.5
            action = "noop"
            reasoning = f"Unknown gateway state: {state}"

        payload = {
            "gateway_state": state,
            "consecutive_timeouts": consecutive_timeouts,
            "task_description": self.task_description,
        }

        # If router available, add routing recommendation
        if self.router is not None:
            try:
                route = self.router.route(self.task_description)
                payload["router_mode"] = route.get("mode", "unknown")
                payload["estimated_seconds"] = route.get("estimated_seconds", 0)
            except Exception as exc:
                payload["router_error"] = str(exc)

        return Decision(
            action_type=action,
            confidence=confidence,
            payload=payload,
            reasoning=reasoning,
        )


class HebbianMeshSense(Sense):
    """Sense adapter for HebbianMeshLayer.

    Observes mesh diversity, chaos factor, and peer affinities.
    """

    def __init__(self, mesh_layer: Any) -> None:
        self.mesh_layer = mesh_layer

    def observe(self) -> Observation:
        stats = getattr(self.mesh_layer, "stats", {})
        diversity = getattr(self.mesh_layer, "chaos_factor", 0.0)

        try:
            diversity_score = self.mesh_layer.get_diversity_score()
        except Exception:
            diversity_score = None

        return Observation(
            timestamp=time.time(),
            source="hebbian_mesh",
            metrics={
                "peer_count": stats.get("peer_count", 0),
                "blacklisted_count": stats.get("blacklisted_count", 0),
                "avg_strength": stats.get("avg_strength", 0.0),
                "avg_trust": stats.get("avg_trust", 0.0),
                "chaos_factor": diversity,
                "diversity_score": diversity_score,
            },
            severity_hint="warning" if diversity is not None and diversity > 0.4 else "info",
        )


class HebbianMeshAct(Act):
    """Act adapter for HebbianMeshLayer.

    Executes routing decisions: gossip, blacklist, or reset affinity.
    """

    def __init__(self, mesh_layer: Any, peer_pool: list[str] | None = None) -> None:
        self.mesh_layer = mesh_layer
        self.peer_pool = peer_pool or []

    def execute(self, decision: Decision) -> ActResult:
        from swarm.hebbian_mesh import HebbianMeshLayer, HebbianOutcome

        if not isinstance(self.mesh_layer, HebbianMeshLayer):
            return ActResult(
                success=False,
                latency_ms=0.0,
                side_effects=["invalid_mesh_layer"],
            )

        start = time.perf_counter()
        side_effects: list[str] = []
        new_obs: list[Observation] = []

        action = decision.action_type

        if action == "gossip" and self.peer_pool:
            try:
                peers = self.mesh_layer.select_peers_for_gossip(
                    self.peer_pool, k=min(3, len(self.peer_pool))
                )
                side_effects.append(f"selected_peers:{peers}")
                new_obs.append(
                    Observation(
                        timestamp=time.time(),
                        source="hebbian_mesh_act",
                        metrics={"selected_peers": peers},
                        severity_hint="info",
                    )
                )
            except Exception as exc:
                side_effects.append(f"gossip_error:{exc}")

        elif action == "reset_affinity":
            peer_id = decision.payload.get("peer_id")
            if peer_id:
                self.mesh_layer.reset_affinity(peer_id)
                side_effects.append(f"reset_affinity:{peer_id}")

        elif action == "blacklist":
            peer_id = decision.payload.get("peer_id")
            if peer_id:
                self.mesh_layer.update_affinity(peer_id, HebbianOutcome.VIOLATION)
                side_effects.append(f"blacklisted:{peer_id}")

        latency = (time.perf_counter() - start) * 1000.0
        return ActResult(
            success=True,
            latency_ms=latency,
            side_effects=side_effects,
            new_observations=new_obs,
        )


class FluxPresetDecide(Decide):
    """Decide adapter for FluxPresetLibrary.

    Evaluates fleet health metrics against FLUX constraint presets.
    """

    def __init__(
        self,
        library: Any,
        preset_name: str = "FleetHealth",
        context_builder: Callable[[Observation], dict[str, Any]] | None = None,
    ) -> None:
        self.library = library
        self.preset_name = preset_name
        self._context_builder = context_builder

    def evaluate(self, observation: Observation) -> Decision:
        from sunset.flux_preset_library import FluxPresetLibrary

        if not isinstance(self.library, FluxPresetLibrary):
            return Decision(
                action_type="noop",
                confidence=1.0,
                payload={"error": "invalid library type"},
                reasoning="FluxPresetLibrary instance required",
            )

        ctx = {}
        if self._context_builder is not None:
            ctx = self._context_builder(observation)
        else:
            # Default context mapping from observation metrics
            ctx = {
                "thermal_headroom": observation.metrics.get("thermal_headroom", 0.5),
                "chaos": observation.metrics.get("chaos_factor", 0.0),
                "diversity_score": observation.metrics.get("diversity_score", 1.0),
                "consecutive_failures": observation.metrics.get("consecutive_failures", 0),
                "last_heartbeat": observation.metrics.get("last_heartbeat", time.time()),
            }

        try:
            results = self.library.apply_preset(self.preset_name, ctx)
        except Exception as exc:
            return Decision(
                action_type="noop",
                confidence=1.0,
                payload={"preset_error": str(exc)},
                reasoning=f"Preset application failed: {exc}",
            )

        all_passed = all(r.get("passed", False) for r in results)
        any_critical = any(r.get("severity", "info") == "critical" for r in results)

        if all_passed:
            return Decision(
                action_type="continue",
                confidence=1.0,
                payload={"preset_results": results},
                reasoning=f"Preset '{self.preset_name}': all constraints passed",
            )

        if any_critical:
            return Decision(
                action_type="block",
                confidence=0.9,
                payload={"preset_results": results},
                reasoning=f"Preset '{self.preset_name}': critical constraint breached",
            )

        return Decision(
            action_type="warn",
            confidence=0.7,
            payload={"preset_results": results},
            reasoning=f"Preset '{self.preset_name}': some constraints breached",
        )


class BreedCoordinationSense(Sense):
    """Sense adapter for MetronomeBridge + MeshVectorTables.

    Observes breeding readiness: beat cycle position, diversity,
    and cross-node parent pool availability.
    """

    def __init__(
        self,
        bridge: Any,
        vector_index: Any | None = None,
    ) -> None:
        self.bridge = bridge
        self.vector_index = vector_index

    def observe(self) -> Observation:
        metrics: dict[str, Any] = {
            "tick_counter": getattr(self.bridge, "_tick_counter", 0),
        }

        # Beat phase: 0=full, 1=thermal, 2=breed, 3=perception
        tick_counter = getattr(self.bridge, "_tick_counter", 0)
        beat_phase = tick_counter % 4
        metrics["beat_phase"] = beat_phase

        # Latency report if available
        if hasattr(self.bridge, "get_latency_report"):
            try:
                metrics["latencies"] = self.bridge.get_latency_report()
            except Exception:
                pass

        # Breedable pool from vector index
        pool_size = 0
        if self.vector_index is not None:
            try:
                pool = self.vector_index.get_breedable_pool(max_results=50)
                pool_size = len(pool)
                metrics["breedable_pool_size"] = pool_size
                metrics["pool_mean_fitness"] = (
                    sum(e.fitness for e in pool) / len(pool) if pool else 0.0
                )
            except Exception:
                pass

        severity = "info"
        if beat_phase == 2 and pool_size == 0:
            severity = "warning"  # breeding beat but no candidates

        return Observation(
            timestamp=time.time(),
            source="breed_coordination",
            metrics=metrics,
            severity_hint=severity,
        )


class BreedCoordinationAct(Act):
    """Act adapter for MetronomeBridge + MeshVectorTables.

    Executes breeding coordination actions: trigger beat dispatch,
    sync payloads, or log readiness state.
    """

    def __init__(
        self,
        bridge: Any,
        vector_index: Any | None = None,
    ) -> None:
        self.bridge = bridge
        self.vector_index = vector_index

    def execute(self, decision: Decision) -> ActResult:
        start = time.perf_counter()
        side_effects: list[str] = []
        new_obs: list[Observation] = []

        action = decision.action_type

        if action == "dispatch_beat":
            beat = decision.payload.get("beat_number", 0)
            try:
                dispatched = self.bridge.on_metronome_beat(beat, tempo_ms=500.0)
                side_effects.append(f"dispatched_{len(dispatched)}_rooms")
                new_obs.append(
                    Observation(
                        timestamp=time.time(),
                        source="breed_coordination_act",
                        metrics={"dispatched_rooms": len(dispatched)},
                        severity_hint="info",
                    )
                )
            except Exception as exc:
                side_effects.append(f"dispatch_error:{exc}")

        elif action == "sync_fleet" and self.vector_index is not None:
            try:
                payload = self.vector_index.get_fleet_sync_payload()
                side_effects.append(f"sync_payload_size:{len(payload)}")
            except Exception as exc:
                side_effects.append(f"sync_error:{exc}")

        elif action == "noop":
            side_effects.append("idle")

        latency = (time.perf_counter() - start) * 1000.0
        return ActResult(
            success=True,
            latency_ms=latency,
            side_effects=side_effects,
            new_observations=new_obs,
        )
