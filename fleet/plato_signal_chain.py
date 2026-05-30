"""PLATO Signal Chain Adapter — room observations → SenseDecideAct → breeding.

Wires PLATO room state (occupancy, diversity, thermal, lifecycle) into the
SDA framework so that what happens in rooms drives fleet-level breeding
decisions automatically.

Usage
-----
    chain = PlatoSignalChain(room_source=room_client)
    chain.start(loop_interval_ms=5000)   # 5-second tick
    # ... later ...
    chain.stop()

Reference: docs/PLATO_SIGNAL_CHAIN.md
"""

from __future__ import annotations

__all__ = [
    "PlatoRoomSense",
    "PlatoBreedingPolicy",
    "PlatoBreedingAct",
    "PlatoSignalChain",
    "RoomObservation",
    "MockRoomSource",
]

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fleet.sense_decide_act import (
    Act,
    ActResult,
    Decide,
    Decision,
    Observation,
    Policy,
    SDALoop,
    Sense,
)

logger = logging.getLogger(__name__)


# ── data structures ─────────────────────────────────────────


@dataclass
class RoomObservation:
    """Structured room state produced by a room source."""

    room_id: str
    timestamp: float
    agent_count: int = 0
    diversity_score: float = 0.0
    thermal_cpu: float = 0.0
    thermal_mem: float = 0.0
    lifecycle_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── room sources ────────────────────────────────────────────


class MockRoomSource:
    """In-memory room source for testing and offline operation."""

    def __init__(self) -> None:
        self._rooms: Dict[str, RoomObservation] = {}
        self._lock = threading.Lock()

    def set_room(self, obs: RoomObservation) -> None:
        with self._lock:
            self._rooms[obs.room_id] = obs

    def get_room(self, room_id: str) -> Optional[RoomObservation]:
        with self._lock:
            return self._rooms.get(room_id)

    def list_rooms(self) -> List[str]:
        with self._lock:
            return list(self._rooms.keys())

    def snapshot(self) -> List[RoomObservation]:
        with self._lock:
            return list(self._rooms.values())


# ── Sense: PlatoRoomSense ───────────────────────────────────


class PlatoRoomSense(Sense):
    """Collect room state from a PLATO-connected source.

    Parameters
    ----------
    source : MockRoomSource or callable
        Object that provides room snapshots. If callable, called as
        ``source() -> List[RoomObservation]``.
    room_ids : list[str] | None
        If provided, only observe these rooms. Otherwise observe all.
    severity_thresholds : dict[str, float]
        Maps metric name to threshold for severity_hint escalation.
        Default: ``{"thermal_cpu": 80.0, "thermal_mem": 85.0}``.
    """

    def __init__(
        self,
        source: MockRoomSource | Callable[[], List[RoomObservation]],
        room_ids: List[str] | None = None,
        severity_thresholds: Dict[str, float] | None = None,
    ) -> None:
        self._source = source
        self._room_ids = set(room_ids) if room_ids else None
        self._severity_thresholds = severity_thresholds or {
            "thermal_cpu": 80.0,
            "thermal_mem": 85.0,
        }
        self._last_observations: List[RoomObservation] = []

    def observe(self) -> Observation:
        """Return a single Observation aggregating all room states.

        The Observation.metrics dict contains:
        - ``room_count`` — number of rooms observed
        - ``total_agents`` — sum of agents across rooms
        - ``mean_diversity`` — average diversity score
        - ``max_thermal_cpu`` — hottest room CPU%
        - ``max_thermal_mem`` — hottest room memory%
        - ``lifecycle_event_count`` — lifecycle events this tick
        - ``room_states`` — list of per-room dicts (for downstream Decide)
        """
        if callable(self._source):
            rooms = self._source()
        else:
            rooms = self._source.snapshot()

        if self._room_ids is not None:
            rooms = [r for r in rooms if r.room_id in self._room_ids]

        self._last_observations = rooms

        if not rooms:
            return Observation(
                timestamp=time.time(),
                source="plato_room_sense",
                metrics={"room_count": 0, "total_agents": 0},
                severity_hint="info",
            )

        total_agents = sum(r.agent_count for r in rooms)
        mean_diversity = (
            sum(r.diversity_score for r in rooms) / len(rooms) if rooms else 0.0
        )
        max_cpu = max((r.thermal_cpu for r in rooms), default=0.0)
        max_mem = max((r.thermal_mem for r in rooms), default=0.0)
        lifecycle_count = sum(len(r.lifecycle_events) for r in rooms)

        severity = "info"
        if max_cpu >= self._severity_thresholds.get("thermal_cpu", 80.0):
            severity = "critical"
        elif max_mem >= self._severity_thresholds.get("thermal_mem", 85.0):
            severity = "warning"
        elif mean_diversity < 0.2 and total_agents > 5:
            severity = "warning"

        metrics: Dict[str, Any] = {
            "room_count": len(rooms),
            "total_agents": total_agents,
            "mean_diversity": round(mean_diversity, 4),
            "max_thermal_cpu": round(max_cpu, 2),
            "max_thermal_mem": round(max_mem, 2),
            "lifecycle_event_count": lifecycle_count,
            "room_states": [
                {
                    "room_id": r.room_id,
                    "agent_count": r.agent_count,
                    "diversity": round(r.diversity_score, 4),
                    "cpu": round(r.thermal_cpu, 2),
                    "mem": round(r.thermal_mem, 2),
                    "events": r.lifecycle_events,
                }
                for r in rooms
            ],
        }

        return Observation(
            timestamp=time.time(),
            source="plato_room_sense",
            metrics=metrics,
            severity_hint=severity,
        )

    def last_rooms(self) -> List[RoomObservation]:
        """Return the raw room observations from the last ``observe()`` call."""
        return list(self._last_observations)


# ── Decide: PlatoBreedingPolicy ─────────────────────────────


class PlatoBreedingPolicy(Decide):
    """Rule-based breeding decisions from PLATO room observations.

    Built-in rules (evaluated in order):

    1. **Thermal critical** → ``sunset`` (evacuate / reduce load)
    2. **Low diversity + high occupancy** → ``breed`` (inject variation)
    3. **Room imbalance** → ``migrate`` (redistribute agents)
    4. **Quiet room with agents** → ``noop`` (monitor)
    5. **Default** → ``noop``

    All rules include reasoning and confidence scores.
    """

    def __init__(
        self,
        diversity_threshold: float = 0.25,
        occupancy_threshold: int = 8,
        thermal_cpu_threshold: float = 80.0,
        migrate_ratio: float = 3.0,
    ) -> None:
        self._policy = Policy()
        self._diversity_threshold = diversity_threshold
        self._occupancy_threshold = occupancy_threshold
        self._thermal_cpu_threshold = thermal_cpu_threshold
        self._migrate_ratio = migrate_ratio
        self._build_rules()

    def _build_rules(self) -> None:
        # Rule 1: thermal critical
        self._policy.add_rule(
            condition=lambda obs: obs.metrics.get("max_thermal_cpu", 0)
            >= self._thermal_cpu_threshold,
            action_type="sunset",
            confidence=0.9,
            reasoning="Thermal critical: max CPU above threshold, reduce fleet load",
        )

        # Rule 2: low diversity + high occupancy → breed
        self._policy.add_rule(
            condition=lambda obs: (
                obs.metrics.get("mean_diversity", 1.0)
                < self._diversity_threshold
                and obs.metrics.get("total_agents", 0)
                >= self._occupancy_threshold
            ),
            action_type="breed",
            confidence=0.85,
            reasoning="Low diversity with high occupancy: inject new agents to restore variation",
        )

        # Rule 3: room imbalance → migrate
        def _imbalance(obs: Observation) -> bool:
            states = obs.metrics.get("room_states", [])
            if len(states) < 2:
                return False
            counts = [s["agent_count"] for s in states]
            return max(counts) >= self._migrate_ratio * (min(counts) + 1)

        self._policy.add_rule(
            condition=_imbalance,
            action_type="migrate",
            confidence=0.75,
            reasoning="Room imbalance detected: redistribute agents across rooms",
        )

        # Rule 4: lifecycle events suggest churn
        self._policy.add_rule(
            condition=lambda obs: obs.metrics.get("lifecycle_event_count", 0) > 2,
            action_type="audit",
            confidence=0.6,
            reasoning="Multiple lifecycle events: audit room stability before breeding",
        )

    def evaluate(self, observation: Observation) -> Decision:
        return self._policy.evaluate(observation)


# ── Act: PlatoBreedingAct ───────────────────────────────────


class PlatoBreedingAct(Act):
    """Execute breeding decisions and record outcomes.

    Parameters
    ----------
    on_breed : callable | None
        Called with ``(room_states: list[dict])`` when decision is ``breed``.
    on_sunset : callable | None
        Called with ``(room_states: list[dict])`` when decision is ``sunset``.
    on_migrate : callable | None
        Called with ``(room_states: list[dict])`` when decision is ``migrate``.
    recorder : callable | None
        Generic recorder called with ``(decision: Decision, result: ActResult)``.
    """

    def __init__(
        self,
        on_breed: Callable[[List[Dict[str, Any]]], None] | None = None,
        on_sunset: Callable[[List[Dict[str, Any]]], None] | None = None,
        on_migrate: Callable[[List[Dict[str, Any]]], None] | None = None,
        recorder: Callable[[Decision, ActResult], None] | None = None,
    ) -> None:
        self._on_breed = on_breed
        self._on_sunset = on_sunset
        self._on_migrate = on_migrate
        self._recorder = recorder
        self._history: List[tuple[Decision, ActResult]] = []
        self._lock = threading.Lock()

    def execute(self, decision: Decision) -> ActResult:
        t0 = time.time()
        room_states = decision.payload.get("room_states", [])

        side_effects: List[str] = []
        new_observations: List[Observation] = []

        try:
            if decision.action_type == "breed":
                side_effects.append("breed_triggered")
                if self._on_breed:
                    self._on_breed(room_states)
                new_observations.append(
                    Observation(
                        timestamp=time.time(),
                        source="plato_breeding_act",
                        metrics={"action": "breed", "rooms": len(room_states)},
                        severity_hint="info",
                    )
                )
            elif decision.action_type == "sunset":
                side_effects.append("sunset_triggered")
                if self._on_sunset:
                    self._on_sunset(room_states)
                new_observations.append(
                    Observation(
                        timestamp=time.time(),
                        source="plato_breeding_act",
                        metrics={"action": "sunset", "rooms": len(room_states)},
                        severity_hint="warning",
                    )
                )
            elif decision.action_type == "migrate":
                side_effects.append("migrate_triggered")
                if self._on_migrate:
                    self._on_migrate(room_states)
                new_observations.append(
                    Observation(
                        timestamp=time.time(),
                        source="plato_breeding_act",
                        metrics={"action": "migrate", "rooms": len(room_states)},
                        severity_hint="info",
                    )
                )
            elif decision.action_type == "audit":
                side_effects.append("audit_logged")
            else:
                side_effects.append("noop")
                new_observations = []  # No feedback for noop

            result = ActResult(
                success=True,
                latency_ms=(time.time() - t0) * 1000,
                side_effects=side_effects,
                new_observations=new_observations,
            )
        except Exception as exc:
            logger.exception("PlatoBreedingAct failed for %s", decision.action_type)
            result = ActResult(
                success=False,
                latency_ms=(time.time() - t0) * 1000,
                side_effects=["error:" + str(exc)],
                new_observations=[],
            )

        with self._lock:
            self._history.append((decision, result))

        if self._recorder:
            try:
                self._recorder(decision, result)
            except Exception:
                logger.warning("Recorder failed", exc_info=True)

        return result

    def history(self) -> List[tuple[Decision, ActResult]]:
        """Return all (decision, result) pairs executed so far."""
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


# ── Convenience wrapper: PlatoSignalChain ────────────────────


class PlatoSignalChain:
    """High-level wrapper: source → Sense → Policy → Act → SDALoop.

    Example::

        source = MockRoomSource()
        source.set_room(RoomObservation(room_id="alpha", ...))

        chain = PlatoSignalChain(source)
        chain.start(loop_interval_ms=2000)
        # ... background thread ticks every 2s ...
        chain.stop()
    """

    def __init__(
        self,
        source: MockRoomSource | Callable[[], List[RoomObservation]],
        room_ids: List[str] | None = None,
        policy: PlatoBreedingPolicy | None = None,
        act: PlatoBreedingAct | None = None,
        loop: SDALoop | None = None,
    ) -> None:
        self.source = source
        self.sense = PlatoRoomSense(source=source, room_ids=room_ids)
        self.decide = policy or PlatoBreedingPolicy()
        self.act = act or PlatoBreedingAct()
        self.loop = loop or SDALoop()
        self._pipeline_name = "plato_signal_chain"
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._tick_count = 0
        self._lock = threading.Lock()

    def register(self) -> None:
        """Register the SDA pipeline with the loop (idempotent)."""
        if self._pipeline_name not in self.loop.list_pipelines():
            self.loop.register(
                sense=self.sense,
                decide=self.decide,
                act=self.act,
                name=self._pipeline_name,
                interval_ms=0.0,  # managed by our own thread
            )

    def tick(self) -> Dict[str, Any]:
        """Run one manual SDA cycle."""
        self.register()
        results = self.loop.tick()
        with self._lock:
            self._tick_count += 1
        return results

    def start(self, loop_interval_ms: float = 5000.0) -> None:
        """Start background thread that ticks every *loop_interval_ms*."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("PlatoSignalChain already running")
            return

        self._stop_event.clear()
        self.register()

        def _run() -> None:
            while not self._stop_event.is_set():
                self.loop.tick()
                with self._lock:
                    self._tick_count += 1
                self._stop_event.wait(timeout=loop_interval_ms / 1000.0)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info(
            "PlatoSignalChain started (interval=%.0f ms)", loop_interval_ms
        )

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("PlatoSignalChain stopped (ticks=%d)", self.tick_count())

    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    def last_results(self) -> Dict[str, Any]:
        """Return most recent tick results (best-effort, may be partial)."""
        # SDALoop.tick() returns fresh results each call; we don't cache them.
        # User should call tick() directly if they need results.
        return {}

    def metrics(self) -> Dict[str, Any]:
        """Return chain-level metrics."""
        return {
            "tick_count": self.tick_count(),
            "pipeline_name": self._pipeline_name,
            "room_sense_last_rooms": len(self.sense.last_rooms()),
            "act_history_size": len(self.act.history()),
        }
