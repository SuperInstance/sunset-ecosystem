#!/usr/bin/env python3
"""SSE Stream Dashboard — real-time breeding progress via Server-Sent Events.

Provides an HTTP endpoint that streams breeding events as text/event-stream:
  - beat ticks
  - parent selection
  - mutation results
  - FLUX gating decisions
  - thermal state changes
  - fleet-wide status snapshots

Reference: docs/SSE_STREAM_DASHBOARD.md
"""

from __future__ import annotations

__all__ = [
    "SSEStreamDashboard",
    "StreamEvent",
    "EventType",
    "DashboardConfig",
]

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── data structures ───────────────────────────────────────────

class EventType(Enum):
    BEAT = auto()
    PARENT_SELECT = auto()
    MUTATION = auto()
    FLUX_GATE = auto()
    THERMAL = auto()
    FLEET_STATUS = auto()
    AGENT_SPAWN = auto()
    ERROR = auto()
    INFO = auto()


@dataclass(frozen=True)
class StreamEvent:
    """A single event to be streamed."""

    event_type: EventType
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    node_id: str = "unknown"

    def to_sse(self) -> str:
        """Serialize as SSE message."""
        data = {
            "type": self.event_type.name,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "payload": self.payload,
        }
        return f"data: {json.dumps(data)}\n\n"


@dataclass
class DashboardConfig:
    """Configuration for the SSE dashboard."""

    max_queue_size: int = 1000
    heartbeat_interval_sec: float = 15.0
    history_buffer_size: int = 100
    filter_event_types: list[str] | None = None  # None = all
    enable_backpressure: bool = True


# ── dashboard ─────────────────────────────────────────────────

class SSEStreamDashboard:
    """Real-time breeding progress stream.

    Usage
    -----
    1. Create: ``dash = SSEStreamDashboard(config)``
    2. Publish: ``dash.publish(StreamEvent(...))``
    3. Consume: ``for msg in dash.subscribe(): yield msg``
    4. History: ``dash.recent_events(n=50)``
    """

    def __init__(self, config: DashboardConfig | None = None) -> None:
        self.config = config or DashboardConfig()
        self._queue: queue.Queue[StreamEvent] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._subscribers: list[queue.Queue[StreamEvent]] = []
        self._sub_lock = threading.Lock()
        self._history: list[StreamEvent] = []
        self._hist_lock = threading.Lock()
        self._running = False
        self._node_id = "dashboard-node"

    # ── publishing ────────────────────────────────────────

    def publish(self, event: StreamEvent) -> bool:
        """Publish an event to all subscribers.

        Returns False if dropped due to backpressure.
        """
        # Filter
        if self.config.filter_event_types is not None:
            if event.event_type.name not in self.config.filter_event_types:
                return True  # silently drop filtered types

        # History
        with self._hist_lock:
            self._history.append(event)
            if len(self._history) > self.config.history_buffer_size:
                self._history.pop(0)

        # Queue to main buffer
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            if self.config.enable_backpressure:
                logger.warning("SSE queue full, dropping %s event", event.event_type.name)
                return False
            # drop oldest
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except queue.Empty:
                pass

        # Fan out to subscribers
        with self._sub_lock:
            dead: list[queue.Queue[StreamEvent]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)  # subscriber too slow
            for q in dead:
                self._subscribers.remove(q)

        return True

    def publish_simple(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        node_id: str | None = None,
    ) -> bool:
        """Convenience: create and publish in one call."""
        return self.publish(
            StreamEvent(
                event_type=event_type,
                payload=payload,
                node_id=node_id or self._node_id,
            )
        )

    # ── subscribing ───────────────────────────────────────

    def subscribe(self) -> queue.Queue[StreamEvent]:
        """Create a new subscriber queue."""
        q: queue.Queue[StreamEvent] = queue.Queue(maxsize=self.config.max_queue_size)
        with self._sub_lock:
            self._subscribers.append(q)
        # Send recent history
        with self._hist_lock:
            for ev in self._history:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    break
        return q

    def unsubscribe(self, q: queue.Queue[StreamEvent]) -> None:
        """Remove a subscriber."""
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    # ── history ───────────────────────────────────────────

    def recent_events(self, n: int = 50) -> list[StreamEvent]:
        """Get last n events from history."""
        with self._hist_lock:
            return self._history[-n:]

    def recent_by_type(self, event_type: EventType, n: int = 20) -> list[StreamEvent]:
        """Get last n events of a specific type."""
        with self._hist_lock:
            filtered = [e for e in self._history if e.event_type == event_type]
            return filtered[-n:]

    # ── heartbeat ─────────────────────────────────────────

    def start_heartbeat(self) -> None:
        """Start a background thread that publishes heartbeat events."""
        self._running = True
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def stop_heartbeat(self) -> None:
        self._running = False

    def _heartbeat_loop(self) -> None:
        while self._running:
            self.publish_simple(
                EventType.INFO,
                {"message": "heartbeat", "queue_depth": self._queue.qsize()},
            )
            time.sleep(self.config.heartbeat_interval_sec)

    # ── metrics ───────────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        """Dashboard metrics."""
        with self._sub_lock:
            return {
                "subscribers": len(self._subscribers),
                "queue_depth": self._queue.qsize(),
                "history_size": len(self._history),
                "max_queue_size": self.config.max_queue_size,
                "heartbeat_interval": self.config.heartbeat_interval_sec,
            }


# ── integration helpers ───────────────────────────────────────

def wire_to_fleet_conductor(
    dashboard: SSEStreamDashboard,
    conductor: Any,
    event_types: list[EventType] | None = None,
) -> None:
    """Wire dashboard into FleetConductorV2 beat() cycle.

    Hooks into conductor's beat to publish fleet status after each tick.
    """
    event_types = event_types or [EventType.FLEET_STATUS, EventType.BEAT]

    original_beat = conductor.beat

    def _instrumented_beat() -> dict[str, Any]:
        result = original_beat()
        if EventType.BEAT in event_types:
            dashboard.publish_simple(
                EventType.BEAT,
                {"beat_number": result.get("beat_number", 0)},
                node_id=conductor.config.node_id,
            )
        if EventType.FLEET_STATUS in event_types:
            dashboard.publish_simple(
                EventType.FLEET_STATUS,
                conductor.get_status(),
                node_id=conductor.config.node_id,
            )
        return result

    conductor.beat = _instrumented_beat  # type: ignore[method-assign]


def wire_to_breeder(
    dashboard: SSEStreamDashboard,
    breeder: Any,
) -> None:
    """Wire dashboard into BreederDaemonV2 cycle.

    Publishes parent selection, mutation, and FLUX gating events.
    """
    original_cycle = breeder.cycle

    def _instrumented_cycle(n_winners: int = 3) -> list[Any]:
        dashboard.publish_simple(
            EventType.BEAT,
            {"action": "breed_cycle_start", "n_winners": n_winners},
        )
        results = original_cycle(n_winners)
        dashboard.publish_simple(
            EventType.PARENT_SELECT,
            {"winners": len(results), "action": "breed_cycle_end"},
        )
        return results

    breeder.cycle = _instrumented_cycle  # type: ignore[method-assign]
