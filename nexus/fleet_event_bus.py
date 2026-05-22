"""Fleet Event Bus — Cross-ship publish/subscribe for the Cocapn Fleet.

Implements a lightweight, asyncio-native event bus that any ship in the
fleet can use to emit or listen for structured events.  Designed to
bridge CCC-OS monitors, sunset-ecosystem breeders, cocapn-health
watchdogs, and PLATO tile pipelines without tight coupling.

Architecture::

    ┌──────────────────────────────────────────────────────────┐
    │                  FleetEventBus                           │
    │  ┌─────────────┐    ┌──────────────┐    ┌──────────┐ │
    │  │  Publisher  │───►│  EventRouter  │───►│ Subscribers│ │
    │  └─────────────┘    └──────────────┘    └──────────┘ │
    │         ▲                                          │
    │         │                                          │
    │  ┌──────┴──────┐  ┌──────────┐  ┌──────────┐     │
    │  │  CCC-OS    │  │ cocapn-  │  │ sunset-  │     │
    │  │  Monitor   │  │ health   │  │ ecosystem│     │
    │  └─────────────┘  └──────────┘  └──────────┘     │
    └──────────────────────────────────────────────────────────┘

Events are plain dicts with a required ``type`` field.  Delivery is
best-effort; dropped events are logged but do not crash the bus.

Usage::

    from nexus.fleet_event_bus import FleetEventBus
    bus = FleetEventBus()

    # Subscribe
    bus.on("ACT_NOW", lambda ev: breeder.spawn_for(ev))
    bus.on("service_down", lambda ev: thermal.rebalance(ev["node"]))

    # Publish
    bus.emit({"type": "ACT_NOW", "category": "architecture",
              "repo": "sunset-ecosystem", "priority": "P0"})

    # Asyncio integration
    await bus.emit_async({"type": "heartbeat", ...})

Thread-safe.  May be used from synchronous (monitor scripts) and
asynchronous (web server) contexts simultaneously.
"""

from __future__ import annotations

__all__ = [
    "FleetEventBus",
    "EventFilter",
    "FleetEvent",
]

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FleetEvent:
    """A structured fleet event with provenance."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"ev-{int(time.time()*1000)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, separators=(",", ":"))


EventFilter = Callable[[FleetEvent], bool]
EventHandler = Callable[[FleetEvent], Any]
AsyncEventHandler = Callable[[FleetEvent], Coroutine[Any, Any, Any]]


class FleetEventBus:
    """Central pub/sub bus for fleet-wide events."""

    def __init__(self) -> None:
        # event type -> list of (filter, handler, is_async)
        self._handlers: dict[str, list[tuple[EventFilter | None, EventHandler | AsyncEventHandler, bool]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: list[FleetEvent] = []
        self._history_limit = 1000

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the asyncio event loop for async handlers."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    # ── subscription API ────────────────────────────────────

    def on(
        self,
        event_type: str,
        handler: EventHandler | AsyncEventHandler,
        filter_fn: EventFilter | None = None,
        _is_async: bool = False,
    ) -> None:
        """Register a handler for ``event_type``.

        Optional ``filter_fn`` receives the event and returns ``True``
        if the handler should fire.
        """
        with self._lock:
            # Detect async handler
            is_async = _is_async or asyncio.iscoroutinefunction(handler)
            self._handlers[event_type].append((filter_fn, handler, is_async))
        logger.debug("Registered %s handler for %s", "async" if is_async else "sync", event_type)

    def on_async(
        self,
        event_type: str,
        handler: AsyncEventHandler,
        filter_fn: EventFilter | None = None,
    ) -> None:
        """Convenience: register an async handler."""
        self.on(event_type, handler, filter_fn=filter_fn, _is_async=True)

    def off(
        self,
        event_type: str,
        handler: EventHandler | AsyncEventHandler,
    ) -> bool:
        """Remove a handler.  Returns True if found and removed."""
        with self._lock:
            entries = self._handlers[event_type]
            for idx, (_f, h, _a) in enumerate(entries):
                if h is handler:
                    entries.pop(idx)
                    return True
            return False

    # ── publishing API ────────────────────────────────────

    def emit(
        self,
        event: dict[str, Any] | FleetEvent,
        source: str = "unknown",
    ) -> None:
        """Fire-and-forget synchronous emit.

        * If ``event`` is a dict, it must contain a ``type`` key.
        * If ``event`` is a ``FleetEvent``, it is used directly.
        * Async handlers are scheduled on the internal event loop.
        """
        if isinstance(event, dict):
            ev = FleetEvent(
                type=event.pop("type"),
                payload=event,
                source=source,
            )
        else:
            ev = event

        # Record in history
        with self._lock:
            self._history.append(ev)
            if len(self._history) > self._history_limit:
                self._history.pop(0)

        handlers = self._handlers.get(ev.type, [])
        if not handlers:
            logger.debug("No handlers for event type %s", ev.type)
            return

        for filt, handler, is_async in handlers:
            try:
                if filt is not None and not filt(ev):
                    continue
                if is_async:
                    loop = self._ensure_loop()
                    asyncio.run_coroutine_threadsafe(handler(ev), loop)
                else:
                    handler(ev)
            except Exception as exc:
                logger.exception("Handler failed for %s: %s", ev.type, exc)

    async def emit_async(
        self,
        event: dict[str, Any] | FleetEvent,
        source: str = "unknown",
    ) -> None:
        """Awaitable emit — useful inside async web servers."""
        if isinstance(event, dict):
            ev = FleetEvent(
                type=event.pop("type"),
                payload=event,
                source=source,
            )
        else:
            ev = event

        with self._lock:
            self._history.append(ev)
            if len(self._history) > self._history_limit:
                self._history.pop(0)

        handlers = self._handlers.get(ev.type, [])
        for filt, handler, is_async in handlers:
            try:
                if filt is not None and not filt(ev):
                    continue
                if is_async:
                    await handler(ev)
                else:
                    handler(ev)
            except Exception as exc:
                logger.exception("Handler failed for %s: %s", ev.type, exc)

    # ── introspection ─────────────────────────────────────

    def list_handlers(self, event_type: str | None = None) -> dict[str, int]:
        """Return a count of registered handlers per event type."""
        with self._lock:
            if event_type is not None:
                return {event_type: len(self._handlers.get(event_type, []))}
            return {k: len(v) for k, v in self._handlers.items()}

    def recent_events(
        self,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[FleetEvent]:
        """Return recent events, optionally filtered by type."""
        with self._lock:
            events = list(reversed(self._history))
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        return events[:limit]

    def stats(self) -> dict[str, Any]:
        """Bus health metrics."""
        with self._lock:
            return {
                "total_event_types": len(self._handlers),
                "total_handlers": sum(len(v) for v in self._handlers.values()),
                "history_size": len(self._history),
                "history_limit": self._history_limit,
                "event_types": list(self._handlers.keys()),
            }
