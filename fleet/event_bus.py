from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class Event:
    """A fleet event."""

    event_type: str
    payload: Dict[str, Any]
    timestamp: float
    source: str
    event_id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }


class EventBus:
    """
    Pub-sub event bus for fleet-wide communication.

    Agents and services publish events; subscribers receive them.
    Supports pattern matching, filtering, and history.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: List[Event] = []
        self._max_history = 1000

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """Unsubscribe from an event type."""
        if event_type not in self._subscribers:
            return False
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            return True
        return False

    def publish(
        self, event_type: str, payload: Dict[str, Any], source: Optional[str] = None
    ) -> Event:
        """Publish an event to all subscribers."""
        event = Event(
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
            source=source or self.fleet_node_id,
        )
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Notify subscribers
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Subscriber errors shouldn't break the bus
                pass

        # Also notify wildcard subscribers
        wildcard_handlers = self._subscribers.get("*", [])
        for handler in wildcard_handlers:
            try:
                handler(event)
            except Exception as e:
                pass

        return event

    def get_history(
        self, event_type: Optional[str] = None, limit: int = 100
    ) -> List[Event]:
        """Get event history, optionally filtered by type."""
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "subscribers": {k: len(v) for k, v in self._subscribers.items()},
            "history_size": len(self._history),
            "event_types": list(set(e.event_type for e in self._history)),
        }

    def export_json(self) -> str:
        """Export event history as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "history": [e.to_dict() for e in self._history[-100:]],
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
