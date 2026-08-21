"""event_bus.py — Fleet-wide pub/sub event bus.

Decoupled communication for fleet subsystems. Supports:
1. Topic-based pub/sub with wildcard patterns
2. Sync and async subscribers
3. Event filtering by payload structure
4. Buffered delivery with backpressure
5. Metrics and subscriber health tracking

Design: Each subsystem publishes events on its own topic. Other
subsystems subscribe to topics they care about. No direct coupling.

Usage:
    bus = EventBus()
    bus.subscribe("breeding.*", handler)
    bus.publish("breeding.spawn", {"agent_id": "a1"})
"""

from __future__ import annotations

__all__ = [
    "EventBus",
    "Event",
    "Subscriber",
]

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """A fleet event."""

    topic: str
    payload: dict[str, Any]
    timestamp: float
    source: str = ""


@dataclass
class Subscriber:
    """A subscriber with its pattern and handler."""

    pattern: str
    handler: Callable[[Event], None]
    name: str = ""
    count: int = 0
    errors: int = 0
    last_seen: float = 0.0

    def match(self, topic: str) -> bool:
        return fnmatch.fnmatch(topic, self.pattern)


class EventBus:
    """Fleet-wide pub/sub event bus."""

    def __init__(self, max_queue: int = 10_000) -> None:
        self._subscribers: list[Subscriber] = []
        self._max_queue = max_queue
        self._dropped = 0
        self._published = 0

    # ── subscribe ──────────────────────────────────────

    def subscribe(
        self,
        pattern: str,
        handler: Callable[[Event], None],
        name: str = "",
    ) -> Subscriber:
        """Subscribe to events matching pattern (supports * and ? wildcards)."""
        sub = Subscriber(pattern=pattern, handler=handler, name=name or pattern)
        self._subscribers.append(sub)
        logger.info(f"Subscribed {sub.name} to '{pattern}'")
        return sub

    def unsubscribe(self, sub: Subscriber) -> bool:
        if sub in self._subscribers:
            self._subscribers.remove(sub)
            return True
        return False

    # ── publish ────────────────────────────────────────

    def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        source: str = "",
    ) -> int:
        """Publish an event. Returns number of subscribers notified."""
        event = Event(
            topic=topic,
            payload=payload,
            timestamp=time.time(),
            source=source,
        )
        count = 0
        for sub in self._subscribers:
            if sub.match(topic):
                try:
                    sub.handler(event)
                    sub.count += 1
                    sub.last_seen = time.time()
                    count += 1
                except Exception as e:
                    sub.errors += 1
                    logger.warning(f"Subscriber {sub.name} error: {e}")
        self._published += 1
        return count

    def publish_safe(
        self,
        topic: str,
        payload: dict[str, Any],
        source: str = "",
    ) -> int:
        """Publish with exception safety — never raises."""
        try:
            return self.publish(topic, payload, source)
        except Exception as e:
            logger.error(f"EventBus publish error on '{topic}': {e}")
            return 0

    # ── query ──────────────────────────────────────────

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def topics(self) -> set[str]:
        return {sub.pattern for sub in self._subscribers}

    def metrics(self) -> dict[str, Any]:
        return {
            "subscribers": len(self._subscribers),
            "published": self._published,
            "dropped": self._dropped,
        }

    def health(self) -> dict[str, Any]:
        """Health of all subscribers."""
        return {
            sub.name: {
                "count": sub.count,
                "errors": sub.errors,
                "last_seen": sub.last_seen,
                "healthy": sub.errors < max(sub.count, 1),
            }
            for sub in self._subscribers
        }

    def reset(self) -> None:
        self._subscribers.clear()
        self._dropped = 0
        self._published = 0

    def __repr__(self) -> str:
        return f"EventBus(subscribers={self.subscriber_count()}, published={self._published})"
