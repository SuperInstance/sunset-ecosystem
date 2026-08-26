"""event_stream.py — Persistent event streaming with replay.

Provides:
1. Append-only event log
2. Event replay from any offset
3. Consumer groups with offset tracking
4. Event filtering by type/topic
5. TTL-based automatic eviction

Usage:
    stream = EventStream(max_events=10_000)
    stream.append({"type": "agent_spawn", "agent_id": "a1"})
    events = stream.replay(since_offset=0, event_type="agent_spawn")
"""

from __future__ import annotations

__all__ = [
    "EventStream",
    "Event",
    "ConsumerGroup",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A single event in the stream."""

    offset: int
    timestamp: float
    data: dict[str, Any]
    topic: str = "default"


@dataclass
class ConsumerGroup:
    """Consumer group offset tracking."""

    name: str
    offsets: dict[str, int] = field(
        default_factory=dict
    )  # topic -> last consumed offset


class EventStream:
    """Persistent event stream with replay and consumer groups."""

    def __init__(
        self, max_events: int = 10_000, ttl_seconds: float | None = None
    ) -> None:
        self._max_events = max_events
        self._ttl = ttl_seconds
        self._events: list[Event] = []
        self._offset = 0
        self._consumer_groups: dict[str, ConsumerGroup] = {}
        self._filters: list[Callable[[Event], bool]] = []

    def append(self, data: dict[str, Any], topic: str = "default") -> Event:
        """Append an event to the stream."""
        event = Event(
            offset=self._offset,
            timestamp=time.time(),
            data=data,
            topic=topic,
        )
        self._events.append(event)
        self._offset += 1

        # Evict old events if over limit
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

        # Evict expired events if TTL set
        if self._ttl is not None:
            cutoff = time.time() - self._ttl
            self._events = [e for e in self._events if e.timestamp > cutoff]

        return event

    def replay(
        self,
        since_offset: int = 0,
        topic: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Replay events from a given offset."""
        result = [
            e
            for e in self._events
            if e.offset >= since_offset
            and (topic is None or e.topic == topic)
            and (event_type is None or e.data.get("type") == event_type)
        ]
        if limit is not None:
            result = result[:limit]
        return result

    def latest(self, topic: str | None = None) -> Event | None:
        """Get the most recent event."""
        events = (
            self._events
            if topic is None
            else [e for e in self._events if e.topic == topic]
        )
        return events[-1] if events else None

    def create_consumer_group(self, name: str) -> ConsumerGroup:
        """Create a consumer group."""
        cg = ConsumerGroup(name=name)
        self._consumer_groups[name] = cg
        return cg

    def consume(
        self, group_name: str, topic: str = "default", limit: int = 100
    ) -> list[Event]:
        """Consume events for a consumer group (advances offset)."""
        cg = self._consumer_groups.get(group_name)
        if cg is None:
            raise ValueError(f"Consumer group '{group_name}' not found")

        since = cg.offsets.get(topic, 0)
        events = self.replay(since_offset=since, topic=topic, limit=limit)
        if events:
            cg.offsets[topic] = events[-1].offset + 1
        return events

    def consumer_offset(self, group_name: str, topic: str = "default") -> int:
        """Get current offset for a consumer group."""
        cg = self._consumer_groups.get(group_name)
        return cg.offsets.get(topic, 0) if cg else 0

    def reset_consumer_offset(
        self, group_name: str, topic: str = "default", offset: int = 0
    ) -> None:
        """Reset a consumer group offset."""
        cg = self._consumer_groups.get(group_name)
        if cg:
            cg.offsets[topic] = offset

    def event_count(self, topic: str | None = None) -> int:
        """Count events (optionally filtered by topic)."""
        if topic is None:
            return len(self._events)
        return sum(1 for e in self._events if e.topic == topic)

    def topics(self) -> list[str]:
        """List all topics."""
        return sorted({e.topic for e in self._events})

    def stats(self) -> dict[str, Any]:
        return {
            "total_events": len(self._events),
            "current_offset": self._offset,
            "topics": self.topics(),
            "consumer_groups": len(self._consumer_groups),
        }

    def __repr__(self) -> str:
        return f"EventStream(events={len(self._events)}, offset={self._offset})"
