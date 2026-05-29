"""Async message bus with pub/sub, topics, and fan-out for fleet-wide events.

Decouples producers from consumers. Topics are hierarchical (room.trap.alpha).
Supports wildcard subscriptions, filtered delivery, and backpressure buffering.

Usage:
    bus = MessageBus()
    bus.subscribe("room.*.trap", handler)
    bus.publish("room.alpha.trap", {"event": "entered"})
    bus.run_sync()  # drain one event
"""
from __future__ import annotations

import fnmatch
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MessageBusError(Exception):
    pass


@dataclass
class Message:
    """A message on the bus."""

    topic: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    headers: Dict[str, str] = field(default_factory=dict)
    id: str = ""


class MessageBus:
    """
    Topic-based message bus with wildcard subscriptions.

    :param max_queue: Max in-flight messages before dropping oldest.
    """

    def __init__(self, max_queue: int = 10000):
        self._max_queue = max_queue
        self._queue: deque = deque()
        self._subscriptions: Dict[str, List[Callable[[Message], None]]] = {}
        self._wildcards: List[str] = []
        self._stats: Dict[str, int] = {"published": 0, "delivered": 0, "dropped": 0}

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, topic: str, payload: Any, headers: Optional[Dict[str, str]] = None) -> None:
        """Publish a message to a topic."""
        msg = Message(
            topic=topic,
            payload=payload,
            headers=headers or {},
        )
        if len(self._queue) >= self._max_queue:
            self._queue.popleft()
            self._stats["dropped"] += 1
        self._queue.append(msg)
        self._stats["published"] += 1

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe(self, pattern: str, handler: Callable[[Message], None]) -> None:
        """Subscribe to a topic pattern (supports * and ** wildcards)."""
        if pattern not in self._subscriptions:
            self._subscriptions[pattern] = []
            if "*" in pattern or "?" in pattern:
                self._wildcards.append(pattern)
        self._subscriptions[pattern].append(handler)

    def unsubscribe(self, pattern: str, handler: Callable[[Message], None]) -> None:
        """Remove a handler from a pattern."""
        if pattern in self._subscriptions:
            self._subscriptions[pattern] = [h for h in self._subscriptions[pattern] if h != handler]

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def run_sync(self, max_messages: int = 1) -> int:
        """Synchronous drain: process up to N messages. Returns count processed."""
        processed = 0
        while self._queue and processed < max_messages:
            msg = self._queue.popleft()
            handlers = self._match_handlers(msg.topic)
            for handler in handlers:
                try:
                    handler(msg)
                    self._stats["delivered"] += 1
                except Exception:
                    logger.exception("Message handler failed for topic=%s", msg.topic)
            processed += 1
        return processed

    def flush(self) -> int:
        """Process all queued messages. Returns count processed."""
        return self.run_sync(max_messages=len(self._queue))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def peek(self, n: int = 1) -> List[Message]:
        """Return next N messages without removing."""
        return list(self._queue)[:n]

    def queue_size(self) -> int:
        return len(self._queue)

    def subscriber_count(self, pattern: Optional[str] = None) -> int:
        if pattern:
            return len(self._subscriptions.get(pattern, []))
        return sum(len(h) for h in self._subscriptions.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_handlers(self, topic: str) -> List[Callable[[Message], None]]:
        """Find all handlers matching a topic."""
        handlers: List[Callable[[Message], None]] = []
        # Exact matches
        if topic in self._subscriptions:
            handlers.extend(self._subscriptions[topic])
        # Wildcard matches
        for pattern in self._wildcards:
            if fnmatch.fnmatch(topic, pattern):
                handlers.extend(self._subscriptions[pattern])
        return handlers

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"<MessageBus queue={len(self._queue)} subs={len(self._subscriptions)}>"
