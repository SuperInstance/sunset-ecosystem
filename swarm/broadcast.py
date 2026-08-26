"""BroadcastingChannel — Pub/sub with Hebbian strengthening between agents."""

from __future__ import annotations

__all__ = ["BroadcastMessage", "BroadcastingChannel"]

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BroadcastMessage:
    """A message broadcast from one agent to subscribers.

    Attributes:
        content: The message payload.
        source_agent: Who sent this.
        target_room: Which room this relates to.
        relevance_score: How relevant this message is (0-1).
        timestamp: When it was sent.
    """

    content: Any
    source_agent: str = ""
    target_room: str = ""
    relevance_score: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        content_str = str(self.content)[:50]
        return (
            f"BroadcastMessage(from={self.source_agent!r}, "
            f"room={self.target_room!r}, rel={self.relevance_score:.2f}, "
            f"content={content_str!r}...)"
        )


class BroadcastingChannel:
    """Pub/sub channel with Hebbian strengthening.

    Agents subscribe to room patterns. When a message is broadcast,
    it reaches all subscribers. Messages that are "found useful"
    (positive reception) strengthen the source→subscriber channel.

    Args:
        max_queue_size: Maximum queued messages per subscriber.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        self._max_queue = max_queue_size
        # subscriber_id → list of room patterns they care about
        self._subscriptions: dict[str, list[str]] = defaultdict(list)
        # subscriber_id → message queue
        self._queues: dict[str, list[BroadcastMessage]] = defaultdict(list)
        # "src→dst" → Hebbian weight
        self._hebbian: dict[str, float] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"BroadcastingChannel(subs={len(self._subscriptions)}, "
            f"channels={len(self._hebbian)})"
        )

    def subscribe(self, agent_id: str, room_pattern: str) -> None:
        """Subscribe an agent to messages matching a room pattern.

        Args:
            agent_id: The subscribing agent.
            room_pattern: Room name or pattern to match.
        """
        with self._lock:
            if room_pattern not in self._subscriptions[agent_id]:
                self._subscriptions[agent_id].append(room_pattern)

    def unsubscribe(self, agent_id: str, room_pattern: str) -> None:
        """Unsubscribe from a room pattern."""
        with self._lock:
            if room_pattern in self._subscriptions.get(agent_id, []):
                self._subscriptions[agent_id].remove(room_pattern)

    def broadcast(self, message: BroadcastMessage) -> list[str]:
        """Broadcast a message to all matching subscribers.

        Returns list of subscriber IDs that received the message.
        """
        recipients: list[str] = []

        with self._lock:
            for agent_id, patterns in self._subscriptions.items():
                if any(message.target_room == p for p in patterns):
                    self._queues[agent_id].append(message)
                    # Trim queue if too long
                    if len(self._queues[agent_id]) > self._max_queue:
                        self._queues[agent_id] = self._queues[agent_id][
                            -self._max_queue :
                        ]
                    recipients.append(agent_id)

                    # Strengthen Hebbian channel
                    key = f"{message.source_agent}→{agent_id}"
                    current = self._hebbian.get(key, 0.1)
                    self._hebbian[key] = min(1.0, current + 0.01)

        return recipients

    def receive(self, agent_id: str) -> list[BroadcastMessage]:
        """Get all pending messages for an agent."""
        with self._lock:
            msgs = list(self._queues.get(agent_id, []))
            self._queues[agent_id] = []
            return msgs

    def feedback(self, source_agent: str, subscriber_id: str, useful: bool) -> None:
        """Report whether a broadcast was useful.

        Strengthens or weakens the Hebbian channel.
        """
        key = f"{source_agent}→{subscriber_id}"
        with self._lock:
            current = self._hebbian.get(key, 0.1)
            if useful:
                self._hebbian[key] = min(1.0, current + 0.05)
            else:
                self._hebbian[key] = max(0.0, current - 0.02)

    def get_channel_weight(self, source: str, subscriber: str) -> float:
        """Get the Hebbian weight between two agents."""
        key = f"{source}→{subscriber}"
        with self._lock:
            return self._hebbian.get(key, 0.0)

    @property
    def subscription_count(self) -> int:
        """Total subscriptions."""
        return sum(len(p) for p in self._subscriptions.values())
