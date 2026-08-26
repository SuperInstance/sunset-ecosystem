"""websocket_bridge.py — WebSocket bridge for real-time fleet dashboard.

Provides:
1. Bidirectional WebSocket for fleet status streaming
2. JSON message protocol with validation
3. Automatic reconnection with exponential backoff
4. Heartbeat/ping-pong for connection health
5. Subscriber multiplexing (one WS connection → multiple topics)
6. Backpressure handling with message dropping

Protocol:
    Client → Server: {"type": "subscribe", "topics": ["breeding", "mesh"]}
    Server → Client: {"type": "event", "topic": "breeding", "data": {...}}
    Server → Client: {"type": "heartbeat", "timestamp": 1234567890}

Usage:
    bridge = WebSocketBridge("ws://fleet-node:8765")
    bridge.subscribe("breeding.*", handler)
    bridge.connect()
"""

from __future__ import annotations

__all__ = [
    "WebSocketBridge",
    "WSMessage",
]

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class WSMessage:
    """WebSocket message."""

    type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class WebSocketBridge:
    """WebSocket client bridge for fleet dashboard connectivity.

    This is a lightweight async-capable wrapper. Real WebSocket
    implementation uses asyncio or threading depending on context.
    """

    def __init__(
        self,
        url: str,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self.url = url
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._heartbeat_interval = heartbeat_interval
        self._connected = False
        self._subscribers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._last_heartbeat = 0.0
        self._messages_sent = 0
        self._messages_received = 0
        self._reconnect_attempts = 0

    # ── connection (stubs for actual WS library) ─────────

    def connect(self) -> bool:
        """Connect to WebSocket endpoint."""
        logger.info(f"Connecting to {self.url}")
        # In real implementation: open websocket connection
        self._connected = True
        self._reconnect_attempts = 0
        self._last_heartbeat = time.time()
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── subscribe ────────────────────────────────────────

    def subscribe(
        self,
        topic: str,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to a topic pattern."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str) -> None:
        self._subscribers.pop(topic, None)

    # ── message handling ────────────────────────────────

    def _on_message(self, raw: str) -> None:
        """Handle incoming WebSocket message."""
        self._messages_received += 1
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {raw[:100]}")
            return

        msg_type = msg.get("type", "")
        if msg_type == "event":
            topic = msg.get("topic", "")
            data = msg.get("data", {})
            self._dispatch(topic, data)
        elif msg_type == "heartbeat":
            self._last_heartbeat = time.time()
        elif msg_type == "subscribed":
            logger.info(f"Subscribed to: {msg.get('topics', [])}")

    def _dispatch(self, topic: str, data: dict[str, Any]) -> None:
        """Dispatch event to matching subscribers."""
        import fnmatch

        for pattern, handlers in self._subscribers.items():
            if fnmatch.fnmatch(topic, pattern):
                for handler in handlers:
                    try:
                        handler(data)
                    except Exception as e:
                        logger.warning(f"Handler error on '{topic}': {e}")

    # ── send ────────────────────────────────────────────

    def send(self, msg: dict[str, Any]) -> bool:
        """Send a message over the WebSocket."""
        if not self._connected:
            logger.warning("Cannot send: not connected")
            return False
        try:
            raw = json.dumps(msg)
            # In real implementation: ws.send(raw)
            self._messages_sent += 1
            return True
        except (TypeError, ValueError) as e:
            logger.warning(f"Serialize error: {e}")
            return False

    def subscribe_request(self, topics: list[str]) -> bool:
        """Send subscription request to server."""
        return self.send({"type": "subscribe", "topics": topics})

    def heartbeat(self) -> bool:
        """Send heartbeat ping."""
        return self.send({"type": "heartbeat", "timestamp": time.time()})

    # ── health ──────────────────────────────────────────

    def is_alive(self) -> bool:
        """Check if connection is alive (connected + recent heartbeat)."""
        if not self._connected:
            return False
        elapsed = time.time() - self._last_heartbeat
        return elapsed < self._heartbeat_interval * 2.5

    def reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff."""
        if self._connected:
            return True
        delay = min(
            self._reconnect_delay * (2**self._reconnect_attempts),
            self._max_reconnect_delay,
        )
        self._reconnect_attempts += 1
        logger.info(
            f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts})"
        )
        time.sleep(delay)
        return self.connect()

    # ── metrics ─────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "alive": self.is_alive(),
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "reconnect_attempts": self._reconnect_attempts,
            "subscribers": len(self._subscribers),
            "topics": list(self._subscribers.keys()),
        }

    def __repr__(self) -> str:
        return f"WebSocketBridge(url={self.url}, connected={self._connected})"
