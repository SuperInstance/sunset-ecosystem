"""Tests for websocket_bridge.py — WebSocket fleet dashboard bridge.

Run: python3 -m pytest tests/test_websocket_bridge.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.websocket_bridge import WSMessage, WebSocketBridge


class TestWSMessage:
    def test_create(self):
        m = WSMessage(type="event", payload={"x": 1})
        assert m.type == "event"
        assert m.payload["x"] == 1
        assert m.timestamp > 0


class TestWebSocketBridge:
    def test_create(self):
        ws = WebSocketBridge("ws://test:8765")
        assert ws.url == "ws://test:8765"
        assert not ws.connected

    def test_connect_disconnect(self):
        ws = WebSocketBridge("ws://test:8765")
        assert ws.connect() is True
        assert ws.connected
        ws.disconnect()
        assert not ws.connected

    def test_subscribe_and_dispatch(self):
        ws = WebSocketBridge("ws://test:8765")
        received = []
        ws.subscribe("breeding.*", lambda data: received.append(data))
        ws._dispatch("breeding.spawn", {"agent": "a1"})
        assert len(received) == 1
        assert received[0]["agent"] == "a1"

    def test_unsubscribe(self):
        ws = WebSocketBridge("ws://test:8765")
        ws.subscribe("topic", lambda d: None)
        ws.unsubscribe("topic")
        assert "topic" not in ws._subscribers

    def test_send_not_connected(self):
        ws = WebSocketBridge("ws://test:8765")
        assert ws.send({"type": "test"}) is False

    def test_send_connected(self):
        ws = WebSocketBridge("ws://test:8765")
        ws.connect()
        assert ws.send({"type": "test"}) is True
        assert ws._messages_sent == 1

    def test_heartbeat(self):
        ws = WebSocketBridge("ws://test:8765")
        ws.connect()
        assert ws.heartbeat() is True

    def test_subscribe_request(self):
        ws = WebSocketBridge("ws://test:8765")
        ws.connect()
        assert ws.subscribe_request(["a", "b"]) is True
        assert ws._messages_sent == 1

    def test_is_alive_connected(self):
        ws = WebSocketBridge("ws://test:8765")
        ws.connect()
        assert ws.is_alive() is True

    def test_is_alive_disconnected(self):
        ws = WebSocketBridge("ws://test:8765")
        assert ws.is_alive() is False

    def test_reconnect(self):
        ws = WebSocketBridge("ws://test:8765")
        ws._connected = False
        ws._reconnect_attempts = 0
        # reconnect() sleeps then calls connect() which resets attempts
        assert ws.reconnect() is True
        assert ws._reconnect_attempts == 0  # reset by connect()

    def test_metrics(self):
        ws = WebSocketBridge("ws://test:8765")
        ws.connect()
        ws.subscribe("a", lambda d: None)
        m = ws.metrics()
        assert m["connected"] is True
        assert m["subscribers"] == 1
        assert "a" in m["topics"]

    def test_on_message_event(self):
        ws = WebSocketBridge("ws://test:8765")
        received = []
        ws.subscribe("test", lambda d: received.append(d))
        ws._on_message('{"type": "event", "topic": "test", "data": {"x": 1}}')
        assert len(received) == 1
        assert received[0]["x"] == 1

    def test_on_message_heartbeat(self):
        ws = WebSocketBridge("ws://test:8765")
        ws._last_heartbeat = 0.0
        ws._on_message('{"type": "heartbeat", "timestamp": 12345}')
        assert ws._last_heartbeat > 0.0

    def test_on_message_invalid_json(self):
        ws = WebSocketBridge("ws://test:8765")
        # Should not crash
        ws._on_message("not json")
        assert ws._messages_received == 1

    def test_dispatch_multiple_handlers(self):
        ws = WebSocketBridge("ws://test:8765")
        a, b = [], []
        ws.subscribe("topic", lambda d: a.append(1))
        ws.subscribe("topic", lambda d: b.append(1))
        ws._dispatch("topic", {})
        assert len(a) == 1
        assert len(b) == 1

    def test_dispatch_wildcard(self):
        ws = WebSocketBridge("ws://test:8765")
        received = []
        ws.subscribe("breeding.*", lambda d: received.append(d))
        ws._dispatch("breeding.spawn", {"x": 1})
        ws._dispatch("mesh.update", {"y": 2})
        assert len(received) == 1

    def test_repr(self):
        ws = WebSocketBridge("ws://test:8765")
        assert "WebSocketBridge" in repr(ws)
