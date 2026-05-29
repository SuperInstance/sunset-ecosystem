"""Tests for SSEStreamDashboard — real-time breeding progress streaming.

Covers StreamEvent, DashboardConfig, SSEStreamDashboard publish/subscribe,
history, metrics, heartbeat, and integration wiring.
"""

import queue
import time
from unittest.mock import MagicMock

import pytest

from fleet.sse_stream_dashboard import (
    DashboardConfig,
    DashboardServer,
    EventType,
    SSEStreamDashboard,
    StreamEvent,
    wire_to_breeder,
    wire_to_fleet_conductor,
)


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------

class TestStreamEvent:
    def test_to_sse(self):
        ev = StreamEvent(EventType.BEAT, {"n": 1}, timestamp=123.0, node_id="n1")
        sse = ev.to_sse()
        assert sse.startswith("data: ")
        assert "BEAT" in sse
        assert "n1" in sse

    def test_default_timestamp(self):
        ev = StreamEvent(EventType.INFO, {})
        assert ev.timestamp > 0

    def test_default_node_id(self):
        ev = StreamEvent(EventType.INFO, {})
        assert ev.node_id == "unknown"


# ---------------------------------------------------------------------------
# SSEStreamDashboard init
# ---------------------------------------------------------------------------

class TestDashboardInit:
    def test_defaults(self):
        dash = SSEStreamDashboard()
        assert dash.config.max_queue_size == 1000

    def test_custom_config(self):
        cfg = DashboardConfig(max_queue_size=10)
        dash = SSEStreamDashboard(cfg)
        assert dash.config.max_queue_size == 10


# ---------------------------------------------------------------------------
# Publish / Subscribe
# ---------------------------------------------------------------------------

class TestPublishSubscribe:
    def test_publish_and_receive(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        ev = StreamEvent(EventType.BEAT, {"tick": 1})
        assert dash.publish(ev) is True
        received = sub.get(timeout=1.0)
        assert received.event_type == EventType.BEAT

    def test_multiple_subscribers(self):
        dash = SSEStreamDashboard()
        sub1 = dash.subscribe()
        sub2 = dash.subscribe()
        dash.publish(StreamEvent(EventType.INFO, {"m": "x"}))
        assert sub1.get(timeout=1.0).payload["m"] == "x"
        assert sub2.get(timeout=1.0).payload["m"] == "x"

    def test_unsubscribe(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        dash.unsubscribe(sub)
        dash.publish(StreamEvent(EventType.INFO, {}))
        # subscriber removed, should not receive
        with pytest.raises(queue.Empty):
            sub.get(timeout=0.1)

    def test_publish_simple(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        dash.publish_simple(EventType.THERMAL, {"temp": 42})
        ev = sub.get(timeout=1.0)
        assert ev.event_type == EventType.THERMAL
        assert ev.payload["temp"] == 42

    def test_filter_event_types(self):
        cfg = DashboardConfig(filter_event_types=["BEAT"])
        dash = SSEStreamDashboard(cfg)
        sub = dash.subscribe()
        # INFO filtered out
        dash.publish(StreamEvent(EventType.INFO, {}))
        with pytest.raises(queue.Empty):
            sub.get(timeout=0.1)
        # BEAT passes
        dash.publish(StreamEvent(EventType.BEAT, {}))
        assert sub.get(timeout=1.0).event_type == EventType.BEAT


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_recent_events(self):
        dash = SSEStreamDashboard()
        for i in range(5):
            dash.publish(StreamEvent(EventType.BEAT, {"i": i}))
        recent = dash.recent_events(3)
        assert len(recent) == 3
        assert recent[-1].payload["i"] == 4

    def test_recent_by_type(self):
        dash = SSEStreamDashboard()
        dash.publish(StreamEvent(EventType.BEAT, {"i": 1}))
        dash.publish(StreamEvent(EventType.INFO, {"i": 2}))
        dash.publish(StreamEvent(EventType.BEAT, {"i": 3}))
        beats = dash.recent_by_type(EventType.BEAT, n=10)
        assert len(beats) == 2

    def test_history_buffer_size(self):
        cfg = DashboardConfig(history_buffer_size=3)
        dash = SSEStreamDashboard(cfg)
        for i in range(5):
            dash.publish(StreamEvent(EventType.BEAT, {"i": i}))
        assert len(dash._history) == 3
        assert dash.recent_events(10)[0].payload["i"] == 2

    def test_subscribe_gets_history(self):
        dash = SSEStreamDashboard()
        dash.publish(StreamEvent(EventType.BEAT, {"i": 1}))
        sub = dash.subscribe()
        # should receive historical event
        ev = sub.get(timeout=1.0)
        assert ev.payload["i"] == 1


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------

class TestBackpressure:
    def test_drops_when_full(self):
        cfg = DashboardConfig(max_queue_size=1, enable_backpressure=True)
        dash = SSEStreamDashboard(cfg)
        # fill the single slot
        dash.publish(StreamEvent(EventType.BEAT, {"i": 1}))
        # next should drop
        result = dash.publish(StreamEvent(EventType.BEAT, {"i": 2}))
        assert result is False

    def test_evict_oldest_when_disabled(self):
        cfg = DashboardConfig(max_queue_size=2, enable_backpressure=False)
        dash = SSEStreamDashboard(cfg)
        dash.publish(StreamEvent(EventType.BEAT, {"i": 1}))
        dash.publish(StreamEvent(EventType.BEAT, {"i": 2}))
        # third evicts first
        dash.publish(StreamEvent(EventType.BEAT, {"i": 3}))
        recent = dash.recent_events(10)
        assert recent[-1].payload["i"] == 3


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_basic(self):
        dash = SSEStreamDashboard()
        m = dash.get_metrics()
        assert m["subscribers"] == 0
        assert m["queue_depth"] == 0
        assert m["history_size"] == 0

    def test_with_subscriber(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        m = dash.get_metrics()
        assert m["subscribers"] == 1
        dash.unsubscribe(sub)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_start_stop(self):
        dash = SSEStreamDashboard()
        dash.config.heartbeat_interval_sec = 0.05
        dash.start_heartbeat()
        assert dash._running is True
        sub = dash.subscribe()
        ev = sub.get(timeout=1.0)
        assert ev.event_type == EventType.INFO
        dash.stop_heartbeat()

    def test_heartbeat_payload(self):
        dash = SSEStreamDashboard()
        dash.config.heartbeat_interval_sec = 0.05
        dash.start_heartbeat()
        sub = dash.subscribe()
        ev = sub.get(timeout=1.0)
        assert "heartbeat" in ev.payload.get("message", "")
        dash.stop_heartbeat()


# ---------------------------------------------------------------------------
# Integration wiring
# ---------------------------------------------------------------------------

class TestWireToFleetConductor:
    def test_instruments_beat(self):
        dash = SSEStreamDashboard()
        conductor = MagicMock()
        conductor.config.node_id = "test-node"
        conductor.beat.return_value = {"beat_number": 7}
        conductor.get_status.return_value = {"ok": True}
        wire_to_fleet_conductor(dash, conductor)
        result = conductor.beat()
        assert result["beat_number"] == 7
        # events should have been published
        events = dash.recent_events(10)
        assert any(e.event_type == EventType.BEAT for e in events)
        assert any(e.event_type == EventType.FLEET_STATUS for e in events)

    def test_filtered_types(self):
        dash = SSEStreamDashboard()
        conductor = MagicMock()
        conductor.config.node_id = "n1"
        conductor.beat.return_value = {}
        conductor.get_status.return_value = {}
        wire_to_fleet_conductor(dash, conductor, event_types=[EventType.BEAT])
        conductor.beat()
        events = dash.recent_events(10)
        assert all(e.event_type == EventType.BEAT for e in events)


class TestWireToBreeder:
    def test_instruments_cycle(self):
        dash = SSEStreamDashboard()
        breeder = MagicMock()
        breeder.cycle.return_value = ["a", "b"]
        wire_to_breeder(dash, breeder)
        result = breeder.cycle(2)
        assert result == ["a", "b"]
        events = dash.recent_events(10)
        assert any(e.event_type == EventType.BEAT for e in events)
        assert any(e.event_type == EventType.PARENT_SELECT for e in events)


# ---------------------------------------------------------------------------
# DashboardServer
# ---------------------------------------------------------------------------

class TestDashboardServer:
    def test_url_before_start(self):
        dash = SSEStreamDashboard()
        server = DashboardServer(dash, host="127.0.0.1", port=9999)
        assert server.url == "http://127.0.0.1:9999"

    def test_start_stop(self):
        dash = SSEStreamDashboard()
        server = DashboardServer(dash, host="127.0.0.1", port=0)
        server.start()
        assert server._server is not None
        assert server._thread is not None
        assert server.url.startswith("http://127.0.0.1:")
        server.stop()
        assert server._server is None
