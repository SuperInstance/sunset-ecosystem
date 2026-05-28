"""Tests for SSE Stream Dashboard (fleet/sse_stream_dashboard.py).

Covers:
    - Event creation and SSE serialization
    - Publish/subscribe fan-out
    - History buffer
    - Backpressure handling
    - Heartbeat thread
    - Filtering by event type
    - Metrics
    - Integration wiring
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from fleet.sse_stream_dashboard import (
    DashboardConfig,
    DashboardServer,
    EventType,
    SSEStreamDashboard,
    StreamEvent,
    serve_dashboard_ui,
    wire_to_breeder,
    wire_to_fleet_conductor,
)


# ── 1. StreamEvent ────────────────────────────────────────

class TestStreamEvent:
    def test_to_sse_format(self):
        ev = StreamEvent(
            event_type=EventType.BEAT,
            payload={"n": 1},
            timestamp=1234.5,
            node_id="node-a",
        )
        sse = ev.to_sse()
        assert sse.startswith("data: ")
        assert "BEAT" in sse
        assert "node-a" in sse
        assert "1234.5" in sse

    def test_to_sse_is_json(self):
        ev = StreamEvent(EventType.INFO, payload={"msg": "hello"})
        sse = ev.to_sse()
        import json
        data = json.loads(sse.replace("data: ", "").strip())
        assert data["type"] == "INFO"
        assert data["payload"]["msg"] == "hello"


# ── 2. Dashboard basics ───────────────────────────────────

class TestDashboardBasics:
    def test_publish_and_subscribe(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        ev = StreamEvent(EventType.BEAT, payload={"n": 1})
        dash.publish(ev)
        received = sub.get(timeout=1.0)
        assert received.event_type == EventType.BEAT
        assert received.payload == {"n": 1}

    def test_multiple_subscribers_receive(self):
        dash = SSEStreamDashboard()
        sub1 = dash.subscribe()
        sub2 = dash.subscribe()
        dash.publish(StreamEvent(EventType.INFO, payload={"x": 1}))
        assert sub1.get(timeout=1.0).payload["x"] == 1
        assert sub2.get(timeout=1.0).payload["x"] == 1

    def test_unsubscribe_removes(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        dash.unsubscribe(sub)
        dash.publish(StreamEvent(EventType.INFO, payload={"x": 1}))
        with pytest.raises(queue.Empty):
            sub.get(timeout=0.1)

    def test_publish_simple(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        dash.publish_simple(EventType.THERMAL, {"temp": 42.0}, node_id="n1")
        ev = sub.get(timeout=1.0)
        assert ev.event_type == EventType.THERMAL
        assert ev.payload["temp"] == 42.0
        assert ev.node_id == "n1"


# ── 3. History ────────────────────────────────────────────

class TestHistory:
    def test_recent_events(self):
        dash = SSEStreamDashboard()
        for i in range(5):
            dash.publish(StreamEvent(EventType.BEAT, payload={"n": i}))
        recent = dash.recent_events(n=3)
        assert len(recent) == 3
        assert recent[-1].payload["n"] == 4

    def test_recent_by_type(self):
        dash = SSEStreamDashboard()
        dash.publish(StreamEvent(EventType.BEAT, payload={"t": "beat"}))
        dash.publish(StreamEvent(EventType.THERMAL, payload={"t": "thermal"}))
        dash.publish(StreamEvent(EventType.BEAT, payload={"t": "beat2"}))
        beats = dash.recent_by_type(EventType.BEAT, n=10)
        assert len(beats) == 2

    def test_history_respects_buffer_size(self):
        cfg = DashboardConfig(history_buffer_size=3)
        dash = SSEStreamDashboard(cfg)
        for i in range(5):
            dash.publish(StreamEvent(EventType.BEAT, payload={"n": i}))
        assert len(dash._history) == 3
        assert dash._history[0].payload["n"] == 2


# ── 4. Backpressure ───────────────────────────────────────

class TestBackpressure:
    def test_queue_full_drops_when_enabled(self):
        cfg = DashboardConfig(max_queue_size=2, enable_backpressure=True)
        dash = SSEStreamDashboard(cfg)
        # Fill queue
        dash.publish(StreamEvent(EventType.INFO, payload={"a": 1}))
        dash.publish(StreamEvent(EventType.INFO, payload={"b": 2}))
        # Third should drop
        ok = dash.publish(StreamEvent(EventType.INFO, payload={"c": 3}))
        assert ok is False

    def test_queue_full_evicts_oldest_when_disabled(self):
        cfg = DashboardConfig(max_queue_size=2, enable_backpressure=False)
        dash = SSEStreamDashboard(cfg)
        dash.publish(StreamEvent(EventType.INFO, payload={"a": 1}))
        dash.publish(StreamEvent(EventType.INFO, payload={"b": 2}))
        ok = dash.publish(StreamEvent(EventType.INFO, payload={"c": 3}))
        assert ok is True


# ── 5. Filtering ──────────────────────────────────────────

class TestFiltering:
    def test_filter_event_types(self):
        cfg = DashboardConfig(filter_event_types=["BEAT", "THERMAL"])
        dash = SSEStreamDashboard(cfg)
        sub = dash.subscribe()
        dash.publish(StreamEvent(EventType.BEAT, payload={"x": 1}))
        dash.publish(StreamEvent(EventType.INFO, payload={"x": 2}))
        dash.publish(StreamEvent(EventType.THERMAL, payload={"x": 3}))
        # Should only receive BEAT and THERMAL
        ev1 = sub.get(timeout=1.0)
        ev2 = sub.get(timeout=1.0)
        assert ev1.event_type == EventType.BEAT
        assert ev2.event_type == EventType.THERMAL
        with pytest.raises(queue.Empty):
            sub.get(timeout=0.1)


# ── 6. Heartbeat ──────────────────────────────────────────

class TestHeartbeat:
    def test_heartbeat_publishes(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        dash.start_heartbeat()
        time.sleep(0.3)  # wait for at least one heartbeat
        dash.stop_heartbeat()
        # Should have received at least one heartbeat
        found = False
        while True:
            try:
                ev = sub.get(timeout=0.5)
                if ev.event_type == EventType.INFO and ev.payload.get("message") == "heartbeat":
                    found = True
                    break
            except queue.Empty:
                break
        assert found


# ── 7. Metrics ────────────────────────────────────────────

class TestMetrics:
    def test_get_metrics(self):
        dash = SSEStreamDashboard()
        m = dash.get_metrics()
        assert "subscribers" in m
        assert "queue_depth" in m
        assert "history_size" in m
        assert m["subscribers"] == 0

    def test_metrics_after_subscribe(self):
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        m = dash.get_metrics()
        assert m["subscribers"] == 1


# ── 8. Integration wiring ─────────────────────────────────

class TestIntegrationWiring:
    def test_wire_to_fleet_conductor(self):
        class FakeConductor:
            def __init__(self):
                self.config = type("Config", (), {"node_id": "test-node"})()
                self._beat_count = 0

            def beat(self):
                self._beat_count += 1
                return {"beat_number": self._beat_count}

            def get_status(self):
                return {"status": "ok"}

        conductor = FakeConductor()
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        wire_to_fleet_conductor(dash, conductor)

        conductor.beat()
        ev = sub.get(timeout=1.0)
        assert ev.event_type == EventType.BEAT
        assert ev.payload["beat_number"] == 1

    def test_wire_to_breeder(self):
        class FakeBreeder:
            def cycle(self, n_winners: int = 3):
                return ["winner_1", "winner_2"]

        breeder = FakeBreeder()
        dash = SSEStreamDashboard()
        sub = dash.subscribe()
        wire_to_breeder(dash, breeder)

        breeder.cycle(n_winners=2)
        # Should get beat (start) and parent_select (end)
        ev1 = sub.get(timeout=1.0)
        ev2 = sub.get(timeout=1.0)
        assert {ev1.event_type, ev2.event_type} == {EventType.BEAT, EventType.PARENT_SELECT}


# ── 9. Dashboard Server ───────────────────────────────────

class TestDashboardServer:
    def test_server_serves_dashboard_html(self):
        dash = SSEStreamDashboard()
        server = DashboardServer(dash, host="127.0.0.1", port=0)  # 0 = auto-assign
        server.start()
        try:
            # Find the actual port
            actual_port = server._server.server_address[1]
            import urllib.request
            url = f"http://127.0.0.1:{actual_port}/dashboard"
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                body = resp.read().decode("utf-8")
                assert resp.status == 200
                assert "COCAPN FLEET" in body
                assert "EventSource" in body
                assert "/events" in body
        finally:
            server.stop()

    def test_server_returns_404_for_unknown_paths(self):
        dash = SSEStreamDashboard()
        server = DashboardServer(dash, host="127.0.0.1", port=0)
        server.start()
        try:
            actual_port = server._server.server_address[1]
            import urllib.request
            url = f"http://127.0.0.1:{actual_port}/notfound"
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(url, timeout=5.0)
            assert exc_info.value.code == 404
        finally:
            server.stop()

    def test_serve_dashboard_ui_convenience(self):
        dash = SSEStreamDashboard()
        server = serve_dashboard_ui(dash, host="127.0.0.1", port=0)
        try:
            actual_port = server._server.server_address[1]
            assert server.url == f"http://127.0.0.1:{actual_port}"
            import urllib.request
            url = f"http://127.0.0.1:{actual_port}/"
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                body = resp.read().decode("utf-8")
                assert "COCAPN FLEET" in body
        finally:
            server.stop()
