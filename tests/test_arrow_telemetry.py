"""Tests for Arrow Telemetry Adapter.

Covers schema validation, ring buffer rotation, batch export, zero-copy,
PLATO event mapping, SSE stream wiring.
"""

import pytest
import time
import json
from pathlib import Path

import numpy as np

pyarrow = pytest.importorskip("pyarrow")

from swarm.arrow_telemetry import (
    TelemetryEvent,
    build_arrow_schema,
    event_to_arrow_record,
    events_to_arrow_record,
    ArrowTelemetryBuffer,
    PLATOStreamAdapter,
    FleetAnalyticsSink,
    wire_to_fleet_conductor,
    wire_to_sse_dashboard,
    EVENT_TYPES,
)


# ---------------------------------------------------------------------------
# TelemetryEvent
# ---------------------------------------------------------------------------

class TestTelemetryEvent:
    def test_create_basic(self):
        ev = TelemetryEvent(
            timestamp=time.time(),
            event_type="AGENT_SPAWN",
            agent_id="agent-1",
            room_id="room-a",
        )
        assert ev.event_type == "AGENT_SPAWN"
        assert ev.agent_id == "agent-1"

    def test_invalid_event_type(self):
        with pytest.raises(ValueError):
            TelemetryEvent(
                timestamp=time.time(),
                event_type="NOT_REAL",
                agent_id="agent-1",
                room_id="room-a",
            )

    def test_payload_and_metrics(self):
        ev = TelemetryEvent(
            timestamp=time.time(),
            event_type="BREED",
            agent_id="agent-1",
            room_id="room-a",
            payload={"parent_ids": ["a", "b"]},
            metrics={"flux_score": 0.95, "thermal_level": 0.3},
        )
        assert ev.payload["parent_ids"] == ["a", "b"]
        assert ev.metrics["flux_score"] == 0.95


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_build_schema(self):
        schema = build_arrow_schema()
        names = schema.names
        assert "timestamp" in names
        assert "event_type" in names
        assert "agent_id" in names
        assert "room_id" in names
        assert "payload_json" in names
        assert "flux_score" in names

    def test_event_to_record(self):
        schema = build_arrow_schema()
        ev = TelemetryEvent(
            timestamp=time.time(),
            event_type="FLUX_GATE",
            agent_id="a1",
            room_id="r1",
            payload={"score": 0.9},
            metrics={"flux_score": 0.9, "latency_ms": 12.0},
        )
        batch = event_to_arrow_record(ev, schema)
        assert batch.num_rows == 1
        assert batch.schema.names == schema.names

    def test_events_to_record_batch(self):
        schema = build_arrow_schema()
        events = [
            TelemetryEvent(
                timestamp=time.time(),
                event_type="BEAT_TICK",
                agent_id="c1",
                room_id="fleet",
            )
            for _ in range(5)
        ]
        batch = events_to_arrow_record(events, schema)
        assert batch.num_rows == 5

    def test_empty_events_to_record(self):
        schema = build_arrow_schema()
        batch = events_to_arrow_record([], schema)
        assert batch.num_rows == 0


# ---------------------------------------------------------------------------
# ArrowTelemetryBuffer
# ---------------------------------------------------------------------------

class TestArrowTelemetryBuffer:
    def test_append_single(self):
        buf = ArrowTelemetryBuffer(max_batches=4, batch_size=2)
        buf.append(TelemetryEvent(
            timestamp=time.time(),
            event_type="AGENT_SPAWN",
            agent_id="a1",
            room_id="r1",
        ))
        assert buf.stats()["pending_events"] == 1

    def test_flush_on_batch_size(self):
        buf = ArrowTelemetryBuffer(max_batches=4, batch_size=2)
        for i in range(2):
            buf.append(TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id=f"a{i}",
                room_id="r1",
            ))
        assert buf.stats()["pending_events"] == 0
        assert buf.stats()["batch_count"] == 1

    def test_ring_buffer_rotation(self):
        buf = ArrowTelemetryBuffer(max_batches=2, batch_size=1)
        for i in range(4):
            buf.append(TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id=f"a{i}",
                room_id="r1",
            ))
        buf.flush()
        assert buf.stats()["batch_count"] == 2
        assert buf.stats()["total_events"] == 4
        assert buf.stats()["dropped_events"] == 2  # 2 batches of 1 dropped

    def test_to_table(self):
        buf = ArrowTelemetryBuffer(max_batches=4, batch_size=2)
        for i in range(4):
            buf.append(TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id=f"a{i}",
                room_id="r1",
            ))
        table = buf.to_table()
        assert table.num_rows == 4
        assert "timestamp" in table.column_names

    def test_slice(self):
        buf = ArrowTelemetryBuffer(max_batches=4, batch_size=2)
        for i in range(6):
            buf.append(TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id=f"a{i}",
                room_id="r1",
            ))
        sliced = buf.slice(0, 3)
        assert sliced.num_rows == 3

    def test_stats(self):
        buf = ArrowTelemetryBuffer()
        assert buf.stats()["total_events"] == 0
        assert buf.stats()["dropped_events"] == 0

    def test_len(self):
        buf = ArrowTelemetryBuffer()
        assert len(buf) == 0
        buf.append(TelemetryEvent(
            timestamp=time.time(),
            event_type="AGENT_SPAWN",
            agent_id="a1",
            room_id="r1",
        ))
        assert len(buf) == 1


# ---------------------------------------------------------------------------
# PLATOStreamAdapter
# ---------------------------------------------------------------------------

class TestPLATOStreamAdapter:
    def test_adapt_agent_spawn(self):
        adapter = PLATOStreamAdapter(room_id="test-room")
        plato = {
            "type": "agent_spawn",
            "timestamp": time.time(),
            "agent_id": "agent-42",
            "room_id": "test-room",
            "generation": 3,
        }
        ev = adapter.adapt(plato)
        assert ev.event_type == "AGENT_SPAWN"
        assert ev.agent_id == "agent-42"
        assert ev.payload["generation"] == 3

    def test_adapt_flux_violation(self):
        adapter = PLATOStreamAdapter()
        plato = {
            "type": "flux_violation",
            "timestamp": time.time(),
            "agent_id": "a1",
            "flux_score": 0.2,
            "thermal_level": 0.9,
        }
        ev = adapter.adapt(plato)
        assert ev.event_type == "FLUX_VIOLATION"
        assert ev.metrics["flux_score"] == 0.2
        assert ev.metrics["thermal_level"] == 0.9

    def test_adapt_unknown_type(self):
        adapter = PLATOStreamAdapter()
        plato = {
            "type": "unknown_event",
            "timestamp": time.time(),
            "agent_id": "a1",
        }
        ev = adapter.adapt(plato)
        assert ev.event_type == "INFO"

    def test_adapt_batch(self):
        adapter = PLATOStreamAdapter()
        plato_events = [
            {"type": "beat_tick", "timestamp": time.time(), "agent_id": "c1"},
            {"type": "error", "timestamp": time.time(), "agent_id": "a1"},
        ]
        events = adapter.adapt_batch(plato_events)
        assert len(events) == 2
        assert events[0].event_type == "BEAT_TICK"
        assert events[1].event_type == "ERROR"

    def test_default_room_id(self):
        adapter = PLATOStreamAdapter()
        plato = {"type": "info", "timestamp": time.time(), "agent_id": "a1"}
        ev = adapter.adapt(plato)
        assert ev.room_id == "default"


# ---------------------------------------------------------------------------
# FleetAnalyticsSink
# ---------------------------------------------------------------------------

class TestFleetAnalyticsSink:
    def test_init_creates_dir(self, tmp_path):
        sink = FleetAnalyticsSink(output_dir=tmp_path / "telemetry")
        assert (tmp_path / "telemetry").exists()

    def test_write_table(self, tmp_path):
        sink = FleetAnalyticsSink(output_dir=tmp_path / "telemetry", parquet_rotation=2)
        schema = build_arrow_schema()
        events = [
            TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id="a1",
                room_id="r1",
            )
            for _ in range(3)
        ]
        table = events_to_arrow_record(events, schema)
        sink.write_table(table)
        # Should not flush yet (3 < 2 rotation? actually 3 >= 2)
        sink.flush()
        files = list((tmp_path / "telemetry").glob("*.parquet"))
        assert len(files) >= 1

    def test_sse_callback(self, tmp_path):
        sink = FleetAnalyticsSink(output_dir=tmp_path / "telemetry")
        received = []
        def callback(event):
            received.append(event)
        sink.register_sse_callback(callback)
        schema = build_arrow_schema()
        table = events_to_arrow_record([
            TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id="a1",
                room_id="r1",
            )
        ], schema)
        sink.write_table(table)
        assert len(received) == 1
        assert received[0]["type"] == "ARROW_BATCH"

    def test_ipc_roundtrip(self, tmp_path):
        sink = FleetAnalyticsSink(output_dir=tmp_path / "telemetry")
        schema = build_arrow_schema()
        events = [
            TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id="a1",
                room_id="r1",
            )
            for _ in range(5)
        ]
        table = events_to_arrow_record(events, schema)
        ipc_path = tmp_path / "test.arrow"
        sink.export_ipc(table, ipc_path)
        read = sink.read_ipc(ipc_path)
        assert read.num_rows == 5
        assert read.column_names == table.column_names


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_wire_to_fleet_conductor(self):
        class MockConductor:
            def __init__(self):
                self.beat_count = 0
                self.identity = type("obj", (object,), {"agent_id": "cond-1"})()
            def beat(self):
                self.beat_count += 1

        conductor = MockConductor()
        buf = ArrowTelemetryBuffer()
        wire_to_fleet_conductor(conductor, buf)
        conductor.beat()
        assert conductor.beat_count == 1
        assert len(buf) == 1

    def test_wire_to_sse_dashboard(self):
        class MockDashboard:
            def __init__(self):
                self.events = []
            def publish(self, event):
                self.events.append(event)

        dashboard = MockDashboard()
        sink = FleetAnalyticsSink()
        wire_to_sse_dashboard(dashboard, sink)
        schema = build_arrow_schema()
        table = events_to_arrow_record([
            TelemetryEvent(
                timestamp=time.time(),
                event_type="AGENT_SPAWN",
                agent_id="a1",
                room_id="r1",
            )
        ], schema)
        sink.write_table(table)
        assert len(dashboard.events) == 1
        assert dashboard.events[0]["type"] == "ARROW_BATCH"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_buffer_concurrent_access(self):
        import threading
        buf = ArrowTelemetryBuffer(max_batches=10, batch_size=5)
        def append_many():
            for i in range(20):
                buf.append(TelemetryEvent(
                    timestamp=time.time(),
                    event_type="AGENT_SPAWN",
                    agent_id=f"t{i}",
                    room_id="r1",
                ))
        threads = [threading.Thread(target=append_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(buf) == 80

    def test_empty_adapter_batch(self):
        adapter = PLATOStreamAdapter()
        assert adapter.adapt_batch([]) == []
