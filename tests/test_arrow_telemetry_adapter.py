"""Tests for Arrow Telemetry Adapter.

Covers event buffering, batch creation, IPC serialization, and fallback behavior.
"""

import json
import time

import pytest

from fleet.arrow_telemetry_adapter import (
    ArrowTelemetryAdapter,
    HAS_PYARROW,
    TelemetryRow,
)
from fleet.sse_stream_dashboard import SSEStreamDashboard, StreamEvent, EventType


# ---------------------------------------------------------------------------
# TelemetryRow
# ---------------------------------------------------------------------------


class TestTelemetryRow:
    def test_from_event_beat(self):
        event = StreamEvent(
            EventType.BEAT,
            {"tick": 7, "thermal": 0.75, "energy": 0.9},
            timestamp=1234567890.0,
            node_id="node-a",
        )
        row = TelemetryRow.from_event(event)
        assert row.event_type == "BEAT"
        assert row.timestamp == 1234567890.0
        assert row.node_id == "node-a"
        assert row.tick == 7
        assert row.thermal == 0.75
        assert row.energy == 0.9
        assert "tick" in row.payload_json

    def test_from_event_defaults(self):
        event = StreamEvent(EventType.INFO, {"msg": "hello"})
        row = TelemetryRow.from_event(event)
        assert row.tick == 0
        assert row.thermal == 0.0
        assert row.energy == 0.0
        assert row.status == ""

    def test_from_event_flux_gate(self):
        event = StreamEvent(
            EventType.FLUX_GATE,
            {"status": "passed", "thermal": 0.6, "energy": 0.8},
        )
        row = TelemetryRow.from_event(event)
        assert row.event_type == "FLUX_GATE"
        assert row.status == "passed"


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    def test_start_stop(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        assert adapter._running is True
        assert adapter._thread is not None
        adapter.stop()
        assert adapter._running is False

    def test_subscriber_added(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        assert adapter._subscriber_queue in dash._subscribers
        adapter.stop()
        assert adapter._subscriber_queue not in dash._subscribers


# ---------------------------------------------------------------------------
# Buffering
# ---------------------------------------------------------------------------


class TestBuffering:
    def test_events_buffered(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash, batch_size=10)
        adapter.start()
        try:
            dash.publish(StreamEvent(EventType.BEAT, {"tick": 1, "thermal": 0.5}))
            time.sleep(0.1)
            assert adapter.get_buffer_count() >= 1
        finally:
            adapter.stop()

    def test_multiple_events(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash, batch_size=50)
        adapter.start()
        try:
            for i in range(5):
                dash.publish(
                    StreamEvent(EventType.BEAT, {"tick": i, "thermal": 0.5 + i * 0.1})
                )
            time.sleep(0.2)
            assert adapter.get_buffer_count() >= 5
        finally:
            adapter.stop()

    def test_drain_batch_reduces_count(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash, batch_size=10)
        adapter.start()
        try:
            for i in range(5):
                dash.publish(StreamEvent(EventType.BEAT, {"tick": i}))
            time.sleep(0.2)
            count_before = adapter.get_buffer_count()
            adapter.drain_batch()
            count_after = adapter.get_buffer_count()
            assert count_after < count_before or count_after == 0
        finally:
            adapter.stop()

    def test_drain_batch_none_when_empty(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        try:
            result = adapter.drain_batch()
            assert result is None
        finally:
            adapter.stop()


# ---------------------------------------------------------------------------
# Arrow batch output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestArrowBatch:
    def test_batch_has_correct_columns(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        try:
            dash.publish(
                StreamEvent(EventType.BEAT, {"tick": 1, "thermal": 0.75, "energy": 0.9})
            )
            time.sleep(0.1)
            batch = adapter.drain_batch()
            assert batch is not None
            assert batch.num_columns == 8
            assert batch.num_rows == 1
        finally:
            adapter.stop()

    def test_batch_values_correct(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        try:
            dash.publish(
                StreamEvent(
                    EventType.THERMAL,
                    {"tick": 3, "thermal": 0.85, "energy": 0.6, "status": "warn"},
                )
            )
            time.sleep(0.1)
            batch = adapter.drain_batch()
            assert batch is not None
            assert batch.column("event_type")[0].as_py() == "THERMAL"
            assert batch.column("tick")[0].as_py() == 3
            assert batch.column("thermal")[0].as_py() == pytest.approx(0.85, abs=0.01)
            assert batch.column("energy")[0].as_py() == pytest.approx(0.6, abs=0.01)
            assert batch.column("status")[0].as_py() == "warn"
        finally:
            adapter.stop()

    def test_drain_all(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        try:
            for i in range(3):
                dash.publish(StreamEvent(EventType.BEAT, {"tick": i}))
            time.sleep(0.2)
            batch = adapter.drain_all()
            assert batch is not None
            assert batch.num_rows == 3
        finally:
            adapter.stop()

    def test_batch_schema_fields(self):
        fields = ArrowTelemetryAdapter.schema_fields()
        assert "event_type" in fields
        assert "timestamp" in fields
        assert "thermal" in fields
        assert "energy" in fields


# ---------------------------------------------------------------------------
# IPC serialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestIPC:
    def test_to_ipc_bytes(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        try:
            dash.publish(StreamEvent(EventType.BEAT, {"tick": 1, "thermal": 0.5}))
            time.sleep(0.1)
            ipc_bytes = adapter.to_ipc_bytes()
            assert ipc_bytes is not None
            assert len(ipc_bytes) > 0
        finally:
            adapter.stop()

    def test_ipc_none_when_empty(self):
        dash = SSEStreamDashboard()
        adapter = ArrowTelemetryAdapter(dash)
        adapter.start()
        try:
            ipc_bytes = adapter.to_ipc_bytes()
            assert ipc_bytes is None
        finally:
            adapter.stop()


# ---------------------------------------------------------------------------
# JSON fallback
# ---------------------------------------------------------------------------


class TestJSONFallback:
    def test_drain_batch_returns_dicts_without_pyarrow(self):
        import fleet.arrow_telemetry_adapter as ata

        original = ata.HAS_PYARROW
        try:
            ata.HAS_PYARROW = False
            dash = SSEStreamDashboard()
            adapter = ArrowTelemetryAdapter(dash)
            adapter.start()
            try:
                dash.publish(StreamEvent(EventType.BEAT, {"tick": 1}))
                time.sleep(0.1)
                result = adapter.drain_batch()
                assert isinstance(result, list)
                assert result[0]["tick"] == 1
            finally:
                adapter.stop()
        finally:
            ata.HAS_PYARROW = original

    def test_ipc_bytes_json_without_pyarrow(self):
        import fleet.arrow_telemetry_adapter as ata

        original = ata.HAS_PYARROW
        try:
            ata.HAS_PYARROW = False
            dash = SSEStreamDashboard()
            adapter = ArrowTelemetryAdapter(dash)
            adapter.start()
            try:
                dash.publish(StreamEvent(EventType.BEAT, {"tick": 2}))
                time.sleep(0.1)
                ipc = adapter.to_ipc_bytes()
                assert isinstance(ipc, bytes)
                parsed = json.loads(ipc)
                assert parsed[0]["tick"] == 2
            finally:
                adapter.stop()
        finally:
            ata.HAS_PYARROW = original
