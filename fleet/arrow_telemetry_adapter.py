"""fleet/arrow_telemetry_adapter.py — Arrow Telemetry Adapter for SSE Stream Dashboard.

Converts SSE StreamEvents into Apache Arrow RecordBatches for zero-copy,
columnar, GPU-ready telemetry.  Provides both streaming and batch APIs.

Usage
-----
    from fleet.arrow_telemetry_adapter import ArrowTelemetryAdapter
    from fleet.sse_stream_dashboard import SSEStreamDashboard, StreamEvent, EventType

    dash = SSEStreamDashboard()
    adapter = ArrowTelemetryAdapter(dash)
    adapter.start()

    dash.publish(StreamEvent(EventType.BEAT, {"tick": 1, "thermal": 0.75}))

    batch = adapter.drain_batch()  # pyarrow RecordBatch
    # batch.column('thermal') -> [0.75]

Features
--------
- Columnar schema: event_type, timestamp, node_id, tick, thermal, energy, status
- Zero-copy from Python dicts via pyarrow arrays
- JSON fallback when pyarrow unavailable
- GPU-ready: contiguous float32 buffers for thermal/energy
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from fleet.sse_stream_dashboard import SSEStreamDashboard, StreamEvent, EventType

logger = logging.getLogger(__name__)

# Optional pyarrow
try:
    import pyarrow as pa

    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore
    HAS_PYARROW = False
    logger.warning("pyarrow not available; arrow_telemetry using JSON fallback")


# ── Schema ────────────────────────────────────────────────────────────

TELEMETRY_SCHEMA = None
if HAS_PYARROW:
    TELEMETRY_SCHEMA = pa.schema(
        [
            ("event_type", pa.string()),
            ("timestamp", pa.float64()),
            ("node_id", pa.string()),
            ("tick", pa.int64()),
            ("thermal", pa.float32()),
            ("energy", pa.float32()),
            ("status", pa.string()),
            ("payload_json", pa.string()),
        ]
    )


# ── Telemetry Row ─────────────────────────────────────────────────────


@dataclass
class TelemetryRow:
    """Normalized row from a StreamEvent."""

    event_type: str = ""
    timestamp: float = 0.0
    node_id: str = ""
    tick: int = 0
    thermal: float = 0.0
    energy: float = 0.0
    status: str = ""
    payload_json: str = "{}"

    @classmethod
    def from_event(cls, event: StreamEvent) -> "TelemetryRow":
        p = event.payload
        return cls(
            event_type=event.event_type.name,
            timestamp=float(event.timestamp),
            node_id=event.node_id,
            tick=int(p.get("tick", 0)),
            thermal=float(p.get("thermal", 0.0)),
            energy=float(p.get("energy", 0.0)),
            status=str(p.get("status", "")),
            payload_json=json.dumps(p),
        )


# ── Adapter ───────────────────────────────────────────────────────────


@dataclass
class ArrowTelemetryAdapter:
    """Consumes SSE events and produces Arrow RecordBatches."""

    dashboard: SSEStreamDashboard
    batch_size: int = 100
    _buffer: List[TelemetryRow] = field(default_factory=list, repr=False)
    _buf_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _subscriber_queue: Optional[queue.Queue[StreamEvent]] = field(
        default=None, repr=False
    )
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _running: bool = False

    def start(self) -> None:
        """Start consuming events from the dashboard."""
        self._running = True
        self._subscriber_queue = queue.Queue()
        self.dashboard._subscribers.append(self._subscriber_queue)
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        logger.info("ArrowTelemetryAdapter started")

    def stop(self) -> None:
        """Stop consuming."""
        self._running = False
        if (
            self._subscriber_queue
            and self._subscriber_queue in self.dashboard._subscribers
        ):
            self.dashboard._subscribers.remove(self._subscriber_queue)
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("ArrowTelemetryAdapter stopped")

    def _consume(self) -> None:
        """Background thread: events -> buffer."""
        while self._running:
            try:
                event = self._subscriber_queue.get(timeout=0.5)
                row = TelemetryRow.from_event(event)
                with self._buf_lock:
                    self._buffer.append(row)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.warning("Telemetry consume error: %s", exc)

    def drain_batch(self) -> Optional[Any]:
        """Return an Arrow RecordBatch from the buffer, or None if empty."""
        with self._buf_lock:
            if not self._buffer:
                return None
            rows = self._buffer[: self.batch_size]
            self._buffer = self._buffer[self.batch_size :]

        if not HAS_PYARROW:
            return [r.__dict__ for r in rows]

        # Build Arrow arrays
        batch = pa.record_batch(
            [
                pa.array([r.event_type for r in rows]),
                pa.array([r.timestamp for r in rows], type=pa.float64()),
                pa.array([r.node_id for r in rows]),
                pa.array([r.tick for r in rows], type=pa.int64()),
                pa.array([r.thermal for r in rows], type=pa.float32()),
                pa.array([r.energy for r in rows], type=pa.float32()),
                pa.array([r.status for r in rows]),
                pa.array([r.payload_json for r in rows]),
            ],
            schema=TELEMETRY_SCHEMA,
        )
        return batch

    def drain_all(self) -> Optional[Any]:
        """Drain entire buffer into one RecordBatch."""
        with self._buf_lock:
            if not self._buffer:
                return None
            rows = self._buffer
            self._buffer = []
        if not HAS_PYARROW:
            return [r.__dict__ for r in rows]
        return pa.record_batch(
            [
                pa.array([r.event_type for r in rows]),
                pa.array([r.timestamp for r in rows], type=pa.float64()),
                pa.array([r.node_id for r in rows]),
                pa.array([r.tick for r in rows], type=pa.int64()),
                pa.array([r.thermal for r in rows], type=pa.float32()),
                pa.array([r.energy for r in rows], type=pa.float32()),
                pa.array([r.status for r in rows]),
                pa.array([r.payload_json for r in rows]),
            ],
            schema=TELEMETRY_SCHEMA,
        )

    def get_buffer_count(self) -> int:
        with self._buf_lock:
            return len(self._buffer)

    def to_ipc_bytes(self) -> Optional[bytes]:
        """Serialize current batch to Arrow IPC format."""
        batch = self.drain_all()
        if batch is None:
            return None
        if not HAS_PYARROW:
            return json.dumps(batch).encode("utf-8")
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
        return sink.getvalue().to_pybytes()

    @staticmethod
    def schema_fields() -> List[str]:
        """Return list of schema field names."""
        if HAS_PYARROW and TELEMETRY_SCHEMA:
            return [f.name for f in TELEMETRY_SCHEMA]
        return [
            "event_type",
            "timestamp",
            "node_id",
            "tick",
            "thermal",
            "energy",
            "status",
            "payload_json",
        ]
