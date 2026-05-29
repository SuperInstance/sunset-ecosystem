# Arrow Telemetry Adapter
# Zero-copy columnar streaming from PLATO → Arrow → Fleet analytics

"""Apache Arrow-based telemetry buffer for fleet-wide analytics.

This module provides zero-copy columnar streaming of PLATO room events
into Apache Arrow RecordBatches for efficient batch processing and
real-time dashboards.

References:
- Apache Arrow: https://arrow.apache.org/
- Plasma store: https://arrow.apache.org/docs/python/plasma.html
"""

from __future__ import annotations

import json
import time
import struct
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path

import numpy as np

# Optional pyarrow import with fallback
_ARR = None
_ARR_ERROR = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _ARR = pa
except ImportError as e:  # pragma: no cover
    _ARR_ERROR = e


def _arrow() -> Any:
    if _ARR is None:
        raise RuntimeError(f"pyarrow not available: {_ARR_ERROR}")
    return _ARR


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    "AGENT_SPAWN", "AGENT_DEATH", "BREED", "MUTATION",
    "FLUX_GATE", "FLUX_VIOLATION", "THERMAL_RISE", "THERMAL_FALL",
    "MESH_SYNC", "BEAT_TICK", "ERROR", "INFO",
]

PAYLOAD_KEYS = [
    "agent_id", "parent_ids", "room_id", "generation",
    "flux_score", "thermal_level", "vector_dim", "confidence",
    "latency_ms", "payload_json",
]


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    """Single PLATO telemetry event."""
    timestamp: float
    event_type: str
    agent_id: str
    room_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {self.event_type}")


# ---------------------------------------------------------------------------
# Arrow schema builder
# ---------------------------------------------------------------------------

def build_arrow_schema() -> Any:
    """Build the Arrow schema for fleet telemetry."""
    pa = _arrow()
    return pa.schema([
        ("timestamp", pa.timestamp("us")),
        ("event_type", pa.dictionary(pa.int8(), pa.string())),
        ("agent_id", pa.string()),
        ("room_id", pa.string()),
        ("payload_json", pa.string()),
        ("flux_score", pa.float64()),
        ("thermal_level", pa.float64()),
        ("latency_ms", pa.float64()),
    ])


def event_to_arrow_record(event: TelemetryEvent, schema: Any) -> Any:
    """Convert a single TelemetryEvent to an Arrow RecordBatch (1 row)."""
    pa = _arrow()
    payload_json = json.dumps(event.payload, default=str)
    return pa.RecordBatch.from_arrays(
        [
            pa.array([int(event.timestamp * 1_000_000)], type=pa.timestamp("us")),
            pa.array([event.event_type]),
            pa.array([event.agent_id]),
            pa.array([event.room_id]),
            pa.array([payload_json]),
            pa.array([float(event.metrics.get("flux_score", 0.0))]),
            pa.array([float(event.metrics.get("thermal_level", 0.0))]),
            pa.array([float(event.metrics.get("latency_ms", 0.0))]),
        ],
        schema=schema.names,
    )


def events_to_arrow_record(events: List[TelemetryEvent], schema: Any) -> Any:
    """Convert a list of TelemetryEvents to an Arrow RecordBatch."""
    pa = _arrow()
    if not events:
        return pa.RecordBatch.from_arrays(
            [pa.array([], t) for t in schema.types], schema=schema.names
        )
    return pa.RecordBatch.from_arrays(
        [
            pa.array([int(e.timestamp * 1_000_000) for e in events], type=pa.timestamp("us")),
            pa.array([e.event_type for e in events]),
            pa.array([e.agent_id for e in events]),
            pa.array([e.room_id for e in events]),
            pa.array([json.dumps(e.payload, default=str) for e in events]),
            pa.array([float(e.metrics.get("flux_score", 0.0)) for e in events]),
            pa.array([float(e.metrics.get("thermal_level", 0.0)) for e in events]),
            pa.array([float(e.metrics.get("latency_ms", 0.0)) for e in events]),
        ],
        schema=schema.names,
    )


# ---------------------------------------------------------------------------
# Arrow telemetry buffer (ring buffer)
# ---------------------------------------------------------------------------

class ArrowTelemetryBuffer:
    """Ring buffer of Arrow RecordBatches with configurable retention.

    Stores events in memory as Arrow RecordBatches. When the batch count
    exceeds *max_batches*, the oldest batch is discarded.
    """

    def __init__(
        self,
        max_batches: int = 64,
        batch_size: int = 1024,
    ) -> None:
        self._schema = build_arrow_schema()
        self._max_batches = max_batches
        self._batch_size = batch_size
        self._events: List[TelemetryEvent] = []
        self._batches: queue.Queue = queue.Queue(maxsize=max_batches)
        self._lock = threading.Lock()
        self._total_events = 0
        self._dropped_events = 0

    @property
    def schema(self) -> Any:
        return self._schema

    def append(self, event: TelemetryEvent) -> None:
        """Append a single event. Flush to a batch when buffer is full."""
        with self._lock:
            self._events.append(event)
            self._total_events += 1
            if len(self._events) >= self._batch_size:
                self._flush()

    def extend(self, events: List[TelemetryEvent]) -> None:
        """Append multiple events."""
        for ev in events:
            self.append(ev)

    def _flush(self) -> None:
        """Internal: flush pending events to a RecordBatch."""
        if not self._events:
            return
        batch = events_to_arrow_record(self._events, self._schema)
        self._events = []
        if self._batches.full():
            try:
                self._batches.get_nowait()
                self._dropped_events += self._batch_size
            except queue.Empty:
                pass
        self._batches.put(batch)

    def flush(self) -> None:
        """Force-flush pending events."""
        with self._lock:
            self._flush()

    def batches(self) -> List[Any]:
        """Return all batches as a list."""
        self.flush()
        with self._lock:
            return list(self._batches.queue)

    def to_table(self) -> Any:
        """Concatenate all batches into a single Arrow Table."""
        pa = _arrow()
        batches = self.batches()
        if not batches:
            return pa.Table.from_batches([], schema=self._schema)
        return pa.Table.from_batches(batches, schema=self._schema)

    def slice(self, start: int, end: int) -> Any:
        """Return a slice of the buffer as a Table."""
        table = self.to_table()
        return table.slice(start, end - start)

    def stats(self) -> Dict[str, int]:
        return {
            "total_events": self._total_events,
            "dropped_events": self._dropped_events,
            "pending_events": len(self._events),
            "batch_count": self._batches.qsize(),
        }

    def __len__(self) -> int:
        return self._total_events


# ---------------------------------------------------------------------------
# PLATO stream adapter
# ---------------------------------------------------------------------------

class PLATOStreamAdapter:
    """Converts PLATO room events to TelemetryEvents."""

    # Map PLATO event names to telemetry event types
    PLATO_TO_TELEMETRY: Dict[str, str] = {
        "agent_spawn": "AGENT_SPAWN",
        "agent_death": "AGENT_DEATH",
        "breed": "BREED",
        "mutation": "MUTATION",
        "flux_check": "FLUX_GATE",
        "flux_violation": "FLUX_VIOLATION",
        "thermal_rise": "THERMAL_RISE",
        "thermal_fall": "THERMAL_FALL",
        "mesh_sync": "MESH_SYNC",
        "beat_tick": "BEAT_TICK",
        "error": "ERROR",
        "info": "INFO",
    }

    def __init__(self, room_id: str = "default") -> None:
        self._room_id = room_id

    def adapt(self, plato_event: Dict[str, Any]) -> TelemetryEvent:
        """Convert a PLATO event dict to TelemetryEvent."""
        plato_type = plato_event.get("type", "info")
        event_type = self.PLATO_TO_TELEMETRY.get(plato_type, "INFO")
        payload = {k: v for k, v in plato_event.items() if k not in ("type", "timestamp", "agent_id")}
        metrics = {}
        if "flux_score" in plato_event:
            metrics["flux_score"] = float(plato_event["flux_score"])
        if "thermal_level" in plato_event:
            metrics["thermal_level"] = float(plato_event["thermal_level"])
        if "latency_ms" in plato_event:
            metrics["latency_ms"] = float(plato_event["latency_ms"])
        return TelemetryEvent(
            timestamp=plato_event.get("timestamp", time.time()),
            event_type=event_type,
            agent_id=plato_event.get("agent_id", "unknown"),
            room_id=plato_event.get("room_id", self._room_id),
            payload=payload,
            metrics=metrics,
        )

    def adapt_batch(self, plato_events: List[Dict[str, Any]]) -> List[TelemetryEvent]:
        return [self.adapt(ev) for ev in plato_events]


# ---------------------------------------------------------------------------
# Fleet analytics sink
# ---------------------------------------------------------------------------

class FleetAnalyticsSink:
    """Batch exporter for Arrow telemetry data.

    Supports Parquet, raw Arrow IPC, and SSE streaming.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        parquet_rotation: int = 10_000,  # rows per parquet file
    ) -> None:
        self._output_dir = output_dir or Path("telemetry")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._parquet_rotation = parquet_rotation
        self._parquet_buffer: List[Any] = []
        self._parquet_counter = 0
        self._lock = threading.Lock()
        self._sse_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register_sse_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._sse_callbacks.append(callback)

    def write_table(self, table: Any) -> None:
        """Write an Arrow Table to parquet and notify SSE."""
        pq = _arrow().parquet
        with self._lock:
            self._parquet_buffer.append(table)
            total_rows = sum(t.num_rows for t in self._parquet_buffer)
            if total_rows >= self._parquet_rotation:
                self._flush_parquet()
        # SSE notification (fire-and-forget)
        for cb in self._sse_callbacks:
            try:
                cb({
                    "type": "ARROW_BATCH",
                    "rows": table.num_rows,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    def _flush_parquet(self) -> None:
        pq = _arrow().parquet
        if not self._parquet_buffer:
            return
        combined = _arrow().Table.from_batches(
            [t.to_batches() for t in self._parquet_buffer]
        )
        path = self._output_dir / f"telemetry_{self._parquet_counter:06d}.parquet"
        pq.write_table(combined, path)
        self._parquet_buffer = []
        self._parquet_counter += 1

    def flush(self) -> None:
        with self._lock:
            self._flush_parquet()

    def snapshot(self) -> Path:
        """Force-flush and return latest parquet path."""
        self.flush()
        files = sorted(self._output_dir.glob("telemetry_*.parquet"))
        return files[-1] if files else self._output_dir

    def export_ipc(self, table: Any, path: Path) -> None:
        """Export table to Arrow IPC file (zero-copy mmap)."""
        with _arrow().ipc.new_file(path, schema=table.schema) as writer:
            writer.write_table(table)

    def read_ipc(self, path: Path) -> Any:
        """Read Arrow IPC file (memory-mapped)."""
        with _arrow().ipc.open_file(path) as reader:
            return reader.read_all()


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def wire_to_fleet_conductor(conductor: Any, buffer: ArrowTelemetryBuffer) -> None:
    """Wire an ArrowTelemetryBuffer to a FleetConductorV2 instance.

    Publishes Arrow batches on every conductor beat.
    """
    original_beat = conductor.beat

    def instrumented_beat() -> None:
        original_beat()
        buffer.append(TelemetryEvent(
            timestamp=time.time(),
            event_type="BEAT_TICK",
            agent_id=conductor.identity.agent_id if hasattr(conductor, "identity") else "conductor",
            room_id="fleet",
            metrics={"latency_ms": 0.0},
        ))

    conductor.beat = instrumented_beat


def wire_to_sse_dashboard(dashboard: Any, sink: FleetAnalyticsSink) -> None:
    """Wire FleetAnalyticsSink to SSEStreamDashboard.

    Serves Arrow-encoded SSE events via the dashboard's event bus.
    """
    def on_arrow_batch(event: Dict[str, Any]) -> None:
        dashboard.publish({
            "type": "ARROW_BATCH",
            "rows": event.get("rows", 0),
            "timestamp": event.get("timestamp", time.time()),
        })
    sink.register_sse_callback(on_arrow_batch)
