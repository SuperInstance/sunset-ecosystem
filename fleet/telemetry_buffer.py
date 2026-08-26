"""Telemetry buffering and batching for efficient metric ingestion.

Buffers telemetry data points and flushes in batches to reduce write
overhead. Supports size-based and time-based flush triggers. Used for
fleet metric collection, log batching, and event buffering.

Usage:
    buf = TelemetryBuffer(max_size=100, flush_interval_sec=60)
    buf.record({"metric": "cpu", "value": 45.0})
    batch = buf.flush()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class TelemetryBuffer:
    """
    Telemetry buffer with batch flush.

    :param max_size: Max items before auto-flush.
    :param flush_interval_sec: Max seconds before auto-flush.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        max_size: int = 100,
        flush_interval_sec: float = 60.0,
        clock: Optional[callable] = None,
    ):
        self._max_size = max_size
        self._flush_interval = flush_interval_sec
        self._clock = clock or time.time
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = self._clock()
        self._total_flushed = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, data: Dict[str, Any]) -> bool:
        """
        Record a telemetry data point.

        :returns: True if buffer should be flushed.
        """
        self._buffer.append(
            {
                "timestamp": self._clock(),
                "data": data,
            }
        )
        return self.should_flush()

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def flush(self) -> List[Dict[str, Any]]:
        """
        Flush the buffer.

        :returns: List of buffered items.
        """
        batch = list(self._buffer)
        self._buffer.clear()
        self._last_flush = self._clock()
        self._total_flushed += len(batch)
        return batch

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        if len(self._buffer) >= self._max_size:
            return True
        elapsed = self._clock() - self._last_flush
        return elapsed >= self._flush_interval

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def time_since_flush(self) -> float:
        """Get seconds since last flush."""
        return self._clock() - self._last_flush

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._buffer),
            "max_size": self._max_size,
            "flush_interval": self._flush_interval,
            "total_flushed": self._total_flushed,
            "time_since_flush": self.time_since_flush(),
        }

    def __repr__(self) -> str:
        return f"<TelemetryBuffer size={len(self._buffer)}/{self._max_size}>"
