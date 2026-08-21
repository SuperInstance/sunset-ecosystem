"""Log shipping with batching and buffering.

Buffers log entries and ships them in batches to destinations. Supports
flush by size, flush by interval, and flush on demand. Used for fleet
log aggregation, remote logging, and audit log shipping.

Usage:
    shipper = LogShipper(batch_size=100, flush_interval_sec=5)
    shipper.append({"level": "info", "message": "hello"})
    batch = shipper.flush()  # Returns current batch
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class LogShipper:
    """
    Log shipper with batching and buffering.

    :param batch_size: Maximum entries per batch.
    :param flush_interval_sec: Auto-flush interval (None for manual only).
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval_sec: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._batch_size = batch_size
        self._flush_interval = flush_interval_sec
        self._clock = clock or time.time
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = self._clock()
        self._total_shipped = 0
        self._total_batches = 0

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, entry: Dict[str, Any]) -> None:
        """
        Append a log entry to the buffer.

        :param entry: Log entry dict.
        """
        self._buffer.append(entry)

    def extend(self, entries: List[Dict[str, Any]]) -> None:
        """Append multiple entries."""
        self._buffer.extend(entries)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        if len(self._buffer) >= self._batch_size:
            return True
        if self._flush_interval and self._buffer:
            elapsed = self._clock() - self._last_flush
            return elapsed >= self._flush_interval
        return False

    def flush(self) -> List[Dict[str, Any]]:
        """
        Flush current buffer and return batch.

        :returns: List of log entries.
        """
        if not self._buffer:
            return []
        batch = list(self._buffer)
        self._buffer.clear()
        self._last_flush = self._clock()
        self._total_shipped += len(batch)
        self._total_batches += 1
        return batch

    def flush_partial(self, max_entries: int) -> List[Dict[str, Any]]:
        """
        Flush up to max_entries from buffer.

        :param max_entries: Maximum entries to flush.
        :returns: Flushed entries.
        """
        if not self._buffer:
            return []
        count = min(max_entries, len(self._buffer))
        batch = self._buffer[:count]
        self._buffer = self._buffer[count:]
        self._last_flush = self._clock()
        self._total_shipped += len(batch)
        self._total_batches += 1
        return batch

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def buffer_size(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def peek(self, n: int = 1) -> List[Dict[str, Any]]:
        """Peek at first n entries without removing."""
        return list(self._buffer[:n])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._buffer),
            "batch_size": self._batch_size,
            "flush_interval": self._flush_interval,
            "total_shipped": self._total_shipped,
            "total_batches": self._total_batches,
            "last_flush_ago": self._clock() - self._last_flush,
        }

    def __repr__(self) -> str:
        return f"<LogShipper buffer={len(self._buffer)} shipped={self._total_shipped}>"
