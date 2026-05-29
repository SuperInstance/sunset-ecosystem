"""Dead letter queue for failed message processing.

Captures messages that failed processing after retry exhaustion.
Supports replay, inspection, and TTL-based cleanup. Used for fleet
message bus fault tolerance.

Usage:
    dlq = DeadLetterQueue(ttl_sec=3600, max_size=1000)
    dlq.enqueue("msg-1", error="timeout", payload={"x": 1})
    failed = dlq.list_failed()
    dlq.replay("msg-1", processor_fn)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DeadLetter:
    """A failed message entry."""

    message_id: str
    error: str
    payload: Dict[str, Any]
    timestamp: float
    retry_count: int = 0


class DeadLetterQueue:
    """
    Fixed-capacity dead letter queue with TTL and replay.

    :param ttl_sec: Time-to-live for entries before auto-cleanup.
    :param max_size: Maximum entries to retain.
    """

    def __init__(self, ttl_sec: float = 3600.0, max_size: int = 1000):
        self._ttl = ttl_sec
        self._max_size = max_size
        self._queue: deque = deque()
        self._index: Dict[str, DeadLetter] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def enqueue(
        self,
        message_id: str,
        error: str,
        payload: Dict[str, Any],
        retry_count: int = 0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add a failed message to the queue."""
        now = timestamp or time.time()
        entry = DeadLetter(
            message_id=message_id,
            error=error,
            payload=payload,
            timestamp=now,
            retry_count=retry_count,
        )
        self._queue.append(entry)
        self._index[message_id] = entry
        self._evict_old(now)
        if len(self._queue) > self._max_size:
            oldest = self._queue.popleft()
            self._index.pop(oldest.message_id, None)

    def dequeue(self, message_id: str) -> Optional[DeadLetter]:
        """Remove and return a message by ID."""
        entry = self._index.pop(message_id, None)
        if entry and entry in self._queue:
            self._queue.remove(entry)
        return entry

    def get(self, message_id: str) -> Optional[DeadLetter]:
        """Get a message without removing."""
        return self._index.get(message_id)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self, message_id: str, processor: Callable[[Dict[str, Any]], bool]
    ) -> bool:
        """
        Replay a dead letter through a processor.

        :param processor: Function taking payload, returns True on success.
        :returns: True if replay succeeded and message was removed.
        """
        entry = self.get(message_id)
        if not entry:
            return False
        try:
            success = processor(entry.payload)
            if success:
                self.dequeue(message_id)
            return success
        except Exception:
            return False

    def replay_all(
        self, processor: Callable[[Dict[str, Any]], bool]
    ) -> Dict[str, bool]:
        """Replay all dead letters. Returns {message_id: success}."""
        results: Dict[str, bool] = {}
        for entry in list(self._queue):
            results[entry.message_id] = self.replay(entry.message_id, processor)
        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._ttl
        while self._queue and self._queue[0].timestamp < cutoff:
            oldest = self._queue.popleft()
            self._index.pop(oldest.message_id, None)

    def purge(self) -> None:
        """Clear all entries."""
        self._queue.clear()
        self._index.clear()

    def size(self) -> int:
        return len(self._queue)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_failed(self) -> List[Dict[str, Any]]:
        """Return all failed messages as dicts."""
        return [
            {
                "message_id": e.message_id,
                "error": e.error,
                "timestamp": e.timestamp,
                "retry_count": e.retry_count,
            }
            for e in self._queue
        ]

    def errors_by_type(self) -> Dict[str, int]:
        """Count failures by error type."""
        counts: Dict[str, int] = {}
        for e in self._queue:
            counts[e.error] = counts.get(e.error, 0) + 1
        return counts

    def __repr__(self) -> str:
        return f"<DeadLetterQueue size={len(self._queue)} max={self._max_size}>"
