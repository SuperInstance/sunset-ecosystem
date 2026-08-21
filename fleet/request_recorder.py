"""HTTP request/response recorder for debugging and replay.

Records requests and responses for later inspection, debugging, and
test fixture generation. Used for fleet API debugging and regression
testing.

Usage:
    rec = RequestRecorder(capacity=100)
    rec.record(method="GET", url="/api", status=200, response={"x": 1})
    entry = rec.get(0)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional


class RequestRecorder:
    """
    Fixed-capacity request/response recorder.

    :param capacity: Maximum entries to retain.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._entries: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        method: str,
        url: str,
        status: int = 0,
        request: Optional[Dict[str, Any]] = None,
        response: Optional[Any] = None,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record a request/response pair."""
        self._entries.append(
            {
                "timestamp": time.time(),
                "method": method,
                "url": url,
                "status": status,
                "request": request,
                "response": response,
                "error": error,
                "latency_ms": latency_ms,
            }
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, index: int) -> Optional[Dict[str, Any]]:
        """Get entry by index (0 = oldest)."""
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    def last(self) -> Optional[Dict[str, Any]]:
        """Get most recent entry."""
        return self._entries[-1] if self._entries else None

    def filter(
        self,
        method: Optional[str] = None,
        status_min: Optional[int] = None,
        url_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter entries by criteria."""
        results: List[Dict[str, Any]] = []
        for entry in self._entries:
            if method and entry["method"] != method:
                continue
            if status_min is not None and entry["status"] < status_min:
                continue
            if url_contains and url_contains not in entry["url"]:
                continue
            results.append(entry)
        return results

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = len(self._entries)
        errors = sum(1 for e in self._entries if e["status"] >= 400 or e["error"])
        return {
            "total": total,
            "errors": errors,
            "capacity": self._capacity,
        }

    def __repr__(self) -> str:
        return f"<RequestRecorder entries={len(self._entries)}>"
