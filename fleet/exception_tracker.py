"""Centralized exception tracking for fleet services.

Captures, deduplicates, and indexes exceptions across fleet nodes.
Supports filtering by service, exception type, and time range. Used
for fleet-wide error monitoring and debugging.

Usage:
    tracker = ExceptionTracker(capacity=1000)
    try:
        risky_operation()
    except Exception:
        tracker.record("my_service")
    recent = tracker.recent(limit=10)
"""
from __future__ import annotations

import hashlib
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional


class ExceptionTracker:
    """
    Fixed-capacity exception tracker with deduplication.

    :param capacity: Maximum exception entries to retain.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._entries: deque = deque(maxlen=capacity)
        self._seen: set = set()  # fingerprint set for dedup

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        service: str,
        exc_type: Optional[str] = None,
        exc_message: Optional[str] = None,
        traceback_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record current or specified exception.

        :param service: Service name.
        :param exc_type: Override exception type (auto-detected if not given).
        :param exc_message: Override message.
        :param traceback_text: Override traceback.
        :param context: Additional context dict.
        """
        if exc_type is None:
            exc_type_obj, exc_value, exc_tb = traceback.sys.exc_info()
            if exc_type_obj is None:
                return
            exc_type = exc_type_obj.__name__
            exc_message = str(exc_value) if exc_message is None else exc_message
            traceback_text = traceback.format_exc() if traceback_text is None else traceback_text
        self.record_manual(
            service=service,
            exc_type=exc_type,
            exc_message=exc_message or "",
            traceback_text=traceback_text or "",
            context=context,
        )

    def record_manual(
        self,
        service: str,
        exc_type: str,
        exc_message: str,
        traceback_text: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an exception manually (no active exception required)."""
        fingerprint = hashlib.md5(
            f"{service}:{exc_type}:{exc_message}".encode("utf-8")
        ).hexdigest()
        if fingerprint in self._seen:
            return
        self._seen.add(fingerprint)
        self._entries.append({
            "timestamp": time.time(),
            "service": service,
            "exc_type": exc_type,
            "exc_message": exc_message,
            "traceback": traceback_text,
            "context": context or {},
            "fingerprint": fingerprint,
        })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, index: int) -> Optional[Dict[str, Any]]:
        """Get entry by index (0 = oldest)."""
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent entries."""
        return list(self._entries)[-limit:]

    def filter(
        self,
        service: Optional[str] = None,
        exc_type: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter entries by criteria."""
        results: List[Dict[str, Any]] = []
        for entry in self._entries:
            if service and entry["service"] != service:
                continue
            if exc_type and entry["exc_type"] != exc_type:
                continue
            if since and entry["timestamp"] < since:
                continue
            results.append(entry)
        return results

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._seen.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        by_service: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for entry in self._entries:
            svc = entry["service"]
            et = entry["exc_type"]
            by_service[svc] = by_service.get(svc, 0) + 1
            by_type[et] = by_type.get(et, 0) + 1
        return {
            "total": len(self._entries),
            "by_service": by_service,
            "by_type": by_type,
        }

    def __repr__(self) -> str:
        return f"<ExceptionTracker entries={len(self._entries)}>"
