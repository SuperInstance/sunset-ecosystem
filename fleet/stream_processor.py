"""Windowed stream processing for real-time fleet metrics.

Processes time-windowed streams of events with aggregation functions.
Used for computing rolling averages, rate limits, and anomaly detection
over fleet telemetry streams.

Usage:
    sp = StreamProcessor(window_sec=5.0)
    sp.push({"metric": "cpu", "value": 42.0})
    sp.push({"metric": "cpu", "value": 45.0})
    result = sp.aggregate("cpu", avg)
    # result ~ 43.5
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """A single event in the stream."""

    timestamp: float
    data: Dict[str, Any]


class StreamProcessor:
    """
    Time-windowed stream processor with per-key aggregation.

    :param window_sec: Time window for retention.
    :param max_events: Maximum events per key before eviction.
    """

    def __init__(self, window_sec: float = 60.0, max_events: int = 1000):
        self._window = window_sec
        self._max_events = max_events
        self._streams: Dict[str, deque] = {}  # key -> deque of StreamEvent
        self._stats: Dict[str, int] = {"pushed": 0, "evicted": 0}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def push(self, data: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        """Push an event into the stream."""
        now = timestamp or time.time()
        self._stats["pushed"] += 1

        # Route to relevant streams based on data keys
        for key, value in data.items():
            if key not in self._streams:
                self._streams[key] = deque()
            stream = self._streams[key]
            stream.append(StreamEvent(timestamp=now, data={key: value}))
            if len(stream) > self._max_events:
                stream.popleft()
                self._stats["evicted"] += 1

        # Evict old events
        self._evict_old(now)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        key: str,
        fn: Callable[[List[Any]], Any],
    ) -> Optional[Any]:
        """
        Aggregate values for a key using a custom function.

        :param key: Data key to aggregate.
        :param fn: Function taking List[Any] -> Any.
        """
        stream = self._streams.get(key)
        if not stream:
            return None
        values = [e.data.get(key) for e in stream]
        return fn(values)

    def count(self, key: str) -> int:
        """Count events for a key in the current window."""
        stream = self._streams.get(key)
        return len(stream) if stream else 0

    def sum(self, key: str) -> float:
        """Sum values for a numeric key."""
        stream = self._streams.get(key)
        if not stream:
            return 0.0
        return sum(e.data.get(key, 0) for e in stream if isinstance(e.data.get(key), (int, float)))

    def avg(self, key: str) -> Optional[float]:
        """Average value for a numeric key."""
        stream = self._streams.get(key)
        if not stream:
            return None
        values = [e.data.get(key) for e in stream if isinstance(e.data.get(key), (int, float))]
        if not values:
            return None
        return sum(values) / len(values)

    def min(self, key: str) -> Optional[Any]:
        """Minimum value for a key."""
        stream = self._streams.get(key)
        if not stream:
            return None
        values = [e.data.get(key) for e in stream]
        return min(values) if values else None

    def max(self, key: str) -> Optional[Any]:
        """Maximum value for a key."""
        stream = self._streams.get(key)
        if not stream:
            return None
        values = [e.data.get(key) for e in stream]
        return max(values) if values else None

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._window
        for key, stream in self._streams.items():
            while stream and stream[0].timestamp < cutoff:
                stream.popleft()

    def clear(self) -> None:
        """Clear all streams."""
        self._streams.clear()

    def keys(self) -> List[str]:
        return list(self._streams.keys())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        total = sum(len(s) for s in self._streams.values())
        return f"<StreamProcessor keys={len(self._streams)} events={total}>"
