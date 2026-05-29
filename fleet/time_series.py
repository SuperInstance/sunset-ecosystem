"""Circular time series buffer with aggregation.

Stores time-ordered samples in a fixed window. Supports downsampling,
aggregation queries, and gap detection. Used for fleet metrics and
telemetry time series.

Usage:
    ts = TimeSeries(capacity=100)
    ts.push(42.0)
    ts.push(45.0)
    assert ts.avg() == 43.5
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Sample:
    """A single time series sample."""

    timestamp: float
    value: float


class TimeSeries:
    """
    Fixed-capacity time series with O(1) append and O(N) queries.

    :param capacity: Maximum samples to retain.
    :param window_sec: Time window for eviction (None = capacity only).
    """

    def __init__(self, capacity: int = 1000, window_sec: Optional[float] = None):
        self._capacity = capacity
        self._window = window_sec
        self._samples: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def push(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a sample."""
        ts = timestamp or time.time()
        self._samples.append(Sample(timestamp=ts, value=value))
        if self._window:
            self._evict_old(ts)

    def extend(self, values: List[Tuple[float, float]]) -> None:
        """Add multiple (timestamp, value) pairs."""
        for ts, val in values:
            self.push(val, ts)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def avg(self) -> Optional[float]:
        """Average of all samples."""
        if not self._samples:
            return None
        return sum(s.value for s in self._samples) / len(self._samples)

    def sum(self) -> float:
        """Sum of all samples."""
        return sum(s.value for s in self._samples)

    def min(self) -> Optional[float]:
        """Minimum value."""
        if not self._samples:
            return None
        return min(s.value for s in self._samples)

    def max(self) -> Optional[float]:
        """Maximum value."""
        if not self._samples:
            return None
        return max(s.value for s in self._samples)

    def count(self) -> int:
        """Number of samples."""
        return len(self._samples)

    def latest(self) -> Optional[Sample]:
        """Most recent sample."""
        return self._samples[-1] if self._samples else None

    def earliest(self) -> Optional[Sample]:
        """Oldest sample."""
        return self._samples[0] if self._samples else None

    def range(self) -> Optional[Tuple[float, float]]:
        """(min, max) value range."""
        if not self._samples:
            return None
        return (self.min(), self.max())

    def to_list(self) -> List[Tuple[float, float]]:
        """Return all samples as (timestamp, value) list."""
        return [(s.timestamp, s.value) for s in self._samples]

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    def downsample(self, bucket_sec: float) -> List[Tuple[float, float]]:
        """
        Downsample into buckets of *bucket_sec* seconds.

        Returns list of (bucket_start, avg_value).
        """
        if not self._samples:
            return []
        buckets: dict = {}
        for s in self._samples:
            bucket = int(s.timestamp / bucket_sec) * bucket_sec
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(s.value)
        return sorted(
            (b, sum(v) / len(v)) for b, v in buckets.items()
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._window
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def clear(self) -> None:
        """Clear all samples."""
        self._samples.clear()

    def __repr__(self) -> str:
        return f"<TimeSeries samples={len(self._samples)}>"
