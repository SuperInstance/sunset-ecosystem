"""Metrics aggregation with rollup windows.

Aggregates time-series metrics into configurable windows (raw, 1m, 5m,
1h). Supports sum, avg, min, max, count aggregations. Used for fleet
monitoring dashboards, SLA tracking, and capacity planning.

Usage:
    agg = MetricsAggregator(window_sec=60)
    agg.record("cpu", 45.0)
    agg.record("cpu", 55.0)
    rollup = agg.rollup("cpu")
    assert rollup["avg"] == 50.0
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class MetricsAggregator:
    """
    Metrics aggregator with time windows.

    :param window_sec: Aggregation window in seconds.
    :param clock: Optional clock function for testing.
    """

    def __init__(self, window_sec: float = 60.0, clock: Optional[callable] = None):
        self._window = window_sec
        self._clock = clock or time.time
        self._data: Dict[str, List[tuple]] = {}  # metric -> [(timestamp, value)]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, metric: str, value: float) -> None:
        """Record a metric value."""
        if metric not in self._data:
            self._data[metric] = []
        self._data[metric].append((self._clock(), value))

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def rollup(self, metric: str, window_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Aggregate metric values in a window.

        :param metric: Metric name.
        :param window_sec: Override window (defaults to constructor value).
        :returns: Dict with sum, avg, min, max, count, or None if no data.
        """
        data = self._data.get(metric, [])
        if not data:
            return None
        window = window_sec or self._window
        cutoff = self._clock() - window
        values = [v for ts, v in data if ts >= cutoff]
        if not values:
            return None
        total = sum(values)
        count = len(values)
        return {
            "sum": total,
            "avg": total / count,
            "min": min(values),
            "max": max(values),
            "count": count,
        }

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def prune(self, metric: str, max_age_sec: Optional[float] = None) -> int:
        """Prune old data points."""
        data = self._data.get(metric, [])
        if not data:
            return 0
        cutoff = self._clock() - (max_age_sec or self._window * 10)
        before = len(data)
        self._data[metric] = [(ts, v) for ts, v in data if ts >= cutoff]
        return before - len(self._data[metric])

    def prune_all(self, max_age_sec: Optional[float] = None) -> int:
        """Prune all metrics."""
        total = 0
        for metric in list(self._data.keys()):
            total += self.prune(metric, max_age_sec)
        return total

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def metrics(self) -> List[str]:
        return list(self._data.keys())

    def count(self, metric: str) -> int:
        return len(self._data.get(metric, []))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total_points = sum(len(v) for v in self._data.values())
        return {
            "metrics": len(self._data),
            "total_points": total_points,
            "window_sec": self._window,
        }

    def __repr__(self) -> str:
        return f"<MetricsAggregator metrics={len(self._data)}>"
