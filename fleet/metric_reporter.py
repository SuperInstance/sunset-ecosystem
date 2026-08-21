"""Metric reporting with aggregation and flush.

Collects metrics (counters, gauges, timers) and flushes them in batches.
Supports aggregation (sum, avg, min, max) and configurable flush intervals.
Used for fleet metrics collection, telemetry aggregation, and performance monitoring.

Usage:
    reporter = MetricReporter(flush_interval_sec=60)
    reporter.counter("requests", 1)
    reporter.gauge("cpu", 0.75)
    reporter.timer("latency", 0.023)
    batch = reporter.flush()  # Returns aggregated metrics
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class MetricReporter:
    """
    Metric reporter with aggregation and flush.

    :param flush_interval_sec: Flush interval (None for manual only).
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        flush_interval_sec: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._flush_interval = flush_interval_sec
        self._clock = clock or time.time
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = {}
        self._last_flush = self._clock()
        self._total_flushes = 0

    # ------------------------------------------------------------------
    # Metric recording
    # ------------------------------------------------------------------

    def counter(self, name: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        self._counters[name] = self._counters.get(name, 0.0) + value

    def gauge(self, name: str, value: float) -> None:
        """Record a gauge metric (overwrites previous)."""
        self._gauges[name] = value

    def timer(self, name: str, duration_sec: float) -> None:
        """Record a timer observation."""
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(duration_sec)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def should_flush(self) -> bool:
        """Check if metrics should be flushed."""
        if not self._counters and not self._gauges and not self._timers:
            return False
        if self._flush_interval:
            elapsed = self._clock() - self._last_flush
            return elapsed >= self._flush_interval
        return False

    def flush(self) -> Dict[str, Any]:
        """
        Flush all metrics and return aggregated batch.

        :returns: Dict with counters, gauges, and timer aggregates.
        """
        result = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "timers": {},
        }

        for name, values in self._timers.items():
            if values:
                result["timers"][name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        # Clear metrics
        self._counters.clear()
        self._gauges.clear()
        self._timers.clear()
        self._last_flush = self._clock()
        self._total_flushes += 1

        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def metric_names(self) -> List[str]:
        """List all metric names currently held."""
        names = (
            set(self._counters.keys())
            | set(self._gauges.keys())
            | set(self._timers.keys())
        )
        return sorted(names)

    def get_counter(self, name: str) -> Optional[float]:
        return self._counters.get(name)

    def get_gauge(self, name: str) -> Optional[float]:
        return self._gauges.get(name)

    def get_timer_stats(self, name: str) -> Optional[Dict[str, Any]]:
        values = self._timers.get(name)
        if not values:
            return None
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "timers": len(self._timers),
            "total_flushes": self._total_flushes,
            "last_flush_ago": self._clock() - self._last_flush,
        }

    def __repr__(self) -> str:
        return f"<MetricReporter counters={len(self._counters)} gauges={len(self._gauges)} timers={len(self._timers)}>"
