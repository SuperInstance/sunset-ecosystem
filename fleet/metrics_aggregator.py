"""metrics_aggregator.py — Fleet-wide metrics aggregation.

Provides:
1. Per-node metric collection with timestamps
2. Cross-node aggregation (sum, avg, min, max, count)
3. Time-windowed queries
4. Metric export (dict/JSON)
5. Alert thresholds

Usage:
    agg = MetricsAggregator()
    agg.record("node-a", "cpu_percent", 45.2)
    agg.record("node-b", "cpu_percent", 32.1)
    summary = agg.aggregate("cpu_percent", window_sec=60)
    # summary.avg, summary.max, summary.min, summary.count
"""
from __future__ import annotations

__all__ = [
    "MetricsAggregator",
    "MetricValue",
    "MetricSummary",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A single metric reading."""
    node_id: str
    name: str
    value: float
    timestamp: float


@dataclass
class MetricSummary:
    """Aggregated metric statistics."""
    name: str
    count: int
    sum: float
    avg: float
    min: float
    max: float
    nodes: int


class MetricsAggregator:
    """Aggregate metrics across fleet nodes."""

    def __init__(self, max_history: int = 10_000) -> None:
        self._max_history = max_history
        self._metrics: list[MetricValue] = []
        self._thresholds: dict[str, tuple[float, float]] = {}  # name -> (min, max)

    def record(self, node_id: str, name: str, value: float) -> None:
        """Record a metric value."""
        self._metrics.append(MetricValue(
            node_id=node_id,
            name=name,
            value=value,
            timestamp=time.time(),
        ))
        if len(self._metrics) > self._max_history:
            self._metrics = self._metrics[-self._max_history:]

        # Check thresholds
        if name in self._thresholds:
            min_val, max_val = self._thresholds[name]
            if value < min_val or value > max_val:
                logger.warning(f"Metric '{name}' out of bounds: {value} (range [{min_val}, {max_val}])")

    def set_threshold(self, name: str, min_val: float, max_val: float) -> None:
        """Set alert thresholds for a metric."""
        self._thresholds[name] = (min_val, max_val)

    def aggregate(self, name: str, window_sec: float | None = None) -> MetricSummary | None:
        """Aggregate a metric over an optional time window."""
        now = time.time()
        values = [
            m.value for m in self._metrics
            if m.name == name and (window_sec is None or now - m.timestamp <= window_sec)
        ]
        if not values:
            return None

        nodes = len({m.node_id for m in self._metrics if m.name == name})
        return MetricSummary(
            name=name,
            count=len(values),
            sum=sum(values),
            avg=sum(values) / len(values),
            min=min(values),
            max=max(values),
            nodes=nodes,
        )

    def latest(self, name: str, node_id: str | None = None) -> float | None:
        """Get the latest value of a metric."""
        candidates = [
            m for m in reversed(self._metrics)
            if m.name == name and (node_id is None or m.node_id == node_id)
        ]
        return candidates[0].value if candidates else None

    def per_node(self, name: str, window_sec: float | None = None) -> dict[str, list[float]]:
        """Get metric values grouped by node."""
        now = time.time()
        result: dict[str, list[float]] = {}
        for m in self._metrics:
            if m.name != name:
                continue
            if window_sec is not None and now - m.timestamp > window_sec:
                continue
            result.setdefault(m.node_id, []).append(m.value)
        return result

    def node_names(self) -> list[str]:
        """Get all known node IDs."""
        return sorted({m.node_id for m in self._metrics})

    def metric_names(self) -> list[str]:
        """Get all known metric names."""
        return sorted({m.name for m in self._metrics})

    def export(self) -> list[dict[str, Any]]:
        """Export all metrics as dicts."""
        return [
            {
                "node_id": m.node_id,
                "name": m.name,
                "value": m.value,
                "timestamp": m.timestamp,
            }
            for m in self._metrics
        ]

    def report(self) -> dict[str, Any]:
        """Summary report of all metrics."""
        names = self.metric_names()
        return {
            "total_readings": len(self._metrics),
            "nodes": len(self.node_names()),
            "metrics": {
                name: self.aggregate(name) for name in names
            },
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()

    def __repr__(self) -> str:
        return f"MetricsAggregator(readings={len(self._metrics)}, nodes={len(self.node_names())})"
