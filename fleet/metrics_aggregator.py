from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetricPoint:
    """A single metric data point."""

    metric_name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


class MetricsAggregator:
    """
    Aggregates metrics from all fleet nodes.

    Collects, buffers, and computes statistics on fleet-wide metrics.
    """

    def __init__(self, fleet_node_id: str = "default", buffer_size: int = 10000):
        self.fleet_node_id = fleet_node_id
        self.buffer_size = buffer_size
        self._metrics: Dict[str, List[MetricPoint]] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}

    def record(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        point = MetricPoint(
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(point)
        # Trim buffer
        if len(self._metrics[metric_name]) > self.buffer_size:
            self._metrics[metric_name] = self._metrics[metric_name][-self.buffer_size :]

    def increment(self, counter_name: str, value: float = 1.0):
        """Increment a counter."""
        self._counters[counter_name] = self._counters.get(counter_name, 0.0) + value

    def gauge(self, gauge_name: str, value: float):
        """Set a gauge value."""
        self._gauges[gauge_name] = value

    def get_series(
        self, metric_name: str, since: Optional[float] = None
    ) -> List[MetricPoint]:
        """Get metric series, optionally filtered by time."""
        points = self._metrics.get(metric_name, [])
        if since:
            points = [p for p in points if p.timestamp >= since]
        return points

    def get_stats(self, metric_name: str) -> Dict[str, Any]:
        """Compute statistics for a metric."""
        points = self._metrics.get(metric_name, [])
        if not points:
            return {"count": 0}
        values = [p.value for p in points]
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "last": values[-1],
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all metrics."""
        return {name: self.get_stats(name) for name in self._metrics}

    def get_counters(self) -> Dict[str, float]:
        """Get all counter values."""
        return dict(self._counters)

    def get_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        return dict(self._gauges)

    def export_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "stats": self.get_all_stats(),
                "counters": self._counters,
                "gauges": self._gauges,
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_all_stats(),
            "counters": len(self._counters),
            "gauges": len(self._gauges),
        }
