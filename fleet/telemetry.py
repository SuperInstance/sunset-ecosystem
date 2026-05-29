"""Metrics collection with Prometheus/StatsD-style exporters.

Lightweight telemetry for fleet node health, breeding throughput,
and resource utilization. Supports counters, gauges, histograms.

Usage:
    telem = TelemetryRegistry()
    telem.counter("breeds_total").inc()
    telem.gauge("cpu_percent").set(42.0)
    telem.histogram("breed_latency_ms").observe(150)
    print(telem.prometheus_format())
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Counter:
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class Gauge:
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class Histogram:
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    buckets: List[float] = field(default_factory=lambda: [10, 50, 100, 250, 500, 1000, 2500, 5000])
    _counts: Dict[float, int] = field(default_factory=lambda: defaultdict(int))
    _sum: float = 0.0
    _total: int = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._total += 1
        for b in self.buckets:
            if value <= b:
                self._counts[b] += 1

    def percentile(self, p: float) -> float:
        """Approximate percentile from bucket counts (naïve)."""
        if self._total == 0:
            return 0.0
        target = int(self._total * p)
        cumulative = 0
        for b in sorted(self.buckets):
            cumulative += self._counts.get(b, 0)
            if cumulative >= target:
                return b
        return float('inf')


class TelemetryRegistry:
    """
    Central telemetry registry for fleet-wide metrics.

    Supports Prometheus text format export and StatsD line format.
    """

    def __init__(self, prefix: str = "fleet"):
        self._prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    # ------------------------------------------------------------------
    # Metric creation / retrieval
    # ------------------------------------------------------------------

    def counter(self, name: str, **labels: str) -> Counter:
        key = self._key(name, labels)
        if key not in self._counters:
            self._counters[key] = Counter(name=name, labels=dict(labels))
        return self._counters[key]

    def gauge(self, name: str, **labels: str) -> Gauge:
        key = self._key(name, labels)
        if key not in self._gauges:
            self._gauges[key] = Gauge(name=name, labels=dict(labels))
        return self._gauges[key]

    def histogram(self, name: str, buckets: Optional[List[float]] = None, **labels: str) -> Histogram:
        key = self._key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = Histogram(
                name=name, labels=dict(labels), buckets=buckets or [10, 50, 100, 250, 500, 1000, 2500, 5000]
            )
        return self._histograms[key]

    # ------------------------------------------------------------------
    # Prometheus text format
    # ------------------------------------------------------------------

    def prometheus_format(self) -> str:
        """Export all metrics in Prometheus text exposition format."""
        lines: List[str] = []
        for key, c in self._counters.items():
            lines.append(f"# TYPE {c.name} counter")
            lines.append(f"{self._prefix}_{c.name}{self._label_str(c.labels)} {c.value}")
        for key, g in self._gauges.items():
            lines.append(f"# TYPE {g.name} gauge")
            lines.append(f"{self._prefix}_{g.name}{self._label_str(g.labels)} {g.value}")
        for key, h in self._histograms.items():
            lines.append(f"# TYPE {h.name} histogram")
            for b in sorted(h.buckets):
                le = "+Inf" if b == float('inf') else str(b)
                lines.append(f"{self._prefix}_{h.name}_bucket{{le=\"{le}\"}}{self._label_str(h.labels, leading_comma=False)} {h._counts.get(b, 0)}")
            lines.append(f"{self._prefix}_{h.name}_sum{self._label_str(h.labels)} {h._sum}")
            lines.append(f"{self._prefix}_{h.name}_count{self._label_str(h.labels)} {h._total}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # StatsD line format
    # ------------------------------------------------------------------

    def statsd_format(self) -> List[str]:
        """Export all metrics as StatsD lines."""
        lines: List[str] = []
        for c in self._counters.values():
            lines.append(f"{self._prefix}.{c.name}:{c.value}|c")
        for g in self._gauges.values():
            lines.append(f"{self._prefix}.{g.name}:{g.value}|g")
        for h in self._histograms.values():
            # StatsD timers/histograms use single values; emit aggregate
            avg = h._sum / h._total if h._total > 0 else 0
            lines.append(f"{self._prefix}.{h.name}:{avg}|ms")
        return lines

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def all_metrics(self) -> Dict[str, Any]:
        return {
            "counters": {k: {"value": v.value} for k, v in self._counters.items()},
            "gauges": {k: {"value": v.value} for k, v in self._gauges.items()},
            "histograms": {k: {"sum": v._sum, "count": v._total} for k, v in self._histograms.items()},
        }

    def _key(self, name: str, labels: Dict[str, str]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f'{name}{{{label_str}}}'

    def _label_str(self, labels: Dict[str, str], leading_comma: bool = True) -> str:
        if not labels:
            return ""
        s = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return "{" + s + "}"

    def __repr__(self) -> str:
        total = len(self._counters) + len(self._gauges) + len(self._histograms)
        return f"<TelemetryRegistry prefix={self._prefix} metrics={total}>"
