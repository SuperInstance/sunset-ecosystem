"""tsdb.py — Lightweight time-series database for fleet metrics.

Ring-buffer based TSDB optimized for append-only write, time-range queries,
and downsampling. Stores numeric metrics with nanosecond timestamps.

Features:
1. Per-metric circular buffer (configurable retention)
2. Time-range queries with aggregation
3. Downsampling: mean, min, max, count per time bucket
4. Series metadata (labels, type, unit)
5. Prometheus-style exposition format export

No external dependencies — pure Python with numpy for aggregation.
"""
from __future__ import annotations

__all__ = [
    "TimeSeriesDB",
    "MetricSeries",
    "DataPoint",
    "Aggregation",
]

import enum
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Aggregation(enum.Enum):
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    LAST = "last"
    STD = "std"


@dataclass(frozen=True)
class DataPoint:
    """Single time-series data point."""
    timestamp_ns: int
    value: float

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9


@dataclass
class MetricSeries:
    """A named time-series with circular buffer storage."""
    name: str
    labels: dict[str, str]
    unit: str = ""
    retention: int = 10_000  # max points to keep

    def __post_init__(self) -> None:
        self._timestamps: np.ndarray = np.zeros(self.retention, dtype=np.int64)
        self._values: np.ndarray = np.zeros(self.retention, dtype=np.float64)
        self._count: int = 0
        self._write_idx: int = 0
        self._last_value: float = 0.0

    # ── write ──────────────────────────────────────────

    def append(self, value: float, timestamp_ns: int | None = None) -> None:
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        self._timestamps[self._write_idx] = timestamp_ns
        self._values[self._write_idx] = value
        self._write_idx = (self._write_idx + 1) % self.retention
        self._count = min(self._count + 1, self.retention)
        self._last_value = value

    # ── read ───────────────────────────────────────────

    def _get_sorted(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, values) in chronological order."""
        if self._count < self.retention:
            return self._timestamps[:self._count], self._values[:self._count]
        # Wraparound: oldest is at _write_idx
        ts = np.concatenate([
            self._timestamps[self._write_idx:],
            self._timestamps[:self._write_idx],
        ])
        vs = np.concatenate([
            self._values[self._write_idx:],
            self._values[:self._write_idx],
        ])
        return ts, vs

    def range_query(
        self,
        start_ns: int | None = None,
        end_ns: int | None = None,
        aggregation: Aggregation = Aggregation.LAST,
        bucket_ns: int | None = None,
    ) -> list[tuple[int, float]]:
        """Query data points in time range with optional downsampling.

        Returns list of (timestamp_ns, aggregated_value).
        """
        ts, vs = self._get_sorted()
        if len(ts) == 0:
            return []

        # Filter by range
        mask = np.ones(len(ts), dtype=bool)
        if start_ns is not None:
            mask &= ts >= start_ns
        if end_ns is not None:
            mask &= ts <= end_ns

        ts = ts[mask]
        vs = vs[mask]

        if len(ts) == 0:
            return []

        if bucket_ns is None or bucket_ns <= 0:
            # No downsampling — return raw
            return [(int(ts[i]), float(vs[i])) for i in range(len(ts))]

        # Downsample into buckets
        buckets: dict[int, list[float]] = {}
        for i in range(len(ts)):
            bucket = int(ts[i] // bucket_ns) * bucket_ns
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(vs[i])

        result: list[tuple[int, float]] = []
        for bucket in sorted(buckets.keys()):
            vals = np.array(buckets[bucket])
            if aggregation == Aggregation.MEAN:
                val = float(np.mean(vals))
            elif aggregation == Aggregation.MIN:
                val = float(np.min(vals))
            elif aggregation == Aggregation.MAX:
                val = float(np.max(vals))
            elif aggregation == Aggregation.SUM:
                val = float(np.sum(vals))
            elif aggregation == Aggregation.COUNT:
                val = float(len(vals))
            elif aggregation == Aggregation.STD:
                val = float(np.std(vals))
            else:  # LAST
                val = float(vals[-1])
            result.append((bucket, val))

        return result

    def last(self) -> DataPoint | None:
        if self._count == 0:
            return None
        ts, vs = self._get_sorted()
        return DataPoint(timestamp_ns=int(ts[-1]), value=float(vs[-1]))

    def stats(self) -> dict[str, Any]:
        if self._count == 0:
            return {"count": 0}
        ts, vs = self._get_sorted()
        return {
            "count": self._count,
            "min": float(np.min(vs)),
            "max": float(np.max(vs)),
            "mean": float(np.mean(vs)),
            "std": float(np.std(vs)),
            "last": float(vs[-1]),
            "time_span_sec": float((ts[-1] - ts[0]) / 1e9) if len(ts) > 1 else 0.0,
        }


# ── TimeSeriesDB ──────────────────────────────────────────────

class TimeSeriesDB:
    """Multi-metric time-series database."""

    def __init__(self) -> None:
        self._series: dict[str, MetricSeries] = {}

    def create_series(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        unit: str = "",
        retention: int = 10_000,
    ) -> MetricSeries:
        """Create or get a metric series."""
        key = self._series_key(name, labels or {})
        if key not in self._series:
            self._series[key] = MetricSeries(
                name=name,
                labels=labels or {},
                unit=unit,
                retention=retention,
            )
        return self._series[key]

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        """Record a single data point."""
        series = self.create_series(name, labels)
        series.append(value, timestamp_ns)

    def query(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Query a series by name and labels."""
        key = self._series_key(name, labels or {})
        if key not in self._series:
            return []
        return self._series[key].range_query(**kwargs)

    def series_names(self) -> set[str]:
        return {s.name for s in self._series.values()}

    def all_series(self) -> dict[str, MetricSeries]:
        return dict(self._series)

    def _series_key(self, name: str, labels: dict[str, str]) -> str:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # ── prometheus exposition ─────────────────────────

    def prometheus_exposition(self) -> str:
        """Export all series in Prometheus text format."""
        lines: list[str] = []
        for key, series in sorted(self._series.items()):
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(series.labels.items()))
            if label_str:
                lines.append(f"# HELP {series.name} {series.name}")
                lines.append(f"# TYPE {series.name} gauge")
                last = series.last()
                if last:
                    lines.append(f'{series.name}{{{label_str}}} {last.value}')
        return "\n".join(lines)

    # ── aggregation across series ─────────────────────

    def aggregate(
        self,
        name: str,
        aggregation: Aggregation = Aggregation.MEAN,
    ) -> float | None:
        """Aggregate all series matching name across label dimensions."""
        matching = [s for k, s in self._series.items() if s.name == name]
        if not matching:
            return None

        if aggregation == Aggregation.LAST:
            vals = [s.last().value for s in matching if s.last()]
        elif aggregation == Aggregation.MEAN:
            vals = [s.stats()["mean"] for s in matching]
        elif aggregation == Aggregation.MAX:
            vals = [s.stats()["max"] for s in matching]
        elif aggregation == Aggregation.MIN:
            vals = [s.stats()["min"] for s in matching]
        elif aggregation == Aggregation.COUNT:
            vals = [s.stats()["count"] for s in matching]
        else:
            vals = []

        if not vals:
            return None
        return float(np.mean(vals))

    def report(self) -> dict[str, Any]:
        return {
            "series_count": len(self._series),
            "unique_names": len(self.series_names()),
            "total_points": sum(s._count for s in self._series.values()),
        }
