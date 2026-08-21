"""
Fleet Telemetry Exporter

Exports fleet metrics to Prometheus, OpenTelemetry, or any metrics backend.
Standardizes breeding, spatial, and health metrics for observability.

Usage:
    from fleet.telemetry_exporter import TelemetryExporter
    exporter = TelemetryExporter()
    exporter.record_breeding_metrics(generation=42, fitness=0.95)
    exporter.export_prometheus()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetricSample:
    """A single metric sample."""

    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    type: str = "gauge"  # gauge, counter, histogram

    def to_prometheus(self) -> str:
        """Convert to Prometheus text format."""
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if label_str:
            return (
                f"{self.name}{{{label_str}}} {self.value} {int(self.timestamp * 1000)}"
            )
        return f"{self.name} {self.value} {int(self.timestamp * 1000)}"

    def to_otel(self) -> Dict[str, Any]:
        """Convert to OpenTelemetry format."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "attributes": self.labels,
            "type": self.type,
        }


class TelemetryExporter:
    """
    Collects and exports fleet metrics.

    Supports Prometheus exposition format and OpenTelemetry.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.samples: List[MetricSample] = []
        self.counters: Dict[str, float] = {}

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: str = "gauge",
    ):
        """Record a metric sample."""
        sample = MetricSample(
            name=name,
            value=value,
            timestamp=time.time(),
            labels={"node": self.fleet_node_id, **(labels or {})},
            type=metric_type,
        )
        self.samples.append(sample)

        if metric_type == "counter":
            self.counters[name] = self.counters.get(name, 0) + value

    def record_breeding_metrics(
        self,
        generation: int,
        population_size: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float,
    ):
        """Record standard breeding metrics."""
        self.record("breeding_generation", generation, {"type": "generation"})
        self.record("breeding_population", population_size, {"type": "population"})
        self.record("breeding_best_fitness", best_fitness, {"type": "fitness"})
        self.record("breeding_avg_fitness", avg_fitness, {"type": "fitness"})
        self.record("breeding_diversity", diversity, {"type": "diversity"})

    def record_spatial_metrics(
        self,
        n_agents: int,
        n_rooms: int,
        avg_agent_density: float,
        collision_rate: float,
    ):
        """Record spatial world metrics."""
        self.record("spatial_agents", n_agents, {"type": "agents"})
        self.record("spatial_rooms", n_rooms, {"type": "rooms"})
        self.record("spatial_density", avg_agent_density, {"type": "density"})
        self.record("spatial_collision_rate", collision_rate, {"type": "safety"})

    def record_health_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        n_active_agents: int,
        queue_depth: int,
    ):
        """Record fleet health metrics."""
        self.record("health_cpu", cpu_percent, {"type": "cpu"})
        self.record("health_memory", memory_percent, {"type": "memory"})
        self.record("health_active_agents", n_active_agents, {"type": "agents"})
        self.record("health_queue_depth", queue_depth, {"type": "queue"})

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = ["# Fleet Telemetry"]
        for sample in self.samples:
            lines.append(sample.to_prometheus())
        return "\n".join(lines)

    def export_otel(self) -> List[Dict[str, Any]]:
        """Export all metrics in OpenTelemetry format."""
        return [s.to_otel() for s in self.samples]

    def export_json(self) -> str:
        """Export all metrics as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "timestamp": time.time(),
                "samples": [s.to_otel() for s in self.samples],
            },
            indent=2,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get statistical summary of all metrics."""
        if not self.samples:
            return {}

        by_name: Dict[str, List[float]] = {}
        for s in self.samples:
            by_name.setdefault(s.name, []).append(s.value)

        return {
            name: {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
            for name, values in by_name.items()
        }

    def clear(self):
        """Clear all samples."""
        self.samples = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
            "samples": len(self.samples),
            "counters": self.counters,
            "summary": self.get_summary(),
        }
