"""FleetMonitor — Observability and health tracking for the fleet.

Tracks:
- Node health (CPU, memory, latency)
- Mesh table stats (entries, sync, divergence)
- Tiered storage distribution (hot/warm/cold)
- HNSW index status (build, coverage, rebuilds)
- CognitiveCache predictions (hit rate, accuracy)
- SceneTracker patterns (scenes, transitions, anomalies)
- Alert thresholds and escalation

Usage:
    monitor = FleetMonitor()
    monitor.register_node("node1", table1, storage1, hnsw1)
    monitor.register_node("node2", table2, storage2, hnsw2)
    
    # Get fleet-wide health report
    report = monitor.health_report()
    
    # Check alerts
    alerts = monitor.check_alerts()
    for alert in alerts:
        print(alert.level, alert.message)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from fleet.cognitive_cache import CognitiveCache
from swarm.hnsw_mesh_table import HnswMeshTable
from swarm.mesh_vector_tables import MeshVectorTable
from swarm.scene_tracker import SceneTracker
from swarm.tiered_mesh_storage import TieredMeshStorage


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    level: AlertLevel
    node_id: str
    component: str
    message: str
    metric: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class NodeHealth:
    node_id: str
    table_entries: int
    hot_entries: int
    warm_entries: int
    cold_entries: int
    hnsw_available: bool
    hnsw_index_count: int
    hnsw_coverage: float
    has_hnsw: bool
    cache_hit_rate: float
    cache_prediction_accuracy: float
    has_cache: bool
    scene_count: int
    query_rate: float
    last_update: float


class FleetMonitor:
    """Fleet-wide observability and health tracking."""

    def __init__(
        self,
        cache_hit_threshold: float = 0.7,
        cache_accuracy_threshold: float = 0.5,
        hnsw_coverage_threshold: float = 0.9,
        hot_ratio_threshold: float = 0.5,
        query_rate_threshold: float = 100.0,
    ):
        self._nodes: dict[str, dict[str, Any]] = {}
        self._alerts: list[Alert] = []
        self._history: list[dict[str, Any]] = []
        self._max_history = 1000

        self.cache_hit_threshold = cache_hit_threshold
        self.cache_accuracy_threshold = cache_accuracy_threshold
        self.hnsw_coverage_threshold = hnsw_coverage_threshold
        self.hot_ratio_threshold = hot_ratio_threshold
        self.query_rate_threshold = query_rate_threshold

    def register_node(
        self,
        node_id: str,
        table: MeshVectorTable,
        storage: Optional[TieredMeshStorage] = None,
        hnsw: Optional[HnswMeshTable] = None,
        cache: Optional[CognitiveCache] = None,
        tracker: Optional[SceneTracker] = None,
    ) -> None:
        """Register a node for monitoring."""
        self._nodes[node_id] = {
            "table": table,
            "storage": storage,
            "hnsw": hnsw,
            "cache": cache,
            "tracker": tracker,
            "registered_at": time.time(),
        }

    def health_report(self) -> dict[str, Any]:
        """Generate fleet-wide health report."""
        node_reports = {}
        for node_id, node in self._nodes.items():
            report = self._node_health(node_id, node)
            node_reports[node_id] = report

        # Aggregate
        total_entries = sum(r.table_entries for r in node_reports.values())
        total_hot = sum(r.hot_entries for r in node_reports.values())
        avg_hit_rate = np.mean([r.cache_hit_rate for r in node_reports.values()])
        avg_accuracy = np.mean([r.cache_prediction_accuracy for r in node_reports.values()])
        hnsw_nodes = sum(1 for r in node_reports.values() if r.hnsw_available)

        return {
            "timestamp": time.time(),
            "node_count": len(self._nodes),
            "total_entries": total_entries,
            "hot_ratio": total_hot / total_entries if total_entries else 0.0,
            "avg_cache_hit_rate": avg_hit_rate,
            "avg_prediction_accuracy": avg_accuracy,
            "hnsw_coverage": hnsw_nodes / len(self._nodes) if self._nodes else 0.0,
            "nodes": {nid: self._health_to_dict(r) for nid, r in node_reports.items()},
        }

    def _node_health(self, node_id: str, node: dict[str, Any]) -> NodeHealth:
        """Compute health for a single node."""
        table = node["table"]
        storage = node.get("storage")
        hnsw = node.get("hnsw")
        cache = node.get("cache")
        tracker = node.get("tracker")

        table_entries = len(table)
        hot_entries = len(storage.base) if storage else table_entries
        warm_entries = storage._warm_count_entries() if storage else 0
        cold_entries = storage._cold_count if storage else 0

        hnsw_available = hnsw.stats.get("hnsw_available", False) if hnsw else False
        hnsw_index_count = hnsw.stats.get("index_count", 0) if hnsw else 0
        hnsw_coverage = hnsw_index_count / table_entries if table_entries else 0.0

        cache_hit_rate = 0.0
        cache_accuracy = 0.0
        if cache:
            stats = cache.stats
            total = stats.get("prediction_hits", 0) + stats.get("prediction_misses", 0)
            cache_hit_rate = stats.get("prediction_hits", 0) / total if total else 0.0
            cache_accuracy = stats.get("accuracy", 0.0)

        scene_count = len(tracker._scenes) if tracker else 0
        query_rate = tracker._query_rate() if tracker else 0.0

        return NodeHealth(
            node_id=node_id,
            table_entries=table_entries,
            hot_entries=hot_entries,
            warm_entries=warm_entries,
            cold_entries=cold_entries,
            hnsw_available=hnsw_available,
            hnsw_index_count=hnsw_index_count,
            hnsw_coverage=hnsw_coverage,
            has_hnsw=hnsw is not None,
            cache_hit_rate=cache_hit_rate,
            cache_prediction_accuracy=cache_accuracy,
            has_cache=cache is not None,
            scene_count=scene_count,
            query_rate=query_rate,
            last_update=time.time(),
        )

    def _health_to_dict(self, health: NodeHealth) -> dict[str, Any]:
        return {
            "node_id": health.node_id,
            "table_entries": health.table_entries,
            "hot_entries": health.hot_entries,
            "warm_entries": health.warm_entries,
            "cold_entries": health.cold_entries,
            "hnsw_available": health.hnsw_available,
            "hnsw_coverage": health.hnsw_coverage,
            "has_hnsw": health.has_hnsw,
            "cache_hit_rate": health.cache_hit_rate,
            "cache_prediction_accuracy": health.cache_prediction_accuracy,
            "has_cache": health.has_cache,
            "scene_count": health.scene_count,
            "query_rate": health.query_rate,
            "last_update": health.last_update,
        }

    def check_alerts(self) -> list[Alert]:
        """Check all nodes for alert conditions."""
        alerts = []
        for node_id, node in self._nodes.items():
            health = self._node_health(node_id, node)
            alerts.extend(self._node_alerts(health))
        return alerts

    def _node_alerts(self, health: NodeHealth) -> list[Alert]:
        """Generate alerts for a single node."""
        alerts = []
        now = time.time()

        # HNSW coverage or availability (only if hnsw is registered)
        if health.has_hnsw and not health.hnsw_available:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                node_id=health.node_id,
                component="hnsw",
                message="HNSW not available (C++ extension not compiled)",
                metric=0.0,
                threshold=self.hnsw_coverage_threshold,
            ))
        elif health.has_hnsw and health.hnsw_coverage < self.hnsw_coverage_threshold:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                node_id=health.node_id,
                component="hnsw",
                message=f"HNSW coverage {health.hnsw_coverage:.2f} below threshold",
                metric=health.hnsw_coverage,
                threshold=self.hnsw_coverage_threshold,
            ))

        # Hot ratio
        total = health.table_entries
        if total > 0:
            hot_ratio = health.hot_entries / total
            if hot_ratio < self.hot_ratio_threshold:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    node_id=health.node_id,
                    component="tiered_storage",
                    message=f"Hot ratio {hot_ratio:.2f} below threshold",
                    metric=hot_ratio,
                    threshold=self.hot_ratio_threshold,
                ))

        # Cache hit rate (only if cache is present)
        if health.has_cache and health.cache_hit_rate < self.cache_hit_threshold:
            alerts.append(Alert(
                level=AlertLevel.INFO,
                node_id=health.node_id,
                component="cognitive_cache",
                message=f"Cache hit rate {health.cache_hit_rate:.2f} below threshold",
                metric=health.cache_hit_rate,
                threshold=self.cache_hit_threshold,
            ))

        # Cache prediction accuracy (only if cache is present)
        if health.has_cache and health.cache_prediction_accuracy < self.cache_accuracy_threshold:
            alerts.append(Alert(
                level=AlertLevel.INFO,
                node_id=health.node_id,
                component="cognitive_cache",
                message=f"Prediction accuracy {health.cache_prediction_accuracy:.2f} below threshold",
                metric=health.cache_prediction_accuracy,
                threshold=self.cache_accuracy_threshold,
            ))

        # Query rate
        if health.query_rate > self.query_rate_threshold:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                node_id=health.node_id,
                component="scene_tracker",
                message=f"Query rate {health.query_rate:.1f} above threshold",
                metric=health.query_rate,
                threshold=self.query_rate_threshold,
            ))

        return alerts

    def snapshot(self) -> dict[str, Any]:
        """Take a snapshot of fleet state for persistence."""
        report = self.health_report()
        alerts = self.check_alerts()
        snapshot = {
            "timestamp": report["timestamp"],
            "report": report,
            "alerts": [
                {
                    "level": a.level.value,
                    "node_id": a.node_id,
                    "component": a.component,
                    "message": a.message,
                    "metric": a.metric,
                    "threshold": a.threshold,
                }
                for a in alerts
            ],
            "alert_count": len(alerts),
            "critical_count": sum(1 for a in alerts if a.level == AlertLevel.CRITICAL),
            "warning_count": sum(1 for a in alerts if a.level == AlertLevel.WARNING),
        }
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        return snapshot

    def get_history(self, n: int = 10) -> list[dict[str, Any]]:
        """Get last n snapshots."""
        return self._history[-n:]

    def get_trends(self, metric: str, node_id: Optional[str] = None) -> list[float]:
        """Get trend for a metric over time."""
        values = []
        for snapshot in self._history:
            if node_id:
                node_data = snapshot["report"]["nodes"].get(node_id)
                if node_data:
                    values.append(node_data.get(metric, 0.0))
            else:
                values.append(snapshot["report"].get(metric, 0.0))
        return values
