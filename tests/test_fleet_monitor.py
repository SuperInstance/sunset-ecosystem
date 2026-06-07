"""Tests for FleetMonitor.

Covers:
- Node registration and health report
- Alert generation (HNSW, cache, query rate)
- Snapshot and history tracking
- Trend analysis
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
import pytest

from fleet.cognitive_cache import CognitiveCache, PredictionEngine
from fleet.fleet_monitor import AlertLevel, FleetMonitor, NodeHealth
from fleet.fleet_memory import FleetMemory
from swarm.hnsw_mesh_table import HnswMeshTable
from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.scene_tracker import SceneTracker, CacheStrategy
from swarm.tiered_mesh_storage import TieredMeshStorage, TierConfig


@pytest.fixture
def monitor() -> FleetMonitor:
    return FleetMonitor()


class TestRegistration:
    def test_register_node(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        monitor.register_node("node1", table)
        assert "node1" in monitor._nodes

    def test_register_with_all_components(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        storage = TieredMeshStorage(base_table=table)
        hnsw = HnswMeshTable(base_table=table)
        tracker = SceneTracker(table, strategy=CacheStrategy())
        cache = CognitiveCache(storage, tracker)

        monitor.register_node(
            "node1", table, storage=storage, hnsw=hnsw, cache=cache, tracker=tracker
        )
        assert "node1" in monitor._nodes


class TestHealthReport:
    def test_empty_report(self, monitor: FleetMonitor) -> None:
        report = monitor.health_report()
        assert report["node_count"] == 0
        assert report["total_entries"] == 0

    def test_single_node_report(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        for i in range(5):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=np.array([float(i), 0.0], dtype=np.float32),
                timestamp=time.time(),
                node_id="test",
                generation=0,
                fitness=0.5,
                signature=f"test_signature_{i}",
            )
            table.insert(entry, skip_verify=True)

        monitor.register_node("node1", table)
        report = monitor.health_report()
        assert report["node_count"] == 1
        assert report["total_entries"] == 5
        assert "node1" in report["nodes"]

    def test_multi_node_report(self, monitor: FleetMonitor) -> None:
        for i in range(3):
            table = MeshVectorTable(table_id=f"node_{i}")
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=np.array([1.0, 0.0], dtype=np.float32),
                timestamp=time.time(),
                node_id=f"node_{i}",
                generation=0,
                fitness=0.6,
                signature=f"test_{i}",
            )
            table.insert(entry, skip_verify=True)
            monitor.register_node(f"node_{i}", table)

        report = monitor.health_report()
        assert report["node_count"] == 3
        assert report["total_entries"] == 3


class TestAlerts:
    def test_hnsw_coverage_alert(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        hnsw = HnswMeshTable(base_table=table)
        monitor.register_node("node1", table, hnsw=hnsw)

        # HNSW coverage is 0 (no entries in index)
        alerts = monitor.check_alerts()
        hnsw_alerts = [a for a in alerts if a.component == "hnsw"]
        assert len(hnsw_alerts) == 1
        assert hnsw_alerts[0].level == AlertLevel.WARNING

    def test_cache_hit_alert(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        storage = TieredMeshStorage(base_table=table)
        tracker = SceneTracker(table, strategy=CacheStrategy())
        cache = CognitiveCache(storage, tracker)
        monitor.register_node("node1", table, storage=storage, cache=cache)

        # No cache hits yet
        alerts = monitor.check_alerts()
        cache_alerts = [a for a in alerts if a.component == "cognitive_cache"]
        assert len(cache_alerts) >= 1

    def test_query_rate_alert(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        tracker = SceneTracker(table, strategy=CacheStrategy())
        monitor.register_node("node1", table, tracker=tracker)
        monitor.query_rate_threshold = 0.0  # Any query rate triggers

        # Simulate queries
        for _ in range(5):
            tracker.track_query("by_id", "none", 1, 1.0, {"agent_id": "test"})

        alerts = monitor.check_alerts()
        rate_alerts = [a for a in alerts if a.component == "scene_tracker"]
        assert len(rate_alerts) >= 1
        assert rate_alerts[0].level == AlertLevel.CRITICAL

    def test_no_alerts_healthy(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        monitor.register_node("node1", table)

        # No thresholds should trigger with empty table
        alerts = monitor.check_alerts()
        assert len(alerts) == 0


class TestSnapshots:
    def test_snapshot(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        monitor.register_node("node1", table)

        snapshot = monitor.snapshot()
        assert "timestamp" in snapshot
        assert "report" in snapshot
        assert "alerts" in snapshot
        assert "alert_count" in snapshot

    def test_history(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        monitor.register_node("node1", table)

        for _ in range(5):
            monitor.snapshot()

        history = monitor.get_history(3)
        assert len(history) == 3

    def test_trends(self, monitor: FleetMonitor) -> None:
        table = MeshVectorTable(table_id="test")
        monitor.register_node("node1", table)

        for _ in range(5):
            monitor.snapshot()

        trends = monitor.get_trends("total_entries")
        assert len(trends) == 5
        assert all(t == 0 for t in trends)


class TestNodeHealth:
    def test_health_dataclass(self) -> None:
        health = NodeHealth(
            node_id="test",
            table_entries=100,
            hot_entries=50,
            warm_entries=30,
            cold_entries=20,
            hnsw_available=True,
            hnsw_index_count=80,
            hnsw_coverage=0.8,
            has_hnsw=True,
            cache_hit_rate=0.75,
            cache_prediction_accuracy=0.6,
            has_cache=True,
            scene_count=3,
            query_rate=10.0,
            last_update=time.time(),
        )
        assert health.node_id == "test"
        assert health.table_entries == 100
