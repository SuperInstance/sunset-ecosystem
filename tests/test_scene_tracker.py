"""Tests for SceneTracker.

Covers:
- Query pattern tracking
- Pattern histogram counting
- Latency statistics
- Scene detection
- Co-occurrence tracking
- Cache recommendations
- Performance metrics
- Stats tracking
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.scene_tracker import SceneTracker, CacheStrategy, QueryPattern


@pytest.fixture
def base_table() -> MeshVectorTable:
    return MeshVectorTable(table_id="test_scenes")


@pytest.fixture
def tracker(base_table: MeshVectorTable) -> SceneTracker:
    return SceneTracker(base_table, strategy=CacheStrategy(
        hot_threshold_accesses=2,
        scene_timeout_seconds=1.0,  # short for fast tests
    ))


class TestQueryTracking:
    def test_track_single(self, tracker: SceneTracker) -> None:
        tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0, query_params={"agent_id": "a"})
        assert tracker._query_count == 1
        assert tracker.stats["total_queries"] == 1

    def test_track_multiple(self, tracker: SceneTracker) -> None:
        for i in range(5):
            tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0, query_params={"agent_id": f"a{i}"})
        assert tracker._query_count == 5

    def test_histogram(self, tracker: SceneTracker) -> None:
        for _ in range(3):
            tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        for _ in range(2):
            tracker.track_query("similarity", "fitness", result_size=5, latency_ms=20.0)
        hot = tracker.get_hot_queries(k=2)
        assert len(hot) == 2
        assert hot[0][1] >= hot[1][1]


class TestLatencyStats:
    def test_latency_stats(self, tracker: SceneTracker) -> None:
        tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        tracker.track_query("by_id", "none", result_size=1, latency_ms=20.0)
        tracker.track_query("similarity", "fitness", result_size=5, latency_ms=30.0)

        stats = tracker.get_latency_stats()
        assert "by_id" in stats
        assert stats["by_id"]["count"] == 2
        assert stats["by_id"]["mean_ms"] == 15.0
        assert "overall" in stats


class TestSceneDetection:
    def test_scene_created(self, tracker: SceneTracker) -> None:
        tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        scene = tracker.get_current_scene()
        assert scene is not None
        assert len(scene.queries) == 2

    def test_scene_timeout(self, tracker: SceneTracker) -> None:
        tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        time.sleep(1.5)  # Wait for timeout
        tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        # Previous scene should be closed, new one started
        scenes = tracker.detect_scenes()
        assert len(scenes) >= 1

    def test_dominant_pattern(self, tracker: SceneTracker) -> None:
        for _ in range(3):
            tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        for _ in range(1):
            tracker.track_query("similarity", "fitness", result_size=5, latency_ms=20.0)
        scene = tracker.get_current_scene()
        assert scene is not None
        assert scene.frequency == 3


class TestCacheRecommendations:
    def test_high_access_promotion(self, base_table: MeshVectorTable, tracker: SceneTracker) -> None:
        # Insert entries
        for i in range(5):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=np.array([float(i), 0.0], dtype=np.float32),
                timestamp=1000.0,
                node_id="test",
                generation=0,
                fitness=0.8,
                signature=f"test_signature_{i}",
            )
            base_table.insert(entry, skip_verify=True)

        # Track queries for same agent multiple times
        for _ in range(3):
            tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0, query_params={"agent_id": "agent_0"})

        recs = tracker.get_cache_recommendations()
        assert "agent_0" in recs

    def test_cooccurrence_preload(self, base_table: MeshVectorTable, tracker: SceneTracker) -> None:
        # Insert entries
        for i in range(5):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=np.array([float(i), 0.0], dtype=np.float32),
                timestamp=1000.0,
                node_id="test",
                generation=0,
                fitness=0.8,
                signature=f"test_signature_{i}",
            )
            base_table.insert(entry, skip_verify=True)

        # Track alternating queries: agent_0 -> agent_1 -> agent_0 -> agent_1
        for _ in range(3):
            tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0, query_params={"agent_id": "agent_0"})
            tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0, query_params={"agent_id": "agent_1"})

        recs = tracker.get_cache_recommendations()
        # agent_0 should recommend agent_1 via co-occurrence
        assert "agent_1" in recs


class TestStats:
    def test_stats(self, tracker: SceneTracker) -> None:
        tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        tracker.track_query("similarity", "fitness", result_size=5, latency_ms=20.0)
        stats = tracker.stats
        assert stats["total_queries"] == 2
        assert stats["unique_patterns"] == 2
        assert stats["mean_latency_ms"] == 15.0

    def test_query_rate(self, tracker: SceneTracker) -> None:
        for _ in range(10):
            tracker.track_query("by_id", "none", result_size=1, latency_ms=10.0)
        stats = tracker.stats
        assert stats["query_rate_per_minute"] >= 10
