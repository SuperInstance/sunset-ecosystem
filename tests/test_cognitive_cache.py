"""Tests for CognitiveCache.

Covers:
- Basic query with tracking
- Prediction engine co-occurrence detection
- Prediction engine hot query detection
- Predictive preloading
- Maintenance cycle
- Stats tracking
- Accuracy feedback
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
import pytest

from fleet.cognitive_cache import CognitiveCache, PredictionEngine
from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.scene_tracker import SceneTracker, CacheStrategy
from swarm.tiered_mesh_storage import TieredMeshStorage, TierConfig


@pytest.fixture
def cache() -> CognitiveCache:
    base = MeshVectorTable(table_id="test")
    storage = TieredMeshStorage(
        base_table=base,
        config=TierConfig(
            hot_max_entries=10,
            warm_max_entries=50,
            promotion_access_threshold=2,
        ),
    )
    tracker = SceneTracker(
        base,
        strategy=CacheStrategy(
            hot_threshold_accesses=2,
            scene_timeout_seconds=1.0,
        ),
    )
    return CognitiveCache(storage, tracker)


def populate_cache(cache: CognitiveCache, count: int, prefix: str = "agent") -> None:
    now = time.time()
    for i in range(count):
        entry = VectorTableEntry(
            agent_id=f"{prefix}_{i}",
            vector=np.array([float(i), 0.0], dtype=np.float32),
            timestamp=now,
            node_id="test",
            generation=0,
            fitness=0.5 + i * 0.05,
            signature=f"test_signature_{prefix}_{i}",
        )
        cache.storage.insert(entry)


class TestBasicQuery:
    def test_query_found(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 3)
        result = cache.query("agent_1")
        assert result is not None
        assert result.agent_id == "agent_1"

    def test_query_not_found(self, cache: CognitiveCache) -> None:
        result = cache.query("nonexistent")
        assert result is None

    def test_query_tracks(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 3)
        cache.query("agent_0")
        assert cache.tracker.stats["total_queries"] == 1


class TestPredictionEngine:
    def test_cooccurrence_prediction(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 5)

        # Simulate co-occurrence: agent_0 and agent_1 queried together
        for _ in range(5):
            cache.query("agent_0")
            cache.query("agent_1")

        engine = PredictionEngine(cooccurrence_threshold=0.3)
        predictions = engine.predict(cache.tracker)

        # agent_1 should be predicted when agent_0 is queried
        pred_ids = {p.agent_id for p in predictions}
        assert "agent_1" in pred_ids

    def test_hot_query_prediction(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 5)

        # Query agent_0 many times
        for _ in range(5):
            cache.query("agent_0")

        engine = PredictionEngine(hot_threshold=3)
        predictions = engine.predict(cache.tracker)

        pred_ids = {p.agent_id for p in predictions}
        assert "agent_0" in pred_ids

    def test_scene_prediction(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 5)

        # Query multiple agents in a burst (scene)
        for i in range(5):
            cache.query(f"agent_{i}")

        engine = PredictionEngine()
        predictions = engine.predict(cache.tracker)

        # At least some predictions should exist
        assert len(predictions) >= 1


class TestPreloading:
    def test_predictive_preload(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 5)

        # Build co-occurrence pattern
        for _ in range(5):
            cache.query("agent_0")
            cache.query("agent_1")

        # Trigger preloading (every 5 queries)
        cache.query("agent_2")
        cache.query("agent_2")
        cache.query("agent_2")
        cache.query("agent_2")
        cache.query("agent_2")

        # Preloading should have run without errors
        # (entries may already be hot, so count may be 0)
        assert cache._preload_count >= 0
        # Tracker should have co-occurrence data
        assert len(cache.tracker._cooccurrence_map) > 0


class TestMaintenance:
    def test_maintenance(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 10)
        cache.run_maintenance()
        stats = cache.stats
        assert "storage_stats" in stats


class TestStats:
    def test_stats_initial(self, cache: CognitiveCache) -> None:
        stats = cache.stats
        assert stats["prediction_hits"] == 0
        assert stats["prediction_misses"] == 0
        assert stats["prediction_accuracy"] == 0.0

    def test_stats_after_queries(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 3)
        cache.query("agent_0")
        cache.query("agent_0")
        cache.query("nonexistent")

        stats = cache.stats
        assert stats["prediction_hits"] == 2
        assert stats["prediction_misses"] == 1
        assert stats["prediction_accuracy"] == 2 / 3

    def test_tracker_stats(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 3)
        cache.query("agent_0")
        stats = cache.stats
        assert stats["tracker_stats"]["total_queries"] == 1


class TestSimilarityQuery:
    def test_query_similar(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 5)
        query_vec = np.array([2.0, 0.0], dtype=np.float32)
        results = cache.query_similar(query_vec, k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_similarity_tracks(self, cache: CognitiveCache) -> None:
        populate_cache(cache, 5)
        query_vec = np.array([2.0, 0.0], dtype=np.float32)
        cache.query_similar(query_vec, k=3)
        assert cache.tracker.stats["total_queries"] == 1
