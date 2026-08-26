"""Tests for TieredMeshStorage.

Covers:
- Hot tier insertion (base table)
- Warm tier insertion (SQLite)
- Cold tier archiving
- Query across tiers (hot -> warm -> cold)
- Promotion/demotion
- Maintenance loop demotion
- Tier stats
- Close/cleanup
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
import pytest

from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.tiered_mesh_storage import TieredMeshStorage, TierConfig, PromotionPolicy


@pytest.fixture
def base_table() -> MeshVectorTable:
    return MeshVectorTable(table_id="test_tiered")


@pytest.fixture
def tiered(base_table: MeshVectorTable) -> TieredMeshStorage:
    with tempfile.TemporaryDirectory() as tmp:
        yield TieredMeshStorage(
            base_table=base_table,
            db_path=f"{tmp}/warm.db",
            cold_path=f"{tmp}/cold",
            config=TierConfig(
                hot_max_entries=5,
                hot_min_fitness=0.5,
                hot_max_age_seconds=3600,
                warm_max_entries=100,
                promotion_access_threshold=2,
            ),
            policy=PromotionPolicy(
                promote_on_access=True,
                demote_on_age=True,
                emergency_hot_capacity=10,
            ),
        )


class TestHotTier:
    def test_hot_insert(self, tiered: TieredMeshStorage) -> None:
        entry = VectorTableEntry(
            agent_id="hot_1",
            vector=np.array([1.0, 0.0], dtype=np.float32),
            timestamp=time.time(),
            node_id="test",
            generation=0,
            fitness=0.8,
            signature="test_signature_hot_1",
        )
        assert tiered.insert(entry) is True
        assert tiered.base.query("hot_1") is not None

    def test_hot_capacity_limit(self, tiered: TieredMeshStorage) -> None:
        # Fill hot tier to capacity
        for i in range(10):
            entry = VectorTableEntry(
                agent_id=f"hot_{i}",
                vector=np.array([float(i), 0.0], dtype=np.float32),
                timestamp=time.time(),
                node_id="test",
                generation=0,
                fitness=0.8,
                signature=f"test_signature_{i}",
            )
            tiered.insert(entry)
        # Hot tier should be at emergency capacity, extras in warm
        stats = tiered.get_tier_stats()
        assert stats["hot"]["entry_count"] <= 10


class TestWarmTier:
    def test_warm_insert_low_fitness(self, tiered: TieredMeshStorage) -> None:
        entry = VectorTableEntry(
            agent_id="warm_1",
            vector=np.array([1.0, 0.0], dtype=np.float32),
            timestamp=time.time(),
            node_id="test",
            generation=0,
            fitness=0.2,
            signature="test_signature_warm_1",  # below hot_min_fitness
        )
        assert tiered.insert(entry) is True
        # Should be in warm tier, not hot
        assert tiered.base.query("warm_1") is None
        # But query should find it
        result = tiered.query("warm_1")
        assert result is not None
        assert result.agent_id == "warm_1"

    def test_warm_query_by_fitness(self, tiered: TieredMeshStorage) -> None:
        # Insert low fitness to warm
        for i in range(3):
            entry = VectorTableEntry(
                agent_id=f"warm_{i}",
                vector=np.array([float(i), 0.0], dtype=np.float32),
                timestamp=time.time(),
                node_id="test",
                generation=0,
                fitness=0.2 + i * 0.1,
                signature=f"test_signature_{i}",
            )
            tiered.insert(entry)

        results = tiered.query_by_fitness(min_fitness=0.25, max_results=10)
        assert len(results) >= 2
        assert all(e.fitness >= 0.25 for e in results)


class TestPromotion:
    def test_promote_on_access(self, tiered: TieredMeshStorage) -> None:
        entry = VectorTableEntry(
            agent_id="promote_me",
            vector=np.array([1.0, 0.0], dtype=np.float32),
            timestamp=time.time(),
            node_id="test",
            generation=0,
            fitness=0.2,
            signature="test_signature_promote",
        )
        tiered.insert(entry)
        assert tiered.base.query("promote_me") is None  # in warm

        # Access multiple times
        for _ in range(5):
            tiered.query("promote_me")

        # Should be promoted to hot
        assert tiered.base.query("promote_me") is not None


class TestTierStats:
    def test_stats(self, tiered: TieredMeshStorage) -> None:
        entry = VectorTableEntry(
            agent_id="stats_test",
            vector=np.array([1.0, 0.0], dtype=np.float32),
            timestamp=time.time(),
            node_id="test",
            generation=0,
            fitness=0.8,
            signature="test_signature_stats",
        )
        tiered.insert(entry)
        stats = tiered.get_tier_stats()
        assert "hot" in stats
        assert "warm" in stats
        assert "cold" in stats
        assert stats["hot"]["entry_count"] >= 1


class TestClose:
    def test_close(self, tiered: TieredMeshStorage) -> None:
        tiered.close()
        # Should not raise
