"""Tests for FleetMemory.

Covers:
- Shard creation and time partitioning
- Remember and recall operations
- Similarity search across shards
- Temporal queries
- Shard eviction (LRU)
- Fleet memory stats
- Close/cleanup
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fleet.fleet_memory import FleetMemory, TemporalQuery, MemoryEntry


@pytest.fixture
def fleet_memory() -> FleetMemory:
    return FleetMemory(
        node_id="test_node",
        dim=2,
        shard_duration_seconds=86400.0,  # 1 day shards
        max_active_shards=3,
    )


class TestRememberAndRecall:
    def test_remember_single(self, fleet_memory: FleetMemory) -> None:
        ok = fleet_memory.remember(
            agent_id="agent_1",
            vector=[1.0, 0.0],
            fitness=0.8,
            context={"task": "test", "result": "success"},
        )
        assert ok is True
        assert fleet_memory._memories_added == 1

    def test_recall_by_similarity(self, fleet_memory: FleetMemory) -> None:
        # Store memories at different positions
        for i in range(10):
            angle = 2 * np.pi * i / 10
            fleet_memory.remember(
                agent_id=f"agent_{i}",
                vector=[np.cos(angle), np.sin(angle)],
                fitness=0.5 + i * 0.05,
                context={"index": i},
            )

        # Recall near angle 0 (should find agent_0 or nearby)
        results = fleet_memory.recall_similar(
            vector=[1.0, 0.0],
            k=3,
        )
        assert len(results) > 0
        assert len(results) <= 3
        # First result should be close to [1, 0]
        assert results[0].entry.agent_id in ["agent_0", "agent_1", "agent_9"]

    def test_recall_temporal_filter(self, fleet_memory: FleetMemory) -> None:
        now = time.time()
        # Store one memory now, one old
        fleet_memory.remember(
            agent_id="recent",
            vector=[1.0, 0.0],
            timestamp=now,
            fitness=0.8,
        )
        fleet_memory.remember(
            agent_id="old",
            vector=[0.0, 1.0],
            timestamp=now - 200000,  # > 2 days ago
            fitness=0.8,
        )

        # Query only recent (last 1 hour)
        results = fleet_memory.recall(TemporalQuery(
            start_time=now - 3600,
            end_time=now + 3600,
            similarity_vector=np.array([1.0, 0.0], dtype=np.float32),
            similarity_k=5,
        ))
        # Should only find recent memory
        assert all(r.entry.agent_id == "recent" for r in results)

    def test_recall_fitness_filter(self, fleet_memory: FleetMemory) -> None:
        fleet_memory.remember(
            agent_id="high_fitness", vector=[1.0, 0.0], fitness=0.9,
        )
        fleet_memory.remember(
            agent_id="low_fitness", vector=[1.0, 0.0], fitness=0.3,
        )

        results = fleet_memory.recall_similar(
            vector=[1.0, 0.0],
            k=10,
            min_fitness=0.5,
        )
        assert all(r.entry.fitness >= 0.5 for r in results)
        assert all(r.entry.agent_id == "high_fitness" for r in results)

    def test_recall_agent_filter(self, fleet_memory: FleetMemory) -> None:
        fleet_memory.remember(agent_id="alice", vector=[1.0, 0.0], fitness=0.8)
        fleet_memory.remember(agent_id="bob", vector=[1.0, 0.0], fitness=0.8)

        results = fleet_memory.recall(TemporalQuery(
            agent_id_filter="alice",
            similarity_vector=np.array([1.0, 0.0], dtype=np.float32),
            similarity_k=10,
        ))
        assert all(r.entry.agent_id == "alice" for r in results)

    def test_recall_keyword_filter(self, fleet_memory: FleetMemory) -> None:
        fleet_memory.remember(
            agent_id="task1", vector=[1.0, 0.0], fitness=0.8,
            context={"task": "deploy kubernetes", "result": "success"},
        )
        fleet_memory.remember(
            agent_id="task2", vector=[1.0, 0.0], fitness=0.8,
            context={"task": "deploy lambda", "result": "failure"},
        )

        results = fleet_memory.recall(TemporalQuery(
            keyword_filter="kubernetes",
            similarity_vector=np.array([1.0, 0.0], dtype=np.float32),
            similarity_k=10,
        ))
        assert all(r.entry.agent_id == "task1" for r in results)


class TestShardManagement:
    def test_shard_creation(self, fleet_memory: FleetMemory) -> None:
        fleet_memory.remember(agent_id="a", vector=[1.0, 0.0])
        shard_id = fleet_memory._timestamp_to_shard_id(time.time())
        assert shard_id in fleet_memory._shards
        assert len(fleet_memory._shards) == 1

    def test_shard_eviction(self, fleet_memory: FleetMemory) -> None:
        # Create memories across multiple days to trigger multiple shards
        now = time.time()
        for day in range(5):
            fleet_memory.remember(
                agent_id=f"day_{day}",
                vector=[float(day), 0.0],
                timestamp=now - day * 86400,
            )
        # With max_active_shards=3, should have evicted oldest
        assert len(fleet_memory._shards) <= 3

    def test_shard_report(self, fleet_memory: FleetMemory) -> None:
        fleet_memory.remember(agent_id="a", vector=[1.0, 0.0])
        report = fleet_memory.get_shard_report()
        assert len(report) >= 1


class TestStats:
    def test_memory_stats(self, fleet_memory: FleetMemory) -> None:
        for i in range(5):
            fleet_memory.remember(
                agent_id=f"agent_{i}",
                vector=[float(i), 0.0],
                fitness=0.8,
            )
        stats = fleet_memory.get_memory_stats()
        assert stats["memories_added"] == 5
        assert stats["node_id"] == "test_node"
        assert stats["shards_active"] >= 1

    def test_query_count(self, fleet_memory: FleetMemory) -> None:
        fleet_memory.remember(agent_id="a", vector=[1.0, 0.0])
        fleet_memory.recall_similar(vector=[1.0, 0.0], k=1)
        fleet_memory.recall_similar(vector=[1.0, 0.0], k=1)
        stats = fleet_memory.get_memory_stats()
        assert stats["queries_served"] == 2


class TestClose:
    def test_close(self, fleet_memory: FleetMemory) -> None:
        fleet_memory.remember(agent_id="a", vector=[1.0, 0.0])
        fleet_memory.close()
        assert len(fleet_memory._shards) == 0
