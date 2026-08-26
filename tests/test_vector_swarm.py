"""Tests for VectorSwarm.

Covers:
- SwarmRouter query planning
- Distributed query by ID
- Distributed similarity search
- Distributed KNN with global ranking
- Fitness range queries
- Consensus ranking
- Stats tracking
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.vector_swarm import VectorSwarm, SwarmRouter, SwarmQueryPlan


@pytest.fixture
def router() -> SwarmRouter:
    r = SwarmRouter()
    # Node 1: 2 shards
    node1 = MeshVectorTable(table_id="node1")
    r.register_node("node1", ["shard_1a", "shard_1b"], node1)
    # Node 2: 1 shard
    node2 = MeshVectorTable(table_id="node2")
    r.register_node("node2", ["shard_2a"], node2)
    return r


@pytest.fixture
def swarm(router: SwarmRouter) -> VectorSwarm:
    return VectorSwarm(router, max_workers=2)


def populate_table(table: MeshVectorTable, prefix: str, count: int) -> None:
    for i in range(count):
        entry = VectorTableEntry(
            agent_id=f"{prefix}_{i}",
            vector=np.array([float(i), 0.0], dtype=np.float32),
            timestamp=1000.0,
            node_id="test",
            generation=0,
            fitness=0.5 + i * 0.05,
            signature=f"test_signature_{prefix}_{i}",
        )
        table.insert(entry, skip_verify=True)


class TestSwarmRouter:
    def test_register_and_route(self, router: SwarmRouter) -> None:
        plan = router.route_query("id", {"agent_id": "test_agent"})
        assert plan.query_type == "id"
        assert len(plan.target_nodes) > 0

    def test_route_by_hash(self, router: SwarmRouter) -> None:
        nodes = router._route_by_hash("some_key")
        assert len(nodes) >= 1

    def test_generate_query_id(self) -> None:
        qid1 = SwarmRouter._generate_query_id("knn", {"k": 5})
        qid2 = SwarmRouter._generate_query_id("knn", {"k": 5})
        assert qid1 != qid2  # includes timestamp

    def test_plan_required_responses(self, router: SwarmRouter) -> None:
        plan = router.route_query("id", {"agent_id": "a"})
        plan.consistency = "quorum"
        assert plan.required_responses >= 1


class TestDistributedQueryById:
    def test_query_by_id(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        # Populate both nodes
        node1 = router._node_index["node1"]["node_ref"]
        node2 = router._node_index["node2"]["node_ref"]
        populate_table(node1, "n1", 5)
        populate_table(node2, "n2", 5)

        results = swarm.query_by_id("n1_2", consistency="all")
        assert len(results) > 0
        # At least one result should have the entry
        found = any(e.agent_id == "n1_2" for r in results for e in r.entries)
        assert found


class TestDistributedSimilarity:
    def test_query_similar(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        node1 = router._node_index["node1"]["node_ref"]
        node2 = router._node_index["node2"]["node_ref"]
        populate_table(node1, "n1", 5)
        populate_table(node2, "n2", 5)

        query_vec = np.array([2.0, 0.0], dtype=np.float32)
        results = swarm.query_similar(query_vec, k=3, consistency="all")
        assert len(results) > 0
        total_entries = sum(len(r.entries) for r in results)
        assert total_entries > 0


class TestDistributedKnn:
    def test_query_knn(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        node1 = router._node_index["node1"]["node_ref"]
        node2 = router._node_index["node2"]["node_ref"]
        populate_table(node1, "n1", 5)
        populate_table(node2, "n2", 5)

        query_vec = np.array([2.0, 0.0], dtype=np.float32)
        results = swarm.query_knn(query_vec, k=3, consistency="all")
        assert len(results) <= 3
        assert len(results) > 0
        # Results should be sorted by distance
        if len(results) > 1:
            assert results[0][1] <= results[1][1]

    def test_knn_deduplication(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        # Same agent on both nodes (simulating replication)
        node1 = router._node_index["node1"]["node_ref"]
        node2 = router._node_index["node2"]["node_ref"]
        entry = VectorTableEntry(
            agent_id="shared_agent",
            vector=np.array([1.0, 0.0], dtype=np.float32),
            timestamp=1000.0,
            node_id="test",
            generation=0,
            fitness=0.8,
            signature="test_signature_shared",
        )
        node1.insert(entry, skip_verify=True)
        node2.insert(entry, skip_verify=True)

        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        results = swarm.query_knn(query_vec, k=5, consistency="all")
        # Should only have one copy of shared_agent
        agent_ids = [e.agent_id for e, _ in results]
        assert agent_ids.count("shared_agent") == 1


class TestFitnessRange:
    def test_query_fitness_range(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        node1 = router._node_index["node1"]["node_ref"]
        node2 = router._node_index["node2"]["node_ref"]
        populate_table(node1, "n1", 5)
        populate_table(node2, "n2", 5)

        results = swarm.query_fitness_range(
            min_fitness=0.6, max_fitness=0.8, consistency="all"
        )
        assert len(results) > 0
        for result in results:
            for entry in result.entries:
                assert 0.6 <= entry.fitness <= 0.8


class TestConsensusRanking:
    def test_consensus_rank(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        node1 = router._node_index["node1"]["node_ref"]
        node2 = router._node_index["node2"]["node_ref"]
        populate_table(node1, "n1", 5)
        populate_table(node2, "n2", 5)

        query_vec = np.array([2.0, 0.0], dtype=np.float32)
        plan = router.route_query("knn", {"vector": query_vec.tolist(), "k": 5})
        plan.consistency = "all"
        results = swarm._execute_plan(plan)

        ranked = swarm.consensus_rank(results, query_vec)
        assert len(ranked) > 0


class TestStats:
    def test_stats(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        stats = swarm.stats
        assert stats["node_count"] == 2
        assert stats["query_count"] == 0
        assert stats["success_count"] == 0

    def test_stats_after_query(self, router: SwarmRouter, swarm: VectorSwarm) -> None:
        node1 = router._node_index["node1"]["node_ref"]
        populate_table(node1, "n1", 3)

        swarm.query_by_id("n1_0", consistency="all")
        stats = swarm.stats
        assert stats["query_count"] >= 1
        assert stats["success_count"] >= 1
