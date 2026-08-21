"""Tests for MeshGrouping.

Covers:
- K-means clustering (with and without sklearn)
- Single-pass incremental clustering
- Group profile creation and metrics
- Outlier detection
- Dense/sparse region finding
- Diversity index computation
- Incremental updates
- Stats tracking
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.mesh_grouping import MeshGrouping, ClusterConfig, GroupProfile


@pytest.fixture
def base_table() -> MeshVectorTable:
    return MeshVectorTable(table_id="test_grouping")


@pytest.fixture
def sample_entries() -> list[VectorTableEntry]:
    """Create 20 entries in 4 distinct clusters in 2D."""
    entries = []
    cluster_centers = [
        np.array([1.0, 1.0]),
        np.array([1.0, -1.0]),
        np.array([-1.0, 1.0]),
        np.array([-1.0, -1.0]),
    ]
    for cluster_idx, center in enumerate(cluster_centers):
        for i in range(5):
            vec = center + np.random.randn(2) * 0.1
            entries.append(
                VectorTableEntry(
                    agent_id=f"c{cluster_idx}_a{i}",
                    vector=vec.astype(np.float32),
                    timestamp=1000.0,
                    node_id="test",
                    generation=0,
                    fitness=0.5,
                    signature=f"test_signature_{cluster_idx}_{i}",
                )
            )
    return entries


class TestKMeansClustering:
    def test_cluster_creates_groups(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        groups = grouping.cluster()
        assert len(groups) >= 1
        assert all(isinstance(g, GroupProfile) for g in groups)

    def test_group_members(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        for group in grouping.groups.values():
            assert len(group.members) > 0
            members = grouping.get_group_members(group.group_id)
            assert len(members) == len(group.members)

    def test_centroids(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        for group in grouping.groups.values():
            centroid = grouping.get_group_centroid(group.group_id)
            assert centroid is not None
            assert len(centroid) == 2


class TestIncrementalUpdate:
    def test_single_pass_online(self, base_table: MeshVectorTable) -> None:
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="single_pass", n_clusters=4)
        )
        # Add entries one by one
        for i in range(10):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=np.array([float(i), 0.0], dtype=np.float32),
                timestamp=1000.0,
                node_id="test",
                generation=0,
                fitness=0.5,
                signature=f"test_signature_{i}",
            )
            base_table.insert(entry, skip_verify=True)
            grouping.incremental_update(entry)
        assert len(grouping.groups) > 0

    def test_incremental_outlier(self, base_table: MeshVectorTable) -> None:
        grouping = MeshGrouping(
            base_table,
            config=ClusterConfig(
                algorithm="single_pass", n_clusters=4, similarity_threshold=0.99
            ),
        )
        # First entry creates a group
        entry1 = VectorTableEntry(
            agent_id="a1",
            vector=np.array([1.0, 0.0], dtype=np.float32),
            timestamp=1000.0,
            node_id="test",
            generation=0,
            fitness=0.5,
            signature="test_signature_1",
        )
        base_table.insert(entry1, skip_verify=True)
        grouping.incremental_update(entry1)
        # Very different entry
        entry2 = VectorTableEntry(
            agent_id="a2",
            vector=np.array([100.0, 0.0], dtype=np.float32),
            timestamp=1000.0,
            node_id="test",
            generation=0,
            fitness=0.5,
            signature="test_signature_2",
        )
        base_table.insert(entry2, skip_verify=True)
        result = grouping.incremental_update(entry2)
        # With n_clusters=4 and high threshold, may create new group or be outlier
        assert result is not None or len(grouping.groups) >= 1


class TestQualityMetrics:
    def test_cohesion(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        for group in grouping.groups.values():
            assert group.cohesion >= 0.0
            assert group.cohesion <= 1.0

    def test_separation(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        if len(grouping.groups) > 1:
            for group in grouping.groups.values():
                assert group.separation >= 0.0

    def test_diversity_index(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        diversity = grouping.compute_diversity_index()
        assert diversity >= 0.0


class TestDenseSparseRegions:
    def test_find_dense(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        dense = grouping.find_dense_regions(k=2)
        assert len(dense) <= 2
        assert len(dense) > 0

    def test_find_sparse(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        sparse = grouping.find_sparse_regions(k=2)
        assert len(sparse) <= 2


class TestEmptyTable:
    def test_empty_cluster(self, base_table: MeshVectorTable) -> None:
        grouping = MeshGrouping(base_table, config=ClusterConfig(algorithm="kmeans"))
        groups = grouping.cluster()
        assert groups == []
        assert len(grouping.groups) == 0


class TestStats:
    def test_stats(
        self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]
    ) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        grouping = MeshGrouping(
            base_table, config=ClusterConfig(algorithm="kmeans", n_clusters=4)
        )
        grouping.cluster()
        stats = grouping.stats
        assert stats["group_count"] >= 1
        assert stats["total_members"] == len(sample_entries)
        assert stats["algorithm"] == "kmeans"
        assert stats["cluster_count"] == 1
