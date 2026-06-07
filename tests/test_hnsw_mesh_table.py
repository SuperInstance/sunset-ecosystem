"""Tests for HnswMeshTable.

Covers:
- HNSW index construction from existing entries
- KNN search (with and without hnswlib)
- Range search
- Novelty neighbors and local density
- Sparse region detection
- Auto-rebuild trigger
- Stats tracking
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry
from swarm.hnsw_mesh_table import HnswMeshTable, HnswIndexConfig


@pytest.fixture
def base_table() -> MeshVectorTable:
    return MeshVectorTable(table_id="test_hnsw")


@pytest.fixture
def sample_entries() -> list[VectorTableEntry]:
    """Create 20 entries with distinct vectors in 2D for easy verification."""
    entries = []
    for i in range(20):
        angle = 2 * np.pi * i / 20
        vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        entries.append(VectorTableEntry(
            agent_id=f"agent_{i:03d}",
            vector=vec,
            timestamp=1000.0 + i,
            node_id="test_node",
            generation=i,
            fitness=0.5 + i * 0.025,
            signature=f"test_signature_{i:03d}",
        ))
    return entries


class TestIndexConstruction:
    def test_build_from_empty(self, base_table: MeshVectorTable) -> None:
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))
        assert hnsw.stats["index_count"] == 0

    def test_build_from_existing(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))
        if hnsw.stats["hnsw_available"]:
            assert hnsw.stats["index_count"] == 20
        else:
            assert hnsw.stats["index_count"] == 0  # fallback mode


class TestKnnSearch:
    def test_knn_basic(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100, space="l2"))

        # Query near agent_0 (angle 0 = [1, 0])
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = hnsw.knn_search(query, k=5)
        assert len(results) <= 5
        assert len(results) > 0
        # First result should be close to agent_0
        assert results[0][0].agent_id in ["agent_000", "agent_001", "agent_019"]

    def test_knn_with_filter(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))

        query = np.array([1.0, 0.0], dtype=np.float32)
        # Filter: only high fitness
        results = hnsw.knn_search(query, k=10, filter_fn=lambda e: e.fitness >= 0.7)
        assert all(e.fitness >= 0.7 for e, _ in results)


class TestRangeSearch:
    def test_range_search(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = hnsw.range_search(query, radius=0.5, max_results=10)
        assert all(d <= 0.5 for _, d in results)


class TestNoveltyAndDensity:
    def test_novelty_neighbors(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))

        entry = sample_entries[0]
        neighbors = hnsw.get_novelty_neighbors(entry, k=5)
        assert len(neighbors) <= 5
        # All neighbors should be different from self
        assert all(e.agent_id != entry.agent_id for e, _ in neighbors)

    def test_local_density(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))

        entry = sample_entries[0]
        density = hnsw.compute_local_density(entry, k=5)
        assert density > 0

    def test_find_sparse_regions(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))

        sparse = hnsw.find_sparse_regions(k=3, n_samples=10)
        assert len(sparse) <= 10
        assert len(sparse) > 0
        # Should be sorted by density ascending
        if len(sparse) > 1:
            assert sparse[0][1] <= sparse[1][1]


class TestInsertAndRebuild:
    def test_insert_updates_index(self, base_table: MeshVectorTable) -> None:
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))
        entry = VectorTableEntry(
            agent_id="agent_new",
            vector=np.array([1.0, 0.0], dtype=np.float32),
            timestamp=1000.0,
            node_id="test",
            generation=0,
            fitness=0.8,
            signature="test_signature_new",
        )
        ok = hnsw.insert(entry)
        assert ok is True
        if hnsw.stats["hnsw_available"]:
            assert hnsw.stats["index_count"] >= 1
        else:
            assert hnsw.stats["index_count"] == 0  # fallback mode

    def test_auto_rebuild(self, base_table: MeshVectorTable) -> None:
        # Small threshold to trigger rebuild
        hnsw = HnswMeshTable(
            base_table,
            config=HnswIndexConfig(dim=2, max_elements=1000),
            auto_rebuild_threshold=0.05,  # 5% change triggers rebuild
        )
        # Insert many entries to trigger rebuild
        for i in range(10):
            entry = VectorTableEntry(
                agent_id=f"agent_{i}",
                vector=np.random.randn(2).astype(np.float32),
                timestamp=1000.0 + i,
                node_id="test",
                generation=0,
                fitness=0.5,
                signature=f"test_signature_{i}",
            )
            hnsw.insert(entry)
        if hnsw.stats["hnsw_available"]:
            assert hnsw.stats["rebuild_count"] >= 1
        else:
            assert hnsw.stats["rebuild_count"] == 0  # no rebuilds in fallback


class TestStats:
    def test_stats(self, base_table: MeshVectorTable, sample_entries: list[VectorTableEntry]) -> None:
        for e in sample_entries:
            base_table.insert(e, skip_verify=True)
        hnsw = HnswMeshTable(base_table, config=HnswIndexConfig(dim=2, max_elements=100))
        stats = hnsw.stats
        assert stats["table_id"] == "test_hnsw"
        if stats["hnsw_available"]:
            assert stats["index_count"] == 20
        else:
            assert stats["index_count"] == 0
        assert stats["entry_count"] == 20
