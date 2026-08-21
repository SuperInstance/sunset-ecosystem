"""Tests for MeshTableStore SQLite persistence.

Run: python3 -m pytest tests/test_mesh_table_store.py -v --tb=short
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest

from swarm.mesh_table_store import MeshTableStore
from swarm.mesh_vector_tables import MeshVectorTable, VectorTableEntry, FleetVectorIndex


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = MeshTableStore(path)
    yield s
    os.unlink(path)


@pytest.fixture
def sample_entries():
    return [
        VectorTableEntry(
            agent_id=f"agent_{i}",
            vector=np.array([0.1 * i, 0.2 * i, 0.3 * i], dtype=np.float32),
            timestamp=time.time() - i,
            node_id="TestNode",
            generation=i,
            fitness=0.5 + 0.1 * i,
            signature="sha256:test",
        )
        for i in range(5)
    ]


# ── save / load roundtrip ───────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_table(self, store, sample_entries):
        table = MeshVectorTable(table_id="pool_a")
        for e in sample_entries:
            table._entries[e.agent_id] = e

        count = store.save_table(table)
        assert count == 5

        loaded = store.load_table("pool_a")
        assert loaded.table_id == "pool_a"
        assert len(loaded._entries) == 5

        # Verify data integrity
        for e in sample_entries:
            assert e.agent_id in loaded._entries
            le = loaded._entries[e.agent_id]
            np.testing.assert_array_almost_equal(le.vector, e.vector)
            assert le.fitness == pytest.approx(e.fitness)
            assert le.generation == e.generation

    def test_save_overwrites(self, store, sample_entries):
        table = MeshVectorTable(table_id="pool_a")
        table._entries[sample_entries[0].agent_id] = sample_entries[0]
        store.save_table(table)

        # Save fewer entries — should clear old and store new
        table2 = MeshVectorTable(table_id="pool_a")
        table2._entries[sample_entries[1].agent_id] = sample_entries[1]
        store.save_table(table2)

        loaded = store.load_table("pool_a")
        assert len(loaded._entries) == 1
        assert sample_entries[1].agent_id in loaded._entries

    def test_save_empty_table(self, store):
        table = MeshVectorTable(table_id="empty")
        count = store.save_table(table)
        assert count == 0
        loaded = store.load_table("empty")
        assert len(loaded._entries) == 0


# ── FleetVectorIndex ────────────────────────────────────────


class TestFleetVectorIndex:
    def test_save_and_load_index(self, store, sample_entries):
        index = FleetVectorIndex("TestNode")
        for i, tid in enumerate(["gen_0", "gen_1"]):
            table = index.get_gen_table(i)
            for e in sample_entries:
                table._entries[e.agent_id] = e

        counts = store.save_index(index)
        assert counts == {"gen_0": 5, "gen_1": 5}

        loaded = store.load_index("TestNode")
        assert len(loaded._gen_tables) == 2
        assert "gen_0" in [t.table_id for t in loaded._gen_tables.values()]
        assert "gen_1" in [t.table_id for t in loaded._gen_tables.values()]

    def test_load_index_with_prefix(self, store, sample_entries):
        index = FleetVectorIndex("TestNode")
        for i, name in enumerate(["alpha_pool", "alpha_test"]):
            table = MeshVectorTable(table_id=name)
            for e in sample_entries[:2]:
                table._entries[e.agent_id] = e
            index._gen_tables[i] = table

        store.save_index(index)

        alpha = store.load_index("TestNode", prefix="alpha")
        table_ids = [t.table_id for t in alpha._gen_tables.values()]
        assert len(table_ids) == 2
        assert "alpha_pool" in table_ids
        assert "alpha_test" in table_ids


# ── query ───────────────────────────────────────────────────


class TestQuery:
    def test_query_by_fitness(self, store, sample_entries):
        table = MeshVectorTable(table_id="pool")
        for e in sample_entries:
            table._entries[e.agent_id] = e
        store.save_table(table)

        results = store.query_by_fitness("pool", min_fitness=0.7)
        assert len(results) == 3  # fitness: 0.5, 0.6, 0.7, 0.8, 0.9

        results = store.query_by_fitness("pool", min_fitness=0.85)
        assert len(results) == 1
        assert results[0].fitness == pytest.approx(0.9)

    def test_count_entries(self, store, sample_entries):
        table = MeshVectorTable(table_id="pool")
        for e in sample_entries:
            table._entries[e.agent_id] = e
        store.save_table(table)

        assert store.count_entries() == 5
        assert store.count_entries("pool") == 5
        assert store.count_entries("nonexistent") == 0


# ── delete ──────────────────────────────────────────────────


class TestDelete:
    def test_drop_table(self, store, sample_entries):
        table = MeshVectorTable(table_id="pool")
        for e in sample_entries:
            table._entries[e.agent_id] = e
        store.save_table(table)

        assert store.count_entries() == 5
        deleted = store.drop_table("pool")
        assert deleted == 5
        assert store.count_entries() == 0

    def test_delete_older_than(self, store, sample_entries):
        table = MeshVectorTable(table_id="pool")
        for e in sample_entries:
            table._entries[e.agent_id] = e
        store.save_table(table)

        # All entries are from "now - i" seconds ago (i up to 4)
        # delete_older_than(0.1) should catch 4 of them (i=1..4), i=0 survives
        assert store.delete_older_than(0.1) == 4
        assert store.count_entries() == 1

    def test_delete_older_than_actual(self, store):
        old = VectorTableEntry(
            agent_id="old_agent",
            vector=np.array([1.0, 2.0], dtype=np.float32),
            timestamp=time.time() - 3600,
            node_id="TestNode",
            generation=0,
            fitness=0.5,
            signature="sha256:old",
        )
        new = VectorTableEntry(
            agent_id="new_agent",
            vector=np.array([3.0, 4.0], dtype=np.float32),
            timestamp=time.time(),
            node_id="TestNode",
            generation=1,
            fitness=0.6,
            signature="sha256:new",
        )
        table = MeshVectorTable(table_id="pool")
        table._entries["old_agent"] = old
        table._entries["new_agent"] = new
        store.save_table(table)

        assert store.count_entries() == 2
        assert store.delete_older_than(1800) == 1  # old_agent is 3600s old
        assert store.count_entries() == 1


# ── edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    def test_nonexistent_table(self, store):
        loaded = store.load_table("never_existed")
        assert loaded.table_id == "never_existed"
        assert len(loaded._entries) == 0

    def test_vector_precision(self, store):
        vec = np.array([1.234567, -9.876543, 0.000001], dtype=np.float32)
        entry = VectorTableEntry(
            agent_id="precise",
            vector=vec,
            timestamp=time.time(),
            node_id="TestNode",
            generation=0,
            fitness=0.99,
            signature="sha256:test",
        )
        table = MeshVectorTable(table_id="pool")
        table._entries["precise"] = entry
        store.save_table(table)

        loaded = store.load_table("pool")
        le = loaded._entries["precise"]
        np.testing.assert_array_almost_equal(le.vector, vec, decimal=5)

    def test_extra_json_roundtrip(self, store):
        entry = VectorTableEntry(
            agent_id="extra_test",
            vector=np.array([1.0, 2.0], dtype=np.float32),
            timestamp=time.time(),
            node_id="TestNode",
            generation=0,
            fitness=0.5,
            signature="sha256:test",
            extra={"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}},
        )
        table = MeshVectorTable(table_id="pool")
        table._entries["extra_test"] = entry
        store.save_table(table)

        loaded = store.load_table("pool")
        assert loaded._entries["extra_test"].extra == entry.extra

    def test_concurrent_writes(self, store, sample_entries):
        """Thread-safe test: multiple tables written concurrently."""
        import threading

        errors = []

        def writer(table_id, entries):
            try:
                table = MeshVectorTable(table_id=table_id)
                for e in entries:
                    table._entries[e.agent_id] = e
                store.save_table(table)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            t = threading.Thread(target=writer, args=(f"pool_{i}", sample_entries))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes failed: {errors}"
        assert store.count_entries() == 20  # 4 pools × 5 entries
