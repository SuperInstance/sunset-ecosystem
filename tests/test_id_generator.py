"""Tests for id_generator.py — ULID-style distributed ID generator.

Run: python3 -m pytest tests/test_id_generator.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.id_generator import IDGenerator, IDBatchGenerator


class TestIDGenerator:
    def test_create(self):
        gen = IDGenerator(node_id=1)
        assert gen.stats()["generated"] == 0

    def test_node_id_bounds(self):
        IDGenerator(node_id=0)
        IDGenerator(node_id=65535)
        with pytest.raises(ValueError):
            IDGenerator(node_id=65536)
        with pytest.raises(ValueError):
            IDGenerator(node_id=-1)

    def test_generate_returns_string(self):
        gen = IDGenerator(node_id=1)
        uid = gen.generate()
        assert isinstance(uid, str)
        assert len(uid) > 0

    def test_uniqueness(self):
        gen = IDGenerator(node_id=1)
        ids = {gen.generate() for _ in range(1000)}
        assert len(ids) == 1000

    def test_sortable_by_time(self):
        gen = IDGenerator(node_id=1)
        ids = []
        for _ in range(10):
            ids.append(gen.generate())
            time.sleep(0.002)  # Ensure different timestamps
        assert ids == sorted(ids)

    def test_extract_timestamp(self):
        gen = IDGenerator(node_id=1)
        before = time.time()
        uid = gen.generate()
        after = time.time()
        ts = gen.extract_timestamp(uid)
        assert before - 0.1 <= ts <= after + 0.1

    def test_node_id_embedded(self):
        # IDs from different nodes should still be unique and sortable
        gen1 = IDGenerator(node_id=1)
        gen2 = IDGenerator(node_id=2)
        ids1 = [gen1.generate() for _ in range(100)]
        ids2 = [gen2.generate() for _ in range(100)]
        assert len(set(ids1) | set(ids2)) == 200

    def test_stats_increment(self):
        gen = IDGenerator(node_id=1)
        for _ in range(5):
            gen.generate()
        assert gen.stats()["generated"] == 5

    def test_high_throughput_no_collision(self):
        gen = IDGenerator(node_id=1)
        ids = [gen.generate() for _ in range(5000)]
        assert len(set(ids)) == 5000

    def test_repr(self):
        gen = IDGenerator(node_id=42)
        assert "IDGenerator" in repr(gen)
        assert "42" in repr(gen)


class TestIDBatchGenerator:
    def test_batch(self):
        gen = IDBatchGenerator(node_id=1)
        batch = gen.generate_batch(100)
        assert len(batch) == 100
        assert len(set(batch)) == 100

    def test_batch_sortable(self):
        gen = IDBatchGenerator(node_id=1)
        batch = gen.generate_batch(50)
        assert batch == sorted(batch)

    def test_batch_stats(self):
        gen = IDBatchGenerator(node_id=1)
        gen.generate_batch(25)
        assert gen.stats()["generated"] == 25
