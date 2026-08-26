"""Tests for id_generator.py — Snowflake-style ID generator.

Run: python3 -m pytest tests/test_id_generator.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.id_generator import IDGenerator


class TestIDGenerator:
    def test_create(self):
        gen = IDGenerator(node_id=1)
        assert gen.node_id == 1

    def test_next_unique(self):
        gen = IDGenerator(node_id=1)
        id1 = gen.next()
        id2 = gen.next()
        assert id1 != id2
        assert id2 > id1

    def test_parse(self):
        gen = IDGenerator(node_id=5)
        id_val = gen.next()
        parsed = gen.parse(id_val)
        assert parsed["node_id"] == 5
        assert "timestamp" in parsed
        assert "sequence" in parsed

    def test_multiple_nodes_unique(self):
        gen1 = IDGenerator(node_id=1)
        gen2 = IDGenerator(node_id=2)
        id1 = gen1.next()
        id2 = gen2.next()
        assert id1 != id2

    def test_invalid_node_id(self):
        with pytest.raises(ValueError):
            IDGenerator(node_id=1024)
        with pytest.raises(ValueError):
            IDGenerator(node_id=-1)

    def test_sequence_rollover(self):
        gen = IDGenerator(node_id=1)
        ids = [gen.next() for _ in range(10)]
        assert len(set(ids)) == 10

    def test_time_ordering(self):
        gen = IDGenerator(node_id=1)
        import time

        id1 = gen.next()
        time.sleep(0.01)
        id2 = gen.next()
        assert id2 > id1

    def test_repr(self):
        gen = IDGenerator(node_id=1)
        assert "IDGenerator" in repr(gen)
