"""Tests for distributed_counter.py — CRDT-based distributed counter.

Run: python3 -m pytest tests/test_distributed_counter.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.distributed_counter import DistributedCounter


class TestDistributedCounter:
    def test_create(self):
        counter = DistributedCounter("node-1")
        assert counter.value() == 0

    def test_increment(self):
        counter = DistributedCounter("node-1")
        counter.increment()
        assert counter.value() == 1
        counter.increment(5)
        assert counter.value() == 6

    def test_merge(self):
        c1 = DistributedCounter("node-1")
        c1.increment(5)
        c2 = DistributedCounter("node-2")
        c2.increment(3)
        c1.merge(c2)
        assert c1.value() == 8

    def test_merge_idempotent(self):
        c1 = DistributedCounter("node-1")
        c1.increment(5)
        c2 = DistributedCounter("node-2")
        c2.increment(3)
        c1.merge(c2)
        c1.merge(c2)
        assert c1.value() == 8

    def test_get_node_value(self):
        c1 = DistributedCounter("node-1")
        c1.increment(5)
        assert c1.get_node_value("node-1") == 5
        assert c1.get_node_value("node-2") == 0

    def test_serialize(self):
        c1 = DistributedCounter("node-1")
        c1.increment(5)
        data = c1.to_dict()
        c2 = DistributedCounter.from_dict(data)
        assert c2.value() == 5

    def test_stats(self):
        c1 = DistributedCounter("node-1")
        c1.increment(5)
        stats = c1.stats()
        assert stats["total"] == 5
        assert stats["nodes"] == 1

    def test_repr(self):
        c1 = DistributedCounter("node-1")
        assert "DistributedCounter" in repr(c1)
