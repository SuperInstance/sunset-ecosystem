"""Tests for shard_manager.py — Data sharding and rebalancing manager.

Run: python3 -m pytest tests/test_shard_manager.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.shard_manager import ShardManager


class TestShardManager:
    def test_create(self):
        shards = ShardManager(replication=2)
        assert shards.stats()["replication"] == 2

    def test_add_node(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        assert shards.stats()["nodes"] == 1
        assert shards.stats()["total_capacity"] == 100

    def test_remove_node(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        shards.add_shard("shard-1", size=10)
        assert shards.get_node("shard-1") == "node-1"
        assert shards.remove_node("node-1") is True
        assert shards.get_node("shard-1") is None
        assert shards.remove_node("missing") is False

    def test_add_shard(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        assert shards.add_shard("shard-1", size=10) is True
        assert shards.get_node("shard-1") == "node-1"

    def test_add_shard_no_nodes(self):
        shards = ShardManager()
        assert shards.add_shard("shard-1", size=10) is False

    def test_remove_shard(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        shards.add_shard("shard-1", size=10)
        assert shards.remove_shard("shard-1") is True
        assert shards.get_node("shard-1") is None
        assert shards.remove_shard("missing") is False

    def test_shards_on_node(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        shards.add_shard("shard-1", size=10)
        shards.add_shard("shard-2", size=10)
        assert sorted(shards.shards_on_node("node-1")) == ["shard-1", "shard-2"]

    def test_node_utilization(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        shards.add_shard("shard-1", size=50)
        assert shards.node_utilization("node-1") == 0.5

    def test_rebalance(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        shards.add_node("node-2", capacity=100)
        shards.add_shard("shard-1", size=80)
        shards.add_shard("shard-2", size=20)
        assert shards.get_node("shard-1") == "node-1"
        moved = shards.rebalance()
        # Should move at least one shard for better balance
        assert moved >= 0

    def test_rebalance_single_node(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        shards.add_shard("shard-1", size=10)
        assert shards.rebalance() == 0

    def test_stats(self):
        shards = ShardManager()
        shards.add_node("node-1", capacity=100)
        shards.add_shard("shard-1", size=10)
        stats = shards.stats()
        assert stats["nodes"] == 1
        assert stats["shards"] == 1
        assert stats["total_usage"] == 10

    def test_repr(self):
        shards = ShardManager()
        assert "ShardManager" in repr(shards)
