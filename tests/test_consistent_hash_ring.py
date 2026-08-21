"""Tests for consistent_hash_ring.py — Ketama-style consistent hash ring.

Run: python3 -m pytest tests/test_consistent_hash_ring.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.consistent_hash_ring import HashRing


class TestHashRing:
    def test_empty_ring(self):
        ring = HashRing()
        assert ring.get_node("any-key") is None
        assert ring.get_nodes("any-key") == []
        assert list(ring.iterate_nodes("any-key")) == []
        assert ring.nodes == []
        assert ring.node_count == 0
        assert ring.virtual_node_count == 0

    def test_add_node(self):
        ring = HashRing(replicas=10)
        ring.add_node("node-a", weight=1)
        assert ring.node_count == 1
        assert ring.virtual_node_count == 10
        assert ring.nodes == ["node-a"]
        assert ring.get_weight("node-a") == 1

    def test_add_weighted_node(self):
        ring = HashRing(replicas=10)
        ring.add_node("node-a", weight=3)
        assert ring.virtual_node_count == 30
        assert ring.get_weight("node-a") == 3

    def test_remove_node(self):
        ring = HashRing(replicas=10)
        ring.add_node("node-a", weight=1)
        ring.add_node("node-b", weight=1)
        ring.remove_node("node-a")
        assert ring.node_count == 1
        assert ring.nodes == ["node-b"]
        assert ring.get_node("key") == "node-b"

    def test_remove_unknown_node(self):
        ring = HashRing(replicas=10)
        ring.add_node("node-a", weight=1)
        ring.remove_node("node-x")  # Should not raise
        assert ring.node_count == 1

    def test_re_add_node(self):
        ring = HashRing(replicas=10)
        ring.add_node("node-a", weight=1)
        ring.add_node("node-a", weight=2)  # Idempotent: re-add with new weight
        assert ring.get_weight("node-a") == 2
        assert ring.virtual_node_count == 20

    def test_get_node_returns_node(self):
        ring = HashRing(replicas=50)
        ring.add_node("node-a", weight=1)
        ring.add_node("node-b", weight=1)
        node = ring.get_node("task-42")
        assert node in ("node-a", "node-b")

    def test_consistent_lookup(self):
        ring = HashRing(replicas=50)
        ring.add_node("node-a", weight=1)
        ring.add_node("node-b", weight=1)
        ring.add_node("node-c", weight=1)
        node1 = ring.get_node("stable-key")
        node2 = ring.get_node("stable-key")
        node3 = ring.get_node("stable-key")
        assert node1 == node2 == node3

    def test_get_nodes_distinct(self):
        ring = HashRing(replicas=50)
        ring.add_node("node-a", weight=1)
        ring.add_node("node-b", weight=1)
        ring.add_node("node-c", weight=1)
        nodes = ring.get_nodes("task-1", n=3)
        assert len(nodes) == 3
        assert len(set(nodes)) == 3
        assert set(nodes) == {"node-a", "node-b", "node-c"}

    def test_get_nodes_n_larger_than_ring(self):
        ring = HashRing(replicas=10)
        ring.add_node("node-a", weight=1)
        nodes = ring.get_nodes("task-1", n=5)
        assert nodes == ["node-a"]

    def test_get_nodes_zero(self):
        ring = HashRing(replicas=10)
        ring.add_node("node-a", weight=1)
        assert ring.get_nodes("task-1", n=0) == []

    def test_iterate_nodes(self):
        ring = HashRing(replicas=30)
        ring.add_node("node-a", weight=1)
        ring.add_node("node-b", weight=1)
        ring.add_node("node-c", weight=1)
        nodes = list(ring.iterate_nodes("task-1"))
        assert len(nodes) == 3
        assert set(nodes) == {"node-a", "node-b", "node-c"}

    def test_distribution_proportional_to_weight(self):
        ring = HashRing(replicas=120)
        ring.add_node("heavy", weight=2)
        ring.add_node("light1", weight=1)
        ring.add_node("light2", weight=1)

        dist = ring.get_key_distribution()
        total = sum(dist.values())

        # heavy should have ~50% of virtual arcs
        assert dist["heavy"] / total > 0.45
        assert dist["heavy"] / total < 0.55

    def test_key_distribution_balanced(self):
        ring = HashRing(replicas=120)
        ring.add_node("n1", weight=1)
        ring.add_node("n2", weight=1)
        ring.add_node("n3", weight=1)
        ring.add_node("n4", weight=1)

        counts = {}
        for i in range(5000):
            node = ring.get_node(f"key-{i}")
            counts[node] = counts.get(node, 0) + 1

        total = sum(counts.values())
        for node, c in counts.items():
            share = c / total
            # Each should be within 30-70% for 5k keys with 120 replicas
            assert 0.15 < share < 0.35, f"{node} has unfair share {share:.2f}"

    def test_minimal_churn_on_remove(self):
        ring = HashRing(replicas=120)
        ring.add_node("n1", weight=1)
        ring.add_node("n2", weight=1)
        ring.add_node("n3", weight=1)
        ring.add_node("n4", weight=1)

        before = {}
        for i in range(2000):
            before[i] = ring.get_node(f"key-{i}")

        ring.remove_node("n2")

        moved = 0
        for i in range(2000):
            after = ring.get_node(f"key-{i}")
            if after != before[i]:
                moved += 1

        churn_rate = moved / 2000
        # With 4 nodes, removing 1 should move ~25% of keys
        assert churn_rate < 0.40, f"Churn too high: {churn_rate:.2%}"
        assert churn_rate > 0.10, f"Churn suspiciously low: {churn_rate:.2%}"

    def test_all_keys_map_after_remove(self):
        ring = HashRing(replicas=50)
        ring.add_node("n1", weight=1)
        ring.add_node("n2", weight=1)
        ring.remove_node("n1")

        for i in range(100):
            node = ring.get_node(f"key-{i}")
            assert node == "n2"

    def test_repr(self):
        ring = HashRing(replicas=10)
        ring.add_node("n1", weight=1)
        assert "HashRing" in repr(ring)
        assert "nodes=1" in repr(ring)

    def test_virtual_node_count_matches_replicas(self):
        ring = HashRing(replicas=80)
        ring.add_node("a", weight=1)
        ring.add_node("b", weight=2)
        assert ring.virtual_node_count == 240  # 80 + 160

    def test_remove_all_nodes(self):
        ring = HashRing(replicas=10)
        ring.add_node("n1", weight=1)
        ring.add_node("n2", weight=1)
        ring.remove_node("n1")
        ring.remove_node("n2")
        assert ring.get_node("key") is None
        assert ring.node_count == 0
        assert ring.virtual_node_count == 0
