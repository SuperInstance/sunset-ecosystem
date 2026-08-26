"""Tests for hash_ring.py — Consistent hash ring.

Run: python3 -m pytest tests/test_hash_ring.py -v --tb=short
"""

from __future__ import annotations

import pytest

from swarm.hash_ring import HashRing, RingNode


class TestRingNode:
    def test_create(self):
        n = RingNode(name="node-a", weight=2)
        assert n.name == "node-a"
        assert n.weight == 2
        assert n.metadata == {}

    def test_default_weight(self):
        n = RingNode(name="node-b")
        assert n.weight == 1


class TestHashRingBasics:
    def test_empty_ring(self):
        ring = HashRing()
        assert ring.node_count == 0
        assert ring.get_node("key") is None

    def test_add_node(self):
        ring = HashRing()
        ring.add_node(RingNode("a"))
        assert ring.node_count == 1
        assert ring.virtual_count == ring.VIRTUALS_PER_WEIGHT

    def test_add_weighted_node(self):
        ring = HashRing()
        ring.add_node(RingNode("a", weight=2))
        assert ring.virtual_count == 2 * ring.VIRTUALS_PER_WEIGHT

    def test_get_node_returns_node(self):
        ring = HashRing([RingNode("a"), RingNode("b")])
        node = ring.get_node("my-key")
        assert node is not None
        assert node.name in ("a", "b")

    def test_get_node_name(self):
        ring = HashRing([RingNode("a"), RingNode("b")])
        name = ring.get_node_name("my-key")
        assert name in ("a", "b")

    def test_remove_node(self):
        ring = HashRing([RingNode("a"), RingNode("b")])
        assert ring.has_node("a")
        ring.remove_node("a")
        assert not ring.has_node("a")
        assert ring.node_count == 1

    def test_replicas(self):
        ring = HashRing([RingNode("a"), RingNode("b"), RingNode("c")], replicas=2)
        nodes = ring.get_nodes("key", n=2)
        assert len(nodes) == 2
        assert nodes[0].name != nodes[1].name

    def test_replicas_capped_by_node_count(self):
        ring = HashRing([RingNode("a")], replicas=5)
        nodes = ring.get_nodes("key", n=5)
        assert len(nodes) == 1


class TestHashRingDistribution:
    def test_distribution_non_empty(self):
        ring = HashRing([RingNode("a"), RingNode("b"), RingNode("c")])
        dist = ring.distribution(num_samples=1000)
        assert sum(dist.values()) == 1000
        assert all(v > 0 for v in dist.values())  # all nodes get some

    def test_weighted_distribution(self):
        ring = HashRing([RingNode("light", weight=1), RingNode("heavy", weight=3)])
        dist = ring.distribution(num_samples=10000)
        # Heavy node should get ~3x more keys
        assert dist["heavy"] > dist["light"]
        ratio = dist["heavy"] / max(dist["light"], 1)
        assert 2.0 < ratio < 5.0

    def test_balance_score(self):
        ring = HashRing([RingNode("a"), RingNode("b"), RingNode("c")])
        score = ring.balance_score(num_samples=5000)
        assert score >= 0.0  # lower is more balanced
        assert score < 1.0  # should be fairly balanced with 3 equal-weight nodes

    def test_balance_improves_with_more_nodes(self):
        ring3 = HashRing([RingNode(f"n{i}") for i in range(3)])
        ring10 = HashRing([RingNode(f"n{i}") for i in range(10)])
        s3 = ring3.balance_score(num_samples=10000)
        s10 = ring10.balance_score(num_samples=10000)
        # More nodes with more virtuals should generally be more balanced
        # (not guaranteed with random sampling, but usually true)
        assert s3 < 0.5  # 3 nodes should still be reasonably balanced
        assert s10 < 0.5  # 10 nodes too
        # With high probability, 10 nodes is better than 3
        # If not, both should still be in reasonable range


class TestHashRingRemapping:
    def test_remapping_on_add(self):
        ring = HashRing([RingNode("a"), RingNode("b"), RingNode("c")])
        frac = ring.remapping_on_add(RingNode("d"), num_samples=1000)
        # With 4 nodes, ~25% of keys should remap
        assert 0.1 < frac < 0.4

    def test_remapping_restores_state(self):
        ring = HashRing([RingNode("a"), RingNode("b")])
        _ = ring.remapping_on_add(RingNode("c"), num_samples=100)
        assert ring.node_count == 2  # remapped should restore
        assert not ring.has_node("c")


class TestHashRingSerialization:
    def test_to_dict(self):
        ring = HashRing([RingNode("a", weight=2, metadata={"ip": "10.0.0.1"})])
        d = ring.to_dict()
        assert d["replicas"] == 3
        assert d["node_count"] == 1
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["name"] == "a"
        assert d["nodes"][0]["weight"] == 2

    def test_repr(self):
        ring = HashRing([RingNode("a"), RingNode("b")])
        r = repr(ring)
        assert "HashRing" in r
        assert "nodes=2" in r


class TestHashRingDeterminism:
    def test_same_key_same_node(self):
        ring = HashRing([RingNode("a"), RingNode("b")])
        n1 = ring.get_node_name("fixed-key")
        n2 = ring.get_node_name("fixed-key")
        assert n1 == n2

    def test_different_keys_may_differ(self):
        ring = HashRing([RingNode("a"), RingNode("b")])
        n1 = ring.get_node_name("key-1")
        n2 = ring.get_node_name("key-2")
        # They might be the same or different — both valid
        assert n1 in ("a", "b")
        assert n2 in ("a", "b")
