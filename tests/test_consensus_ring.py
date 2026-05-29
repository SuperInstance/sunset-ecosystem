"""Tests for consensus_ring.py — Hash ring with distributed consensus.

Run: python3 -m pytest tests/test_consensus_ring.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.consensus_ring import ConsensusHashRing, ConsensusError


class TestConsensusHashRing:
    def test_create(self):
        ring = ConsensusHashRing("node-1")
        assert ring.node_id == "node-1"

    def test_direct_add_remove(self):
        ring = ConsensusHashRing("node-1")
        ring.add_node_direct("node-2", weight=1)
        assert ring.ring_size() == 1
        ring.remove_node_direct("node-2")
        assert ring.ring_size() == 0

    def test_get_node(self):
        ring = ConsensusHashRing("node-1")
        ring.add_node_direct("node-2", weight=1)
        node = ring.get_node("some-key")
        assert node == "node-2"

    def test_propose_and_vote(self):
        ring = ConsensusHashRing("node-1")
        key = ring.propose_add("node-2")
        ring.vote(key, True)
        assert ring.check_quorum(key) is True

    def test_commit_add(self):
        ring = ConsensusHashRing("node-1")
        key = ring.propose_add("node-2")
        ring.vote(key, True)
        ring.commit(key)
        assert ring.ring_size() == 1
        assert ring.get_node("key") == "node-2"

    def test_commit_remove(self):
        ring = ConsensusHashRing("node-1")
        ring.add_node_direct("node-2")
        key = ring.propose_remove("node-2")
        ring.vote(key, True)
        ring.commit(key)
        assert ring.ring_size() == 0

    def test_commit_without_quorum(self):
        ring = ConsensusHashRing("node-1")
        key = ring.propose_add("node-2")
        with pytest.raises(ConsensusError):
            ring.commit(key)

    def test_unknown_proposal(self):
        ring = ConsensusHashRing("node-1")
        with pytest.raises(ConsensusError):
            ring.vote("missing", True)

    def test_majority_quorum(self):
        ring = ConsensusHashRing("node-1")
        key = ring.propose_add("node-2")
        ring.vote(key, True, voter="a")
        ring.vote(key, True, voter="b")
        ring.vote(key, False, voter="c")
        assert ring.check_quorum(key) is True

    def test_stats(self):
        ring = ConsensusHashRing("node-1")
        ring.propose_add("a")
        ring.vote("add:a", True)
        ring.commit("add:a")
        stats = ring.stats()
        assert stats["proposals"] == 1
        assert stats["commits"] == 1

    def test_repr(self):
        ring = ConsensusHashRing("node-1")
        assert "node-1" in repr(ring)
