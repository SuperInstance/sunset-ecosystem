"""Tests for gossip_protocol.py — Epidemic gossip for state sync.

Run: python3 -m pytest tests/test_gossip_protocol.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.gossip_protocol import GossipProtocol, GossipMessage


class TestGossipProtocol:
    def test_create(self):
        g = GossipProtocol("node-1")
        assert g.node_id == "node-1"

    def test_add_remove_peer(self):
        g = GossipProtocol("node-1")
        g.add_peer("node-2", "http://node-2:8080")
        assert "node-2" in g.peers()
        assert g.remove_peer("node-2") is True
        assert g.remove_peer("missing") is False

    def test_set_get_state(self):
        g = GossipProtocol("node-1")
        g.set_state({"x": 1})
        assert g.get_state()["x"] == 1

    def test_round_no_peers(self):
        g = GossipProtocol("node-1")
        msgs = g.round()
        assert len(msgs) == 0

    def test_round_with_peers(self):
        g = GossipProtocol("node-1", fanout=2)
        g.add_peer("a", "addr-a")
        g.add_peer("b", "addr-b")
        g.add_peer("c", "addr-c")
        msgs = g.round()
        assert len(msgs) == 2

    def test_receive_updates_state(self):
        g = GossipProtocol("node-1")
        g.set_state({"x": 1})
        msg = GossipMessage(
            sender="node-2",
            digest="different",
            payload={"x": 2, "y": 3},
        )
        updated = g.receive(msg)
        assert updated is True
        assert g.get_state()["x"] == 2
        assert g.get_state()["y"] == 3

    def test_receive_ignores_same_digest(self):
        g = GossipProtocol("node-1")
        g.set_state({"x": 1})
        msg = GossipMessage(
            sender="node-2",
            digest=g._default_digest({"x": 1}),
            payload={"x": 1},
        )
        updated = g.receive(msg)
        assert updated is False

    def test_receive_ignores_self(self):
        g = GossipProtocol("node-1")
        msg = GossipMessage(sender="node-1", digest="abc")
        assert g.receive(msg) is False

    def test_stats(self):
        g = GossipProtocol("node-1")
        g.add_peer("a", "addr")
        g.round()
        stats = g.stats()
        assert stats["rounds"] == 1
        assert stats["messages_sent"] == 1

    def test_repr(self):
        g = GossipProtocol("node-1")
        assert "node-1" in repr(g)
