"""Tests for service_discovery.py — Fleet node discovery.

Run: python3 -m pytest tests/test_service_discovery.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from swarm.service_discovery import ServiceDiscovery, NodeInfo


class TestServiceDiscovery:
    def test_create(self):
        sd = ServiceDiscovery(ttl=30.0)
        assert sd.node_count() == 0

    def test_register(self):
        sd = ServiceDiscovery()
        sd.register("node-1", {"ip": "10.0.0.1", "capabilities": ["gpu"]})
        assert sd.node_count() == 1

    def test_get(self):
        sd = ServiceDiscovery()
        sd.register("node-1", {"ip": "10.0.0.1"})
        node = sd.get("node-1")
        assert node is not None
        assert node.node_id == "node-1"
        assert node.metadata["ip"] == "10.0.0.1"

    def test_heartbeat(self):
        sd = ServiceDiscovery()
        sd.register("node-1", {"ip": "10.0.0.1"}, ttl=1.0)
        assert sd.heartbeat("node-1") is True
        assert sd.get("node-1").healthy is True

    def test_heartbeat_unknown(self):
        sd = ServiceDiscovery()
        assert sd.heartbeat("unknown") is False

    def test_unregister(self):
        sd = ServiceDiscovery()
        sd.register("node-1", {})
        assert sd.unregister("node-1") is True
        assert sd.node_count() == 0
        assert sd.unregister("node-1") is False

    def test_find_by_capability(self):
        sd = ServiceDiscovery()
        sd.register("a", {"capabilities": ["gpu", "fast"]})
        sd.register("b", {"capabilities": ["cpu"]})
        sd.register("c", {"capabilities": ["gpu"]})
        nodes = sd.find_by_capability("gpu")
        assert len(nodes) == 2
        ids = {n.node_id for n in nodes}
        assert ids == {"a", "c"}

    def test_find_by_tag(self):
        sd = ServiceDiscovery()
        sd.register("a", {"region": "us-east", "tier": "prod"})
        sd.register("b", {"region": "eu-west", "tier": "prod"})
        nodes = sd.find_by_tag("region", "us-east")
        assert len(nodes) == 1
        assert nodes[0].node_id == "a"

    def test_expiration(self):
        sd = ServiceDiscovery(ttl=0.1)
        sd.register("node-1", {})
        assert sd.node_count() == 1
        time.sleep(0.15)
        sd.force_cleanup()
        assert sd.node_count() == 0

    def test_heartbeat_prevents_expiration(self):
        sd = ServiceDiscovery(ttl=0.1)
        sd.register("node-1", {})
        time.sleep(0.05)
        sd.heartbeat("node-1")
        time.sleep(0.08)
        sd.force_cleanup()
        assert sd.node_count() == 1

    def test_healthy_nodes(self):
        sd = ServiceDiscovery()
        sd.register("a", {})
        sd.register("b", {})
        assert len(sd.healthy_nodes()) == 2

    def test_all_nodes(self):
        sd = ServiceDiscovery()
        sd.register("a", {"x": 1})
        sd.register("b", {"x": 2})
        nodes = sd.all_nodes()
        assert len(nodes) == 2

    def test_report(self):
        sd = ServiceDiscovery()
        sd.register("a", {})
        sd.register("b", {})
        r = sd.report()
        assert r["total"] == 2
        assert r["healthy"] == 2
        assert r["stale"] == 0

    def test_repr(self):
        sd = ServiceDiscovery()
        assert "ServiceDiscovery" in repr(sd)
