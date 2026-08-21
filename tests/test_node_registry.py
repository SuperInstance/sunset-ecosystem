"""Tests for node_registry.py — Fleet node registry.

Run: python3 -m pytest tests/test_node_registry.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.node_registry import NodeRegistry


class TestNodeRegistry:
    def test_create(self):
        reg = NodeRegistry()
        assert reg.stats()["total"] == 0

    def test_register_and_get(self):
        reg = NodeRegistry()
        reg.register("node-1", {"host": "10.0.0.1"}, capabilities=["breed"])
        info = reg.get("node-1")
        assert info is not None
        assert info["metadata"]["host"] == "10.0.0.1"

    def test_unregister(self):
        reg = NodeRegistry()
        reg.register("node-1", {})
        assert reg.unregister("node-1") is True
        assert reg.unregister("missing") is False

    def test_heartbeat(self):
        reg = NodeRegistry()
        reg.register("node-1", {})
        assert reg.heartbeat("node-1") is True
        assert reg.heartbeat("missing") is False
        assert reg.get("node-1")["healthy"] is True

    def test_mark_unhealthy(self):
        reg = NodeRegistry()
        reg.register("node-1", {})
        assert reg.mark_unhealthy("node-1") is True
        assert reg.get("node-1")["healthy"] is False

    def test_find_by_capability(self):
        reg = NodeRegistry()
        reg.register("a", {}, capabilities=["breed"])
        reg.register("b", {}, capabilities=["train"])
        assert reg.find_by_capability("breed") == ["a"]

    def test_find_by_tag(self):
        reg = NodeRegistry()
        reg.register("a", {}, tags={"zone": "us-east"})
        reg.register("b", {}, tags={"zone": "us-west"})
        assert reg.find_by_tag("zone", "us-east") == ["a"]

    def test_healthy_unhealthy(self):
        reg = NodeRegistry()
        reg.register("a", {})
        reg.register("b", {})
        reg.mark_unhealthy("b")
        assert reg.healthy_nodes() == ["a"]
        assert reg.unhealthy_nodes() == ["b"]

    def test_stale_nodes(self):
        reg = NodeRegistry()
        reg.register("a", {})
        # Artificially set last heartbeat to old time
        reg.get("a")["last_heartbeat"] = time.time() - 1000
        assert "a" in reg.stale_nodes(threshold_sec=500)

    def test_list_nodes(self):
        reg = NodeRegistry()
        reg.register("a", {})
        reg.register("b", {})
        assert sorted(reg.list_nodes()) == ["a", "b"]

    def test_stats(self):
        reg = NodeRegistry()
        reg.register("a", {})
        reg.register("b", {})
        reg.mark_unhealthy("b")
        stats = reg.stats()
        assert stats["total"] == 2
        assert stats["healthy"] == 1
        assert stats["unhealthy"] == 1

    def test_repr(self):
        reg = NodeRegistry()
        reg.register("a", {})
        assert "NodeRegistry" in repr(reg)
