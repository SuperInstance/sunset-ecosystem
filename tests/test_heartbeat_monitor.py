"""Tests for heartbeat_monitor.py — Node liveness detection.

Run: python3 -m pytest tests/test_heartbeat_monitor.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.heartbeat_monitor import HeartbeatMonitor


class TestHeartbeatMonitor:
    def test_create(self):
        m = HeartbeatMonitor()
        assert len(m.nodes()) == 0

    def test_beat_and_healthy(self):
        m = HeartbeatMonitor()
        m.beat("node-1")
        assert m.status("node-1") == "healthy"

    def test_suspected(self):
        m = HeartbeatMonitor(suspicion_threshold=1.0, dead_timeout=60.0)
        m.beat("node-1")
        time.sleep(0.03)
        m.beat("node-1")
        time.sleep(0.03)
        m.beat("node-1")
        time.sleep(0.15)
        assert m.status("node-1") == "suspected"

    def test_dead_timeout(self):
        m = HeartbeatMonitor(dead_timeout=0.05)
        m.beat("node-1")
        time.sleep(0.06)
        assert m.status("node-1") == "dead"

    def test_phi_computation(self):
        m = HeartbeatMonitor()
        m.beat("node-1")
        time.sleep(0.01)
        m.beat("node-1")
        time.sleep(0.01)
        m.beat("node-1")
        phi = m.phi("node-1")
        assert phi >= 0.0

    def test_remove(self):
        m = HeartbeatMonitor()
        m.beat("node-1")
        assert m.remove("node-1") is True
        assert m.remove("missing") is False

    def test_node_lists(self):
        m = HeartbeatMonitor(dead_timeout=60.0)
        m.beat("a")
        m.beat("b")
        assert sorted(m.nodes()) == ["a", "b"]
        assert "a" in m.healthy_nodes()

    def test_unknown_node_dead(self):
        m = HeartbeatMonitor()
        assert m.status("missing") == "dead"
        assert m.phi("missing") == float("inf")

    def test_single_beat_no_phi(self):
        m = HeartbeatMonitor()
        m.beat("node-1")
        assert m.phi("node-1") == 0.0

    def test_repr(self):
        m = HeartbeatMonitor()
        m.beat("a")
        assert "HeartbeatMonitor" in repr(m)
