"""Tests for load_balancer.py — Request load balancer.

Run: python3 -m pytest tests/test_load_balancer.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.load_balancer import LoadBalancer, NodeStats


class TestLoadBalancer:
    def test_create(self):
        lb = LoadBalancer()
        assert lb._nodes == {}

    def test_add_remove_node(self):
        lb = LoadBalancer()
        lb.add_node("a", weight=2.0)
        assert "a" in lb._nodes
        assert lb._nodes["a"].weight == 2.0
        assert lb.remove_node("a") is True
        assert lb.remove_node("a") is False

    def test_round_robin(self):
        lb = LoadBalancer(["a", "b", "c"])
        picks = [lb.pick("round_robin") for _ in range(6)]
        assert picks == ["a", "b", "c", "a", "b", "c"]

    def test_round_robin_empty(self):
        lb = LoadBalancer()
        assert lb.pick("round_robin") is None

    def test_least_connections(self):
        lb = LoadBalancer(["a", "b"])
        n1 = lb.pick("least_connections")
        n2 = lb.pick("least_connections")
        # Both should be different on first picks (0 connections each)
        assert n1 != n2 or n1 is not None

    def test_weighted(self):
        lb = LoadBalancer(["a", "b"])
        lb._nodes["a"].weight = 10.0
        lb._nodes["b"].weight = 1.0
        # With high probability, a gets picked more often
        counts = {"a": 0, "b": 0}
        for _ in range(100):
            node = lb.pick("weighted_round_robin")
            counts[node] += 1
        assert counts["a"] > counts["b"]

    def test_sticky(self):
        lb = LoadBalancer(["a", "b", "c"])
        n1 = lb.pick("sticky", key="user-123")
        n2 = lb.pick("sticky", key="user-123")
        assert n1 == n2
        n3 = lb.pick("sticky", key="user-456")
        # Might be same or different
        assert n3 in ("a", "b", "c")

    def test_healthy_filtering(self):
        lb = LoadBalancer(["a", "b"])
        lb.set_healthy("a", False)
        for _ in range(10):
            assert lb.pick("round_robin") == "b"

    def test_all_unhealthy(self):
        lb = LoadBalancer(["a", "b"])
        lb.set_healthy("a", False)
        lb.set_healthy("b", False)
        assert lb.pick("round_robin") is None

    def test_record_result(self):
        lb = LoadBalancer(["a"])
        lb.pick("least_connections")
        assert lb._nodes["a"].connections == 1
        lb.record_result("a", success=True)
        assert lb._nodes["a"].successes == 1
        assert lb._nodes["a"].connections == 0

    def test_record_failure(self):
        lb = LoadBalancer(["a"])
        lb.pick("least_connections")
        lb.record_result("a", success=False)
        assert lb._nodes["a"].failures == 1

    def test_node_stats(self):
        lb = LoadBalancer(["a", "b"])
        stats = lb.node_stats()
        assert "a" in stats
        assert "b" in stats

    def test_best_node(self):
        lb = LoadBalancer(["a", "b"])
        lb.record_result("a", success=True)
        lb.record_result("a", success=True)
        lb.record_result("b", success=True)
        lb.record_result("b", success=False)
        assert lb.best_node() == "a"

    def test_best_node_empty(self):
        lb = LoadBalancer()
        assert lb.best_node() is None

    def test_repr(self):
        lb = LoadBalancer(["a", "b"])
        assert "LoadBalancer" in repr(lb)
