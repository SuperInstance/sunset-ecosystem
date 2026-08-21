"""Tests for traffic_splitter.py — Traffic splitting for A/B and canary.

Run: python3 -m pytest tests/test_traffic_splitter.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.traffic_splitter import TrafficSplitter


class TestTrafficSplitter:
    def test_create(self):
        splitter = TrafficSplitter()
        assert splitter.stats()["sticky"] is True

    def test_add_variant(self):
        splitter = TrafficSplitter()
        splitter.add_variant("control", 80)
        assert splitter.total_weight() == 80

    def test_remove_variant(self):
        splitter = TrafficSplitter()
        splitter.add_variant("control", 80)
        assert splitter.remove_variant("control") is True
        assert splitter.remove_variant("missing") is False

    def test_route(self):
        splitter = TrafficSplitter()
        splitter.add_variant("control", 80)
        splitter.add_variant("treatment", 20)
        variant = splitter.route(user_id="user-1")
        assert variant in ["control", "treatment"]

    def test_sticky_routing(self):
        splitter = TrafficSplitter(sticky=True)
        splitter.add_variant("control", 50)
        splitter.add_variant("treatment", 50)
        v1 = splitter.route(user_id="user-1")
        v2 = splitter.route(user_id="user-1")
        assert v1 == v2

    def test_non_sticky(self):
        splitter = TrafficSplitter(sticky=False)
        splitter.add_variant("control", 50)
        splitter.add_variant("treatment", 50)
        # Non-sticky may return different (probabilistic)
        variants = {splitter.route(user_id="user-1") for _ in range(100)}
        assert len(variants) > 0

    def test_no_variants(self):
        splitter = TrafficSplitter()
        assert splitter.route() is None

    def test_variants(self):
        splitter = TrafficSplitter()
        splitter.add_variant("a", 50)
        splitter.add_variant("b", 50)
        assert sorted(splitter.variants()) == ["a", "b"]

    def test_weights(self):
        splitter = TrafficSplitter()
        splitter.add_variant("a", 30)
        splitter.add_variant("b", 70)
        assert splitter.weights() == {"a": 30, "b": 70}

    def test_get_assignment(self):
        splitter = TrafficSplitter(sticky=True)
        splitter.add_variant("control", 100)
        splitter.route(user_id="user-1")
        assert splitter.get_assignment("user-1") == "control"

    def test_stats(self):
        splitter = TrafficSplitter()
        splitter.add_variant("control", 80)
        assert splitter.stats()["variants"] == 1
        assert splitter.stats()["total_weight"] == 80

    def test_repr(self):
        splitter = TrafficSplitter()
        assert "TrafficSplitter" in repr(splitter)
