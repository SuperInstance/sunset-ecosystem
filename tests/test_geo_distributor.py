"""Tests for geo_distributor.py — Geographic distribution of fleet nodes.

Run: python3 -m pytest tests/test_geo_distributor.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.geo_distributor import GeoDistributor


class TestGeoDistributor:
    def test_create(self):
        geo = GeoDistributor()
        assert geo.stats()["regions"] == 0

    def test_add_region(self):
        geo = GeoDistributor()
        geo.add_region("us-east", ["node-1", "node-2"])
        assert geo.nodes_in_region("us-east") == ["node-1", "node-2"]

    def test_remove_region(self):
        geo = GeoDistributor()
        geo.add_region("us-east", ["node-1"])
        assert geo.remove_region("us-east") is True
        assert geo.remove_region("missing") is False

    def test_add_node(self):
        geo = GeoDistributor()
        geo.add_node("us-east", "node-1")
        assert geo.nodes_in_region("us-east") == ["node-1"]

    def test_remove_node(self):
        geo = GeoDistributor()
        geo.add_region("us-east", ["node-1"])
        assert geo.remove_node("node-1") is True
        assert geo.remove_node("missing") is False

    def test_select(self):
        geo = GeoDistributor(strategy="random")
        geo.add_region("us-east", ["node-1", "node-2"])
        node = geo.select("us-east")
        assert node in ["node-1", "node-2"]

    def test_select_empty(self):
        geo = GeoDistributor()
        assert geo.select("missing") is None

    def test_select_round_robin(self):
        geo = GeoDistributor(strategy="round_robin")
        geo.add_region("us-east", ["node-1", "node-2"])
        assert geo.select("us-east") == "node-1"
        assert geo.select("us-east") == "node-2"
        assert geo.select("us-east") == "node-1"

    def test_select_any(self):
        geo = GeoDistributor()
        geo.add_region("us-east", ["node-1"])
        geo.add_region("eu-west", ["node-2"])
        assert geo.select_any() in ["node-1", "node-2"]

    def test_failover(self):
        geo = GeoDistributor()
        geo.add_region("us-east", ["node-1", "node-2"])
        alt = geo.failover("us-east", "node-1")
        assert alt == "node-2"

    def test_node_region(self):
        geo = GeoDistributor()
        geo.add_region("us-east", ["node-1"])
        assert geo.node_region("node-1") == "us-east"
        assert geo.node_region("missing") is None

    def test_node_count(self):
        geo = GeoDistributor()
        geo.add_region("us-east", ["node-1", "node-2"])
        geo.add_region("eu-west", ["node-3"])
        assert geo.node_count() == 3

    def test_repr(self):
        geo = GeoDistributor()
        assert "GeoDistributor" in repr(geo)
