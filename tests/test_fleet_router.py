"""Tests for fleet_router.py — Pattern-based message router.

Run: python3 -m pytest tests/test_fleet_router.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.fleet_router import FleetRouter


class TestFleetRouter:
    def test_create(self):
        router = FleetRouter()
        assert len(router.patterns()) == 0

    def test_exact_match(self):
        router = FleetRouter()
        received = []
        router.add_route("fleet.breed.alpha", lambda msg: received.append(msg))
        router.route("fleet.breed.alpha", "payload")
        assert received == ["payload"]

    def test_star_wildcard(self):
        router = FleetRouter()
        received = []
        router.add_route("fleet.breed.*", lambda msg: received.append(msg))
        router.route("fleet.breed.alpha", "p1")
        router.route("fleet.breed.beta", "p2")
        assert len(received) == 2

    def test_hash_wildcard(self):
        router = FleetRouter()
        received = []
        router.add_route("fleet.#", lambda msg: received.append(msg))
        router.route("fleet.breed", "p1")
        router.route("fleet.breed.alpha", "p2")
        assert len(received) == 2

    def test_no_match(self):
        router = FleetRouter()
        received = []
        router.add_route("other.*", lambda msg: received.append(msg))
        count = router.route("fleet.breed", "payload")
        assert count == 0
        assert received == []

    def test_multiple_handlers(self):
        router = FleetRouter()
        results = []
        router.add_route("test", lambda msg: results.append(1))
        router.add_route("test", lambda msg: results.append(2))
        router.route("test", None)
        assert sorted(results) == [1, 2]

    def test_remove_route(self):
        router = FleetRouter()
        handler = lambda msg: None
        router.add_route("x", handler)
        assert router.remove_route("x", handler) is True
        assert router.handler_count("x") == 0

    def test_route_one(self):
        router = FleetRouter()
        received = []
        router.add_route("test", lambda msg: received.append("first"))
        router.add_route("test", lambda msg: received.append("second"))
        router.route_one("test", None)
        assert received == ["first"]

    def test_handler_error_not_fatal(self):
        router = FleetRouter()
        results = []
        router.add_route("test", lambda msg: (_ for _ in ()).throw(ValueError("boom")))
        router.add_route("test", lambda msg: results.append("ok"))
        router.route("test", None)
        assert results == ["ok"]

    def test_repr(self):
        router = FleetRouter()
        router.add_route("a", lambda msg: None)
        assert "FleetRouter" in repr(router)
