"""Tests for leader_elector.py — Distributed leader election.

Run: python3 -m pytest tests/test_leader_elector.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.leader_elector import LeaderElector


class TestLeaderElector:
    def test_create(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        assert elector.stats()["node_id"] == "node-1"
        assert elector.list_roles() == []

    def test_elect_first(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        assert elector.elect("coordinator") is True
        assert elector.is_leader("coordinator") is True

    def test_elect_contested(self):
        store = {}
        e1 = LeaderElector("node-1", ttl_seconds=5.0)
        e2 = LeaderElector("node-2", ttl_seconds=5.0)
        e1._store = store
        e2._store = store
        assert e1.elect("coordinator") is True
        assert e2.elect("coordinator") is False
        assert e1.is_leader("coordinator") is True
        assert e2.is_leader("coordinator") is False

    def test_heartbeat(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        elector.elect("coordinator")
        assert elector.heartbeat("coordinator") is True
        assert elector.is_leader("coordinator") is True

    def test_heartbeat_not_leader(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        assert elector.heartbeat("coordinator") is False

    def test_expiration(self):
        fake_time = [0.0]

        def clock():
            return fake_time[0]

        elector = LeaderElector("node-1", ttl_seconds=5.0, clock=clock)
        elector.elect("coordinator")
        assert elector.is_leader("coordinator") is True

        fake_time[0] = 6.0
        assert elector.is_leader("coordinator") is False
        assert elector.get_leader("coordinator") is None

    def test_heartbeat_prevents_expiration(self):
        fake_time = [0.0]

        def clock():
            return fake_time[0]

        elector = LeaderElector("node-1", ttl_seconds=5.0, clock=clock)
        elector.elect("coordinator")
        fake_time[0] = 4.0
        elector.heartbeat("coordinator")
        fake_time[0] = 8.0
        assert elector.is_leader("coordinator") is True

    def test_stolen_after_expiration(self):
        fake_time = [0.0]

        def clock():
            return fake_time[0]

        store = {}
        e1 = LeaderElector("node-1", ttl_seconds=5.0, clock=clock)
        e2 = LeaderElector("node-2", ttl_seconds=5.0, clock=clock)
        e1._store = store
        e2._store = store
        e1.elect("coordinator")
        fake_time[0] = 6.0
        assert e2.elect("coordinator") is True
        assert e2.is_leader("coordinator") is True

    def test_resign(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        elector.elect("coordinator")
        assert elector.resign("coordinator") is True
        assert elector.is_leader("coordinator") is False

    def test_resign_not_leader(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        assert elector.resign("coordinator") is False

    def test_get_leader(self):
        store = {}
        e1 = LeaderElector("node-1", ttl_seconds=5.0)
        e2 = LeaderElector("node-2", ttl_seconds=5.0)
        e1._store = store
        e2._store = store
        e1.elect("coordinator")
        assert e1.get_leader("coordinator") == "node-1"
        assert e2.get_leader("coordinator") == "node-1"

    def test_metadata(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        elector.elect("coordinator", metadata={"host": "10.0.0.1", "port": "8080"})
        assert elector.get_leader_metadata("coordinator") == {
            "host": "10.0.0.1",
            "port": "8080",
        }

    def test_metadata_expired(self):
        fake_time = [0.0]

        def clock():
            return fake_time[0]

        elector = LeaderElector("node-1", ttl_seconds=5.0, clock=clock)
        elector.elect("coordinator", metadata={"a": "b"})
        fake_time[0] = 6.0
        assert elector.get_leader_metadata("coordinator") is None

    def test_cleanup_expired(self):
        fake_time = [0.0]

        def clock():
            return fake_time[0]

        elector = LeaderElector("node-1", ttl_seconds=5.0, clock=clock)
        elector.elect("coordinator")
        fake_time[0] = 6.0
        assert elector.cleanup_expired() == 1
        assert elector.list_roles() == []

    def test_list_roles(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        elector.elect("coordinator")
        elector.elect("replicator")
        assert sorted(elector.list_roles()) == ["coordinator", "replicator"]

    def test_on_change_callback(self):
        events = []

        def cb(role, event):
            events.append((role, event))

        elector = LeaderElector("node-1", ttl_seconds=5.0)
        elector.on_change("coordinator", cb)
        elector.elect("coordinator")
        elector.resign("coordinator")
        assert events == [("coordinator", "resigned")]

    def test_repr(self):
        elector = LeaderElector("node-1", ttl_seconds=5.0)
        assert "LeaderElector" in repr(elector)
        assert "node-1" in repr(elector)
