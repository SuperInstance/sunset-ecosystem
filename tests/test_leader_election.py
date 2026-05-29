"""Tests for leader_election.py — Leader election with heartbeats.

Run: python3 -m pytest tests/test_leader_election.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.leader_election import LeaderElection


class TestLeaderElection:
    def test_create(self):
        election = LeaderElection("node-1", timeout_sec=5)
        assert election.node_id == "node-1"
        assert election.timeout_sec == 5

    def test_become_leader(self):
        election = LeaderElection("node-1")
        assert election.become_leader() is True
        assert election.is_leader() is True
        assert election.get_leader() == "node-1"

    def test_step_down(self):
        election = LeaderElection("node-1")
        election.become_leader()
        election.step_down()
        assert election.is_leader() is False
        assert election.get_leader() is None

    def test_heartbeat(self):
        election = LeaderElection("node-1", timeout_sec=5, clock=lambda: 0)
        election.become_leader()
        election.heartbeat()
        election._clock = lambda: 3
        assert election.leader_expired() is False

    def test_leader_expired(self):
        election = LeaderElection("node-1", timeout_sec=5, clock=lambda: 0)
        election.become_leader()
        election._clock = lambda: 10
        assert election.leader_expired() is True
        assert election.get_leader() is None

    def test_can_become_leader(self):
        election = LeaderElection("node-1", timeout_sec=5, clock=lambda: 0)
        election.become_leader()
        assert election.can_become_leader() is False
        election._clock = lambda: 10
        assert election.can_become_leader() is True

    def test_external_heartbeat(self):
        election = LeaderElection("node-2", timeout_sec=5, clock=lambda: 0)
        election.heartbeat("node-1")
        assert election.get_leader() == "node-1"
        assert election.is_leader() is False
        election._clock = lambda: 10
        assert election.get_leader() is None

    def test_stats(self):
        election = LeaderElection("node-1")
        election.become_leader()
        stats = election.stats()
        assert stats["is_leader"] is True
        assert stats["leader"] == "node-1"

    def test_repr(self):
        election = LeaderElection("node-1")
        assert "LeaderElection" in repr(election)
