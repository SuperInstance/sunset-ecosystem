"""Tests for agent_migration.py — Agent state migration between nodes.

Run: python3 -m pytest tests/test_agent_migration.py -v --tb=short
"""
from __future__ import annotations

import pytest

from swarm.agent_migration import AgentMigrator, AgentState, MigrationResult


class TestAgentMigration:
    def test_create(self):
        m = AgentMigrator()
        assert m.success_rate() == 0.0

    def test_migrate_success(self):
        m = AgentMigrator()
        state = {"weights": [0.1, 0.2], "generation": 5}
        result = m.migrate("agent-1", "node-a", "node-b", state)
        assert result.success is True
        assert result.agent_id == "agent-1"
        assert result.from_node == "node-a"
        assert result.to_node == "node-b"
        assert result.duration_ms >= 0.0
        assert result.state_hash != ""

    def test_migrate_state_hash(self):
        m = AgentMigrator()
        state = {"data": "hello"}
        result = m.migrate("a", "x", "y", state)
        # Same state should produce same hash
        result2 = m.migrate("a", "x", "y", state)
        assert result.state_hash == result2.state_hash

    def test_state_hash_changes(self):
        m = AgentMigrator()
        r1 = m.migrate("a", "x", "y", {"v": 1})
        r2 = m.migrate("a", "x", "y", {"v": 2})
        assert r1.state_hash != r2.state_hash

    def test_history(self):
        m = AgentMigrator()
        m.migrate("a", "x", "y", {})
        m.migrate("b", "x", "z", {})
        history = m.history()
        assert len(history) == 2

    def test_success_rate(self):
        m = AgentMigrator()
        # All succeed by default (placeholders return True)
        m.migrate("a", "x", "y", {})
        m.migrate("b", "x", "y", {})
        assert m.success_rate() == 1.0

    def test_avg_duration(self):
        m = AgentMigrator()
        m.migrate("a", "x", "y", {})
        assert m.avg_duration_ms() >= 0.0

    def test_report(self):
        m = AgentMigrator()
        m.migrate("a", "x", "y", {})
        r = m.report()
        assert r["total"] == 1
        assert r["successful"] == 1
        assert r["failed"] == 0
        assert r["success_rate"] == 1.0

    def test_max_history(self):
        m = AgentMigrator(max_history=2)
        m.migrate("a", "x", "y", {})
        m.migrate("b", "x", "y", {})
        m.migrate("c", "x", "y", {})
        assert len(m.history()) == 2

    def test_agent_state_hash(self):
        state = AgentState(
            agent_id="a",
            node_id="x",
            checkpoint_data={"key": "value"},
            timestamp=123.0,
        )
        h1 = state.compute_hash()
        h2 = state.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_repr(self):
        m = AgentMigrator()
        assert "AgentMigrator" in repr(m)
