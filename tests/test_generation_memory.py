"""Tests for GenerationMemory — agent generation tracking across sunset lifecycle.

Covers AgentGeneration, GenerationHistory, GenerationMemory:
register, sunset, get, get_history, get_lineage, get_children, persistence.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from logos.generation_memory import (
    AgentGeneration,
    GenerationHistory,
    GenerationMemory,
)


# ---------------------------------------------------------------------------
# AgentGeneration
# ---------------------------------------------------------------------------

class TestAgentGeneration:
    def test_repr(self):
        g = AgentGeneration(agent_id="a1", name="Alpha", generation=1, created_at=datetime.now(timezone.utc))
        assert "Alpha" in repr(g)
        assert "gen=1" in repr(g)

    def test_to_dict_roundtrip(self):
        g = AgentGeneration(
            agent_id="a1",
            name="Alpha",
            generation=1,
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            sunset_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            purpose="test",
            achievements=["did thing"],
            sunset_reason="retired",
            onboarding_docs=["docs/setup.md"],
            children_spawned=["a2"],
            parent_id="root",
            patterns_preserved=["pattern1"],
            lessons_learned=["lesson1"],
            metadata={"key": "val"},
        )
        d = g.to_dict()
        g2 = AgentGeneration.from_dict(d)
        assert g2.agent_id == "a1"
        assert g2.name == "Alpha"
        assert g2.generation == 1
        assert g2.created_at == g.created_at
        assert g2.sunset_at == g.sunset_at
        assert g2.purpose == "test"
        assert g2.achievements == ["did thing"]
        assert g2.sunset_reason == "retired"
        assert g2.onboarding_docs == ["docs/setup.md"]
        assert g2.children_spawned == ["a2"]
        assert g2.parent_id == "root"
        assert g2.patterns_preserved == ["pattern1"]
        assert g2.lessons_learned == ["lesson1"]
        assert g2.metadata == {"key": "val"}

    def test_from_dict_no_sunset(self):
        d = {
            "agent_id": "a1",
            "name": "A",
            "generation": 1,
            "created_at": "2024-01-01T00:00:00+00:00",
            "sunset_at": None,
            "purpose": "",
            "achievements": [],
            "sunset_reason": None,
            "onboarding_docs": [],
            "children_spawned": [],
            "parent_id": None,
            "patterns_preserved": [],
            "lessons_learned": [],
            "metadata": {},
        }
        g = AgentGeneration.from_dict(d)
        assert g.sunset_at is None


# ---------------------------------------------------------------------------
# GenerationHistory
# ---------------------------------------------------------------------------

class TestGenerationHistory:
    def test_repr(self):
        h = GenerationHistory(generations=[], total_generations=0, active_agents=0)
        assert "gens=0" in repr(h)


# ---------------------------------------------------------------------------
# GenerationMemory
# ---------------------------------------------------------------------------

class TestGenerationMemory:
    def test_register(self):
        mem = GenerationMemory()
        g = mem.register("a1", "Alpha", generation=1, purpose="test")
        assert g.agent_id == "a1"
        assert g.name == "Alpha"
        assert g.generation == 1
        assert g.sunset_at is None

    def test_register_with_parent(self):
        mem = GenerationMemory()
        mem.register("p1", "Parent", generation=0)
        c = mem.register("c1", "Child", generation=1, parent_id="p1")
        assert c.parent_id == "p1"
        parent = mem.get("p1")
        assert parent is not None
        assert "c1" in parent.children_spawned

    def test_get_missing(self):
        mem = GenerationMemory()
        assert mem.get("ghost") is None

    def test_sunset(self):
        mem = GenerationMemory()
        mem.register("a1", "Alpha", generation=1)
        ok = mem.sunset("a1", "retired", lessons=["lesson1"], patterns=["pat1"])
        assert ok is True
        g = mem.get("a1")
        assert g is not None
        assert g.sunset_reason == "retired"
        assert g.lessons_learned == ["lesson1"]
        assert g.patterns_preserved == ["pat1"]
        assert g.sunset_at is not None

    def test_sunset_missing(self):
        mem = GenerationMemory()
        ok = mem.sunset("ghost", "gone")
        assert ok is False

    def test_get_history(self):
        mem = GenerationMemory()
        mem.register("a1", "First", generation=1)
        mem.register("a2", "Second", generation=2)
        mem.sunset("a1", "done")
        hist = mem.get_history()
        assert hist.total_generations == 2
        assert hist.active_agents == 1  # a2 still active
        assert len(hist.generations) == 2

    def test_get_history_exclude_sunset(self):
        mem = GenerationMemory()
        mem.register("a1", "A", generation=1)
        mem.sunset("a1", "done")
        mem.register("a2", "B", generation=2)
        hist = mem.get_history(include_sunset=False)
        assert hist.total_generations == 1
        assert hist.active_agents == 1

    def test_get_lineage(self):
        mem = GenerationMemory()
        mem.register("g0", "Grandparent", generation=0)
        mem.register("p1", "Parent", generation=1, parent_id="g0")
        mem.register("c2", "Child", generation=2, parent_id="p1")
        lineage = mem.get_lineage("c2")
        assert len(lineage) == 2
        assert lineage[0].agent_id == "g0"
        assert lineage[1].agent_id == "p1"

    def test_get_lineage_no_parent(self):
        mem = GenerationMemory()
        mem.register("a1", "Orphan", generation=1)
        assert mem.get_lineage("a1") == []

    def test_get_children(self):
        mem = GenerationMemory()
        mem.register("p1", "Parent", generation=0)
        mem.register("c1", "Child1", generation=1, parent_id="p1")
        mem.register("c2", "Child2", generation=1, parent_id="p1")
        children = mem.get_children("p1")
        assert len(children) == 2
        ids = {c.agent_id for c in children}
        assert ids == {"c1", "c2"}

    def test_get_children_missing(self):
        mem = GenerationMemory()
        assert mem.get_children("ghost") == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        store = tmp_path / "generations.json"
        mem = GenerationMemory(str(store))
        mem.register("a1", "Alpha", generation=1)
        mem.register("a2", "Beta", generation=2, parent_id="a1")
        mem.sunset("a1", "retired")

        # Create new instance pointing at same file
        mem2 = GenerationMemory(str(store))
        assert mem2.get("a1") is not None
        assert mem2.get("a2") is not None
        assert mem2.get("a1").sunset_reason == "retired"

    def test_load_corrupt(self, tmp_path):
        store = tmp_path / "bad.json"
        store.write_text("not json")
        mem = GenerationMemory(str(store))
        # should not crash, just start empty
        assert mem.get("x") is None

    def test_save_no_path(self):
        mem = GenerationMemory()
        mem.register("a1", "A", generation=1)
        # _save should be a no-op without store_path
        assert mem._store_path is None
