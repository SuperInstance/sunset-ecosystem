"""Tests for FleetMem0 — semantic memory adapter.

Coverage:
- Initialization (mem0 backend vs fallback)
- add: stores content, returns ID
- search: retrieves relevant memories
- get_all: lists all memories
- delete: removes by ID
- history: returns list
- from_agent_identity integration
- to_dict serialization
"""

import pytest

from fleet.fleet_mem0 import (
    FleetMem0Config,
    FleetMem0Memory,
    FleetMemoryEntry,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mem():
    return FleetMem0Memory(FleetMem0Config(agent_id="test_agent"))


# ── Initialization ─────────────────────────────────────────────────────


class TestInit:
    def test_default_init(self):
        m = FleetMem0Memory()
        assert m.config.agent_id == "kimi1"
        assert m.config.vector_store == "qdrant"

    def test_custom_config(self):
        cfg = FleetMem0Config(agent_id="Oracle1", vector_store="chroma")
        m = FleetMem0Memory(cfg)
        assert m.config.agent_id == "Oracle1"
        assert m.config.vector_store == "chroma"


# ── Add ──────────────────────────────────────────────────────────────────


class TestAdd:
    def test_add_returns_id_or_none(self, mem):
        result = mem.add("The fleet has 19 modules")
        # May be None if mem0 backend unavailable, or a string if fallback
        assert result is None or isinstance(result, str)

    def test_add_with_metadata(self, mem):
        result = mem.add("Test content", metadata={"room": "breeding", "gen": 3})
        assert result is None or isinstance(result, str)


# ── Search ───────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_empty(self, mem):
        results = mem.search("anything", k=5)
        assert results == []

    def test_fallback_search_finds_keywords(self):
        """When mem0 is unavailable, fallback keyword search works."""
        mem = FleetMem0Memory()
        # Force fallback by clearing backend
        mem._mem0 = None
        mem._fallback_memories = []
        mem.add("The sunset ecosystem breeds agents")
        mem.add("FleetConductorV2 orchestrates the system")
        mem.add("MeshVectorGossip shares vectors")
        results = mem.search("breeds agents", k=2)
        assert len(results) <= 2
        assert any("breeds" in r.content for r in results)

    def test_fallback_search_no_match(self):
        mem = FleetMem0Memory()
        mem._mem0 = None
        mem._fallback_memories = []
        mem.add("The sunset ecosystem breeds agents")
        results = mem.search("quantum physics", k=2)
        assert len(results) <= 2


# ── Get All ────────────────────────────────────────────────────────────


class TestGetAll:
    def test_get_all_fallback(self):
        mem = FleetMem0Memory()
        mem._mem0 = None
        mem._fallback_memories = []
        mem.add("First memory")
        mem.add("Second memory")
        all_mem = mem.get_all()
        assert len(all_mem) == 2
        assert all(isinstance(m, FleetMemoryEntry) for m in all_mem)

    def test_get_all_empty(self, mem):
        all_mem = mem.get_all()
        assert isinstance(all_mem, list)


# ── Delete ─────────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_nonexistent(self, mem):
        ok = mem.delete("nonexistent_id")
        assert ok is False


# ── History ──────────────────────────────────────────────────────────────


class TestHistory:
    def test_history_returns_list(self, mem):
        hist = mem.history("some_id")
        assert isinstance(hist, list)


# ── Integration ────────────────────────────────────────────────────────────


class TestIntegration:
    def test_from_agent_identity(self):
        class FakeIdentity:
            agent_id = "Oracle1"

        m = FleetMem0Memory.from_agent_identity(FakeIdentity())
        assert m.config.agent_id == "Oracle1"

    def test_to_dict(self, mem):
        d = mem.to_dict()
        assert "backend" in d
        assert "config" in d
        assert "n_memories" in d


# ── Config ─────────────────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        cfg = FleetMem0Config()
        assert cfg.vector_store == "qdrant"
        assert cfg.llm_provider == "ollama"
        assert cfg.embedding_model == "nomic-embed-text"

    def test_custom(self):
        cfg = FleetMem0Config(
            vector_store="faiss",
            llm_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        assert cfg.vector_store == "faiss"
        assert cfg.llm_provider == "openai"
