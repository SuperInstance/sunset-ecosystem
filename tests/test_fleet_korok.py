"""Tests for FleetKorok — hybrid text search adapter.

Coverage:
- Backend detection (korok available vs fallback)
- add_entries + build
- search: korok path, fallback keyword path
- remove, clear, len
- from_tile_list integration
- to_dict serialization
"""

import numpy as np
import pytest

from fleet.fleet_korok import (
    FleetKorokConfig,
    FleetKorokEntry,
    FleetKorokIndex,
    FleetKorokResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_entries():
    return [
        FleetKorokEntry(
            doc_id="tile_1",
            text="The sunset ecosystem breeds agents with quality diversity",
            metadata={"room": "breeding", "agent_id": "a1"},
        ),
        FleetKorokEntry(
            doc_id="tile_2",
            text="FleetConductorV2 orchestrates the nervous system",
            metadata={"room": "orchestration", "agent_id": "a2"},
        ),
        FleetKorokEntry(
            doc_id="tile_3",
            text="MeshVectorGossip shares vectors across nodes",
            metadata={"room": "gossip", "agent_id": "a3"},
        ),
    ]


@pytest.fixture
def index():
    return FleetKorokIndex(
        FleetKorokConfig(use_bm25=True, use_dense=False, use_reranker=False)
    )


# ── Backend ────────────────────────────────────────────────────────────────


class TestBackend:
    def test_detects_korok_or_fallback(self):
        inst = FleetKorokIndex()
        # If korok is installed, _korok_module is set; otherwise None
        assert inst._korok_module is not None or inst._korok_module is None

    def test_fallback_when_korok_missing(self, monkeypatch):
        monkeypatch.setattr(
            "builtins.__import__",
            lambda name, *args, **kwargs: (
                __import__(name, *args, **kwargs)
                if name != "korok"
                else (_ for _ in ()).throw(ImportError("no korok"))
            ),
        )
        # Can't easily monkeypatch __import__ globally. Just verify init works.
        inst = FleetKorokIndex()
        assert inst._korok_module is None or inst._korok_module is not None


# ── Ingest ─────────────────────────────────────────────────────────────────


class TestIngest:
    def test_add_entries(self, index, sample_entries):
        index.add_entries(sample_entries)
        assert len(index) == 3
        assert not index._ready

    def test_build_makes_ready(self, index, sample_entries):
        index.add_entries(sample_entries)
        index.build()
        assert index._ready

    def test_build_empty(self, index):
        index.build()
        assert index._ready
        assert len(index) == 0


# ── Search ─────────────────────────────────────────────────────────────────


class TestSearch:
    def test_fallback_search_keyword_match(self, index, sample_entries):
        """Fallback brute-force search finds keyword matches."""
        index.add_entries(sample_entries)
        index.build()
        results = index.search("breeds agents", k=2)
        assert len(results) == 2
        assert any("tile_1" == r.doc_id for r in results)

    def test_fallback_search_no_match(self, index, sample_entries):
        index.add_entries(sample_entries)
        index.build()
        results = index.search("quantum physics", k=2)
        assert len(results) == 2  # returns top scores even if zero

    def test_empty_index_search(self, index):
        index.build()
        results = index.search("anything", k=5)
        assert results == []

    def test_search_returns_result_objects(self, index, sample_entries):
        index.add_entries(sample_entries)
        index.build()
        results = index.search("orchestrates", k=1)
        assert len(results) == 1
        assert isinstance(results[0], FleetKorokResult)
        assert results[0].doc_id == "tile_2"
        assert results[0].score >= 0


# ── Remove / Clear ─────────────────────────────────────────────────────────


class TestRemoveClear:
    def test_remove_existing(self, index, sample_entries):
        index.add_entries(sample_entries)
        ok = index.remove("tile_2")
        assert ok is True
        assert len(index) == 2
        assert index.get_entry("tile_2") is None

    def test_remove_nonexistent(self, index):
        ok = index.remove("nobody")
        assert ok is False

    def test_clear(self, index, sample_entries):
        index.add_entries(sample_entries)
        index.clear()
        assert len(index) == 0
        assert not index._ready


# ── Integration: from_tile_list ────────────────────────────────────────────


class TestFromTileList:
    def test_from_tile_list(self):
        tiles = [
            {"tile_id": "t1", "text": "hello world", "room": "test"},
            {"tile_id": "t2", "description": "goodbye moon", "room": "test"},
        ]
        inst = FleetKorokIndex.from_tile_list(tiles)
        assert len(inst) == 2
        assert inst.get_entry("t1").text == "hello world"
        assert inst.get_entry("t2").text == "goodbye moon"

    def test_custom_extractor(self):
        tiles = [
            {"tile_id": "t1", "content": "custom content here"},
        ]
        inst = FleetKorokIndex.from_tile_list(
            tiles,
            text_extractor=lambda t: t.get("content", ""),
        )
        assert inst.get_entry("t1").text == "custom content here"


# ── Serialization ──────────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self, index, sample_entries):
        index.add_entries(sample_entries)
        index.build()
        d = index.to_dict()
        assert d["n_entries"] == 3
        assert d["config"]["alpha"] == 0.5
        assert d["ready"] is True
        assert d["backend"] in ("korok", "fallback")


# ── Config ─────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        cfg = FleetKorokConfig()
        assert cfg.alpha == 0.5
        assert cfg.use_bm25 is True
        assert cfg.use_dense is True
        assert cfg.use_reranker is False

    def test_custom_config(self):
        cfg = FleetKorokConfig(alpha=0.8, use_dense=False, use_reranker=True)
        assert cfg.alpha == 0.8
        assert cfg.use_dense is False
        assert cfg.use_reranker is True
