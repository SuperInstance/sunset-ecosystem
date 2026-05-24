"""Tests for sunset ↔ PLATO bridge."""

import json
import time

import pytest

pytest.importorskip("plato_core", reason="plato_core not installed")

from plato_core.types import (
    LamportClock,
    TileLifecycle,
    TileType,
    TrainingTile,
    content_hash,
)

from sunset.agent import AgentPhase
from sunset.plato_bridge import PlatoBridge
from sunset.trinity_scorer import trinity_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bridge() -> PlatoBridge:
    return PlatoBridge(room="test-sunset")


# ---------------------------------------------------------------------------
# Trinity score round-trip
# ---------------------------------------------------------------------------

class TestTrinityScoreRoundTrip:
    def test_write_read_roundtrip(self, bridge: PlatoBridge):
        scores = {"ethos": 0.8, "pathos": 0.7, "logos": 0.9}
        bridge.write_trinity_score("agent-001", scores)
        result = bridge.read_trinity_scores("agent-001")
        assert result is not None
        assert abs(result["ethos"] - 0.8) < 1e-9
        assert abs(result["pathos"] - 0.7) < 1e-9
        assert abs(result["logos"] - 0.9) < 1e-9

    def test_composite_matches_scorer(self, bridge: PlatoBridge):
        scores = {"ethos": 0.5, "pathos": 0.6, "logos": 0.4}
        bridge.write_trinity_score("agent-002", scores)
        result = bridge.read_trinity_scores("agent-002")
        expected = trinity_score(0.5, 0.6, 0.4)
        assert abs(result["composite"] - expected) < 1e-12

    def test_read_missing_returns_none(self, bridge: PlatoBridge):
        assert bridge.read_trinity_scores("no-such-agent") is None

    def test_overwrite_replaces(self, bridge: PlatoBridge):
        bridge.write_trinity_score("agent-003", {"ethos": 0.1, "pathos": 0.2, "logos": 0.3})
        bridge.write_trinity_score("agent-003", {"ethos": 0.9, "pathos": 0.9, "logos": 0.9})
        result = bridge.read_trinity_scores("agent-003")
        assert abs(result["ethos"] - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# Epilogue storage and retrieval
# ---------------------------------------------------------------------------

class TestEpilogue:
    def test_write_read_epilogue(self, bridge: PlatoBridge):
        bridge.write_epilogue(
            "agent-010",
            "I explored the constraint space but found no viable path.",
            generation=3,
            peak_score=0.12,
        )
        result = bridge.read_epilogue("agent-010")
        assert result is not None
        assert "constraint space" in result["epilogue_text"]
        assert result["generation"] == 3
        assert abs(result["peak_trinity_score"] - 0.12) < 1e-9

    def test_read_missing_epilogue(self, bridge: PlatoBridge):
        assert bridge.read_epilogue("ghost-agent") is None

    def test_epilogue_tile_type(self, bridge: PlatoBridge):
        tile = bridge.write_epilogue("agent-011", "Farewell.")
        assert tile.tile_type == TileType.EVALUATION


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------

class TestLifecycleTransitions:
    def test_incubating_to_competing(self, bridge: PlatoBridge):
        bridge.write_lifecycle_event("agent-100", AgentPhase.INCUBATING)
        bridge.write_lifecycle_event("agent-100", AgentPhase.COMPETING)
        result = bridge.read_lifecycle("agent-100")
        assert result["phase"] == "competing"

    def test_full_lifecycle_chain(self, bridge: PlatoBridge):
        """INCUBATING → COMPETING → SUNSETTING in sequence."""
        for phase in [AgentPhase.INCUBATING, AgentPhase.COMPETING, AgentPhase.SUNSETTING]:
            bridge.write_lifecycle_event("agent-101", phase)
        result = bridge.read_lifecycle("agent-101")
        assert result["phase"] == "sunsetting"

    def test_lifecycle_tile_has_lamport(self, bridge: PlatoBridge):
        clock = LamportClock(0)
        b = PlatoBridge(room="lamport-test", clock=clock)
        tile = b.write_lifecycle_event("agent-102", AgentPhase.INCUBATING)
        assert tile.lamport > 0

    def test_superseded_previous_lifecycle(self, bridge: PlatoBridge):
        tile1 = bridge.write_lifecycle_event("agent-103", AgentPhase.INCUBATING)
        assert tile1.is_active()
        tile2 = bridge.write_lifecycle_event("agent-103", AgentPhase.COMPETING)
        # tile1 was superseded (its state mutated in-memory before being overwritten)
        assert tile1.state == TileLifecycle.SUPERSEDED
        # The current tile in the store is tile2
        current = bridge.get_tile("sunset-lifecycle-agent-103")
        assert current.payload if hasattr(current, 'payload') else current._payload["phase"] == "competing"


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------

class TestHashDeterminism:
    def test_same_scores_same_hash(self, bridge: PlatoBridge):
        scores = {"ethos": 0.5, "pathos": 0.5, "logos": 0.5}
        bridge.write_trinity_score("agent-200", scores)
        tile1 = bridge.get_tile(f"sunset-trinity-agent-200")

        # Create a fresh bridge, write same data
        bridge2 = PlatoBridge(room="test-sunset")
        bridge2.write_trinity_score("agent-200", scores)
        tile2 = bridge2.get_tile(f"sunset-trinity-agent-200")

        assert tile1.content_hash == tile2.content_hash

    def test_different_scores_different_hash(self, bridge: PlatoBridge):
        bridge.write_trinity_score("agent-201", {"ethos": 0.1, "pathos": 0.1, "logos": 0.1})
        bridge.write_trinity_score("agent-202", {"ethos": 0.9, "pathos": 0.9, "logos": 0.9})
        t1 = bridge.get_tile("sunset-trinity-agent-201")
        t2 = bridge.get_tile("sunset-trinity-agent-202")
        assert t1.content_hash != t2.content_hash


# ---------------------------------------------------------------------------
# Tile structure
# ---------------------------------------------------------------------------

class TestTileStructure:
    def test_tile_has_correct_room(self, bridge: PlatoBridge):
        tile = bridge.write_trinity_score("agent-300", {"ethos": 0.5, "pathos": 0.5, "logos": 0.5})
        assert tile.room == "test-sunset"

    def test_tile_is_active(self, bridge: PlatoBridge):
        tile = bridge.write_epilogue("agent-301", "Done.")
        assert tile.is_active()

    def test_all_tiles_returns_everything(self, bridge: PlatoBridge):
        bridge.write_trinity_score("a", {"ethos": 0.5, "pathos": 0.5, "logos": 0.5})
        bridge.write_epilogue("a", "bye")
        bridge.write_lifecycle_event("a", AgentPhase.SUNSETTING)
        # 3 distinct tile IDs, but lifecycle overwrites per-agent so
        # trinity + epilogue + lifecycle = 3
        assert len(bridge.all_tiles()) == 3
