"""Tests for sunset-ecosystem ↔ PLATO tile-store bridge.

Covers both the legacy PlatoBridge API and the AgentTileAdapter-based API:
- Trinity score serialization & round-trip
- Epilogue serialization
- Seed bank serialization
- Lifecycle transition serialization
- Agent snapshot serialization
- Read filtering by agent_id, tile_type, state
- Persistence round-trip
- Clear operation
- Hash determinism
- Tile structure
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("plato_core", reason="plato_core not installed")

from plato_core.types import (
    LamportClock,
    TileLifecycle,
    TileType,
    TrainingTile,
    content_hash,
)

from sunset.agent import Agent, AgentPhase
from sunset.plato_bridge import AgentTileAdapter, PlatoBridge
from sunset.seed_bank import SeedBank, SeedEntry
from sunset.sunset_documents import Epilogue, Onboarding
from sunset.trinity_scorer import trinity_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge() -> PlatoBridge:
    return PlatoBridge(room="test-sunset")


# ---------------------------------------------------------------------------
# AgentTileAdapter tests
# ---------------------------------------------------------------------------


class TestAgentTileAdapter:
    def test_phase_to_lifecycle(self):
        assert AgentTileAdapter.phase_to_lifecycle(AgentPhase.INCUBATING) == TileLifecycle.ACTIVE
        assert AgentTileAdapter.phase_to_lifecycle(AgentPhase.SUNSETTING) == TileLifecycle.SUPERSEDED
        assert AgentTileAdapter.phase_to_lifecycle(AgentPhase.ASLEEP) == TileLifecycle.SUPERSEDED

    def test_trinity_tile(self):
        tile = AgentTileAdapter.trinity_tile("a1", 0.9, 0.8, 0.7)
        assert tile.tile_type == TileType.METRICS
        assert tile.room == "a1"
        desc = json.loads(tile.description)
        assert desc["ethos"] == pytest.approx(0.9)
        assert desc["fitness"] == pytest.approx(trinity_score(0.9, 0.8, 0.7))

    def test_epilogue_tile(self):
        epilogue = Epilogue(
            agent_id="a1",
            what_i_tried="foo",
            what_i_found="bar",
            peak_trinity_score=0.5,
            generation=2,
        )
        tile = AgentTileAdapter.epilogue_tile(epilogue)
        assert tile.tile_type == TileType.EVALUATION
        desc = json.loads(tile.description)
        assert desc["what_i_tried"] == "foo"
        assert desc["peak_trinity_score"] == pytest.approx(0.5)

    def test_seed_tile(self):
        onboarding = Onboarding(
            agent_id="a1",
            letter_to_children="hello",
            variant="mutation",
            generation=3,
        )
        bank = SeedBank()
        bank.store(onboarding, relevance=0.9, novelty=0.8)
        entry = list(bank._entries.values())[0]
        tile = AgentTileAdapter.seed_tile(entry)
        assert tile.tile_type == TileType.CHECKPOINT
        desc = json.loads(tile.description)
        assert desc["variant"] == "mutation"
        assert desc["relevance"] == pytest.approx(0.9)

    def test_lifecycle_tile(self):
        tile = AgentTileAdapter.lifecycle_tile(
            "a1", AgentPhase.INCUBATING, AgentPhase.COMPETING, reason="ready"
        )
        assert tile.tile_type == TileType.PREDICTION
        assert len(tile.lifecycle_events) == 1
        event = tile.lifecycle_events[0]
        assert event.reason == "ready"
        assert event.from_state == TileLifecycle.ACTIVE
        assert event.to_state == TileLifecycle.ACTIVE


# ---------------------------------------------------------------------------
# Legacy PlatoBridge tests (room-based API)
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
        assert tile1.state == TileLifecycle.SUPERSEDED
        current = bridge.get_tile("sunset-lifecycle-agent-103")
        assert current._payload["phase"] == "competing"


class TestHashDeterminism:
    def test_same_scores_same_hash(self, bridge: PlatoBridge):
        scores = {"ethos": 0.5, "pathos": 0.5, "logos": 0.5}
        bridge.write_trinity_score("agent-200", scores)
        tile1 = bridge.get_tile("sunset-trinity-agent-200")

        bridge2 = PlatoBridge(room="test-sunset")
        bridge2.write_trinity_score("agent-200", scores)
        tile2 = bridge2.get_tile("sunset-trinity-agent-200")

        assert tile1.content_hash == tile2.content_hash

    def test_different_scores_different_hash(self, bridge: PlatoBridge):
        bridge.write_trinity_score("agent-201", {"ethos": 0.1, "pathos": 0.1, "logos": 0.1})
        bridge.write_trinity_score("agent-202", {"ethos": 0.9, "pathos": 0.9, "logos": 0.9})
        t1 = bridge.get_tile("sunset-trinity-agent-201")
        t2 = bridge.get_tile("sunset-trinity-agent-202")
        assert t1.content_hash != t2.content_hash


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
        assert len(bridge.all_tiles()) == 3


# ---------------------------------------------------------------------------
# Adapter-style PlatoBridge tests
# ---------------------------------------------------------------------------


class TestPlatoBridgeAdapter:
    def test_write_and_read_trinity_score(self):
        bridge = PlatoBridge()
        tile = bridge.write_trinity_score("a1", 0.9, 0.8, 0.7)
        assert bridge.get_tile(tile.tile_id) is not None
        results = bridge.read_trinity_scores(agent_id="a1")
        assert len(results) == 1
        assert results[0].tile_id == tile.tile_id

    def test_write_epilogue(self):
        bridge = PlatoBridge()
        epilogue = Epilogue(agent_id="a1", what_i_tried="x", what_i_found="y")
        tile = bridge.write_epilogue(epilogue)
        assert tile.tile_type == TileType.EVALUATION
        assert bridge.read_epilogues(agent_id="a1")[0].tile_id == tile.tile_id

    def test_write_seed_bank(self):
        bridge = PlatoBridge()
        onboarding = Onboarding(agent_id="a1", letter_to_children="hi")
        bank = SeedBank()
        bank.store(onboarding, relevance=0.7, novelty=0.6)
        entry = list(bank._entries.values())[0]
        tile = bridge.write_seed_bank(entry)
        assert tile.tile_type == TileType.CHECKPOINT
        assert len(bridge.read_seed_bank(agent_id="a1")) == 1

    def test_write_lifecycle_transition(self):
        bridge = PlatoBridge()
        tile = bridge.write_lifecycle_transition(
            "a1", AgentPhase.COMPETING, AgentPhase.SUNSETTING, reason="lost"
        )
        assert tile.tile_type == TileType.PREDICTION
        events = bridge.read_lifecycle(agent_id="a1")[0].lifecycle_events
        assert events[0].reason == "lost"

    def test_write_agent_snapshot(self):
        bridge = PlatoBridge()
        agent = Agent(id="a1", generation=2, phase=AgentPhase.BREEDING, trinity_score=0.85)
        tile = bridge.write_agent_snapshot(agent)
        assert tile.tile_type == TileType.METRICS
        assert tile.state == TileLifecycle.ACTIVE
        desc = json.loads(tile.description)
        assert desc["generation"] == 2
        assert desc["phase"] == "breeding"

    def test_read_filters(self):
        bridge = PlatoBridge()
        bridge.write_trinity_score("a1", 1.0, 1.0, 1.0)
        bridge.write_trinity_score("a2", 0.5, 0.5, 0.5)
        bridge.write_epilogue(Epilogue(agent_id="a1"))

        assert len(bridge.read_tiles(agent_id="a1")) == 2
        assert len(bridge.read_tiles(tile_type=TileType.METRICS)) == 2
        assert len(bridge.read_tiles(agent_id="a1", tile_type=TileType.EVALUATION)) == 1

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.json"
            bridge = PlatoBridge(store_path=str(path))
            bridge.write_trinity_score("a1", 0.9, 0.8, 0.7)
            bridge.write_lifecycle_transition("a1", AgentPhase.INCUBATING, AgentPhase.COMPETING)

            bridge2 = PlatoBridge(store_path=str(path))
            assert len(bridge2._tiles) == 2
            assert bridge2.get_tile("trinity:a1") is not None
            assert len(bridge2.read_lifecycle(agent_id="a1")) == 1

    def test_clear(self):
        bridge = PlatoBridge()
        bridge.write_trinity_score("a1", 1.0, 1.0, 1.0)
        bridge.clear()
        assert len(bridge._tiles) == 0
