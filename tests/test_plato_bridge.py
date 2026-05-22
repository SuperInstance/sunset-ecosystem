"""Tests for sunset-ecosystem ↔ PLATO tile-store bridge.

Covers:
- Trinity score serialization
- Epilogue serialization
- Seed bank serialization
- Lifecycle transition serialization
- Agent snapshot serialization
- Read filtering by agent_id, tile_type, state
- Persistence round-trip
- Clear operation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from plato_core.types import TileLifecycle, TileType
from sunset.agent import Agent, AgentPhase
from sunset.plato_bridge import AgentTileAdapter, PlatoBridge
from sunset.seed_bank import SeedBank
from sunset.sunset_documents import Epilogue, Onboarding
from sunset.trinity_scorer import trinity_score


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


class TestPlatoBridge:
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

            # Reload from disk
            bridge2 = PlatoBridge(store_path=str(path))
            assert len(bridge2._tiles) == 2
            assert bridge2.get_tile("trinity:a1") is not None
            assert len(bridge2.read_lifecycle(agent_id="a1")) == 1

    def test_clear(self):
        bridge = PlatoBridge()
        bridge.write_trinity_score("a1", 1.0, 1.0, 1.0)
        bridge.clear()
        assert len(bridge._tiles) == 0
