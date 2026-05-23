"""Tests for RoomGridPlatoObserver.

Mocks plato_core types so tests run without the PLATO dependency.
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import sys

# ── Mock plato_core before importing observer ──
_mock_plato = types.ModuleType("plato_core")
_mock_plato_types = types.ModuleType("plato_core.types")


class _MockLamportClock:
    def __init__(self):
        self._tick = 0

    def tick(self) -> int:
        self._tick += 1
        return self._tick


class _MockTileLifecycle:
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class _MockTileType:
    METRICS = "metrics"
    EVALUATION = "evaluation"


class _MockTrainingTile:
    def __init__(self, tile_id: str, room: str, tile_type: str, state: str,
                 lamport: int, name: str, description: str, content_hash: str,
                 base_model: str, source_room: str, parent_tile: str = "") -> None:
        self.tile_id = tile_id
        self.room = room
        self.tile_type = tile_type
        self.state = state
        self.lamport = lamport
        self.name = name
        self.description = description
        self.content_hash = content_hash
        self.base_model = base_model
        self.source_room = source_room
        self.parent_tile = parent_tile
        self._payload: Dict[str, Any] = {}

    def transition(self, new_state: str, reason: str = "", lamport: int = 0) -> None:
        self.state = new_state


_mock_plato_types.LifecycleEvent = type("LifecycleEvent", (), {})  # stub
_mock_plato_types.LamportClock = _MockLamportClock
_mock_plato_types.TileLifecycle = _MockTileLifecycle
_mock_plato_types.TileType = _MockTileType
_mock_plato_types.TrainingTile = _MockTrainingTile
_mock_plato_types.content_hash = lambda x: "mock-hash"

_mock_plato.types = _mock_plato_types
sys.modules["plato_core"] = _mock_plato
sys.modules["plato_core.types"] = _mock_plato_types

# Now safe to import
from sunset.roomgrid_plato_observer import RoomGridPlatoObserver
from sunset.plato_bridge import PlatoBridge
from nerve.room_grid import RoomGrid


class TestRoomGridPlatoObserver:
    """Observer writes PLATO tiles on RoomGrid events."""

    def test_observer_writes_diversity_tile(self):
        grid = RoomGrid(n=10)
        bridge = PlatoBridge(room="test-roomgrid")
        obs = RoomGridPlatoObserver(bridge=bridge)
        grid.attach_plato_observer(obs)

        # Run a few ticks to activate rooms
        for _ in range(3):
            grid.tick(np.random.randn(64))

        tiles = bridge.all_tiles()
        diversity_tiles = [t for t in tiles if "diversity" in t.tile_id]
        assert len(diversity_tiles) == 3
        # Last tile should have payload
        last = diversity_tiles[-1]
        assert last._payload["tick"] == 3
        assert last._payload["room_count"] == 10
        assert "diversity_score" in last._payload

    def test_observer_writes_thermal_tile_when_thermal_available(self):
        grid = RoomGrid(n=5)
        # Mock thermal manager
        class MockThermal:
            def snapshot(self):
                return {"cpu_percent": 12.5, "memory_percent": 45.0}
        grid.thermal = MockThermal()

        bridge = PlatoBridge(room="test-thermal")
        obs = RoomGridPlatoObserver(bridge=bridge)
        grid.attach_plato_observer(obs)

        grid.tick(np.random.randn(64))

        tiles = bridge.all_tiles()
        thermal_tiles = [t for t in tiles if "thermal" in t.tile_id]
        assert len(thermal_tiles) == 1
        assert thermal_tiles[0]._payload["cpu_percent"] == 12.5

    def test_observer_skips_thermal_when_none(self):
        grid = RoomGrid(n=5)
        bridge = PlatoBridge(room="test-no-thermal")
        obs = RoomGridPlatoObserver(bridge=bridge)
        grid.attach_plato_observer(obs)

        grid.tick(np.random.randn(64))

        tiles = bridge.all_tiles()
        thermal_tiles = [t for t in tiles if "thermal" in t.tile_id]
        assert len(thermal_tiles) == 0

    def test_observer_writes_occupancy_tile(self):
        grid = RoomGrid(n=5)
        bridge = PlatoBridge(room="test-occupancy")
        obs = RoomGridPlatoObserver(bridge=bridge)
        grid.attach_plato_observer(obs)

        grid.tick(np.random.randn(64))

        tiles = bridge.all_tiles()
        occupancy_tiles = [t for t in tiles if "occupancy" in t.tile_id]
        assert len(occupancy_tiles) == 1
        assert "active_rooms" in occupancy_tiles[0]._payload
        assert "cold_rooms" in occupancy_tiles[0]._payload

    def test_on_agent_spawn_writes_lifecycle_tile(self):
        grid = RoomGrid(n=5)
        bridge = PlatoBridge(room="test-spawn")
        obs = RoomGridPlatoObserver(bridge=bridge)

        tile_id = obs.on_agent_spawn(grid, "agent-42", "harbor")

        assert tile_id is not None
        assert "agent-42" in tile_id
        tile = bridge.get_tile(tile_id)
        assert tile is not None
        assert tile._payload["phase"] is not None

    def test_on_agent_sunset_writes_lifecycle_tile(self):
        grid = RoomGrid(n=5)
        bridge = PlatoBridge(room="test-sunset")
        obs = RoomGridPlatoObserver(bridge=bridge)

        tile_id = obs.on_agent_sunset(grid, "agent-99", "thermal_limit")

        assert tile_id is not None
        assert "agent-99" in tile_id
        tile = bridge.get_tile(tile_id)
        assert "sunset" in tile._payload.get("reason", "")

    def test_multiple_ticks_lamport_monotonic(self):
        grid = RoomGrid(n=5)
        bridge = PlatoBridge(room="test-lamport")
        obs = RoomGridPlatoObserver(bridge=bridge)
        grid.attach_plato_observer(obs)

        for _ in range(5):
            grid.tick(np.random.randn(64))

        tiles = bridge.all_tiles()
        lamports = [t.lamport for t in tiles]
        assert lamports == sorted(lamports)
        assert len(set(lamports)) == len(lamports)  # all unique

    def test_batch_tick_writes_tiles(self):
        grid = RoomGrid(n=5)
        bridge = PlatoBridge(room="test-batch")
        obs = RoomGridPlatoObserver(bridge=bridge)
        grid.attach_plato_observer(obs)

        signals = np.random.randn(3, 64)
        grid.tick_batch(signals)

        tiles = bridge.all_tiles()
        # 3 ticks × up to 3 tile types = up to 9 tiles
        assert len(tiles) >= 3

    def test_invalid_observer_rejected(self):
        grid = RoomGrid(n=5)
        class BadObserver:
            pass
        with pytest.raises(TypeError):
            grid.attach_plato_observer(BadObserver())
