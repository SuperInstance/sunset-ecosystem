"""RoomGrid → PLATO tile observer.

Hooks into RoomGrid.tick() to persist every tick's state as PLATO tiles:
- Diversity scores per room (HDC + cosine)
- Thermal snapshots (CPU/GPU/memory)
- Room occupancy matrix
- Agent lifecycle transitions

Uses FM's PlatoBridge (sunset/plato_bridge.py) for actual tile persistence.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional

from sunset.plato_bridge import PlatoBridge

if TYPE_CHECKING:
    from sunset.room_grid import RoomGrid


class RoomGridPlatoObserver:
    """Observes RoomGrid ticks and writes PLATO tiles.

    Attach to a RoomGrid instance; every tick produces 1–4 tiles:
    1. ``diversity`` — RoomGrid.diversity() scores
    2. ``thermal`` — Thermal snapshot if available
    3. ``occupancy`` — Room occupancy counts
    4. ``lifecycle`` — Any agent phase transitions this tick

    Tiles are written with Lamport clock ordering so causality is
    preserved across the PLATO mesh.
    """

    def __init__(self, bridge: Optional[PlatoBridge] = None) -> None:
        self.bridge = bridge or PlatoBridge(room="sunset-roomgrid")
        self._last_thermal: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Hook API — called by RoomGrid.tick() observers
    # ------------------------------------------------------------------

    def on_tick(self, grid: RoomGrid, tick: int, duration_ms: float) -> List[str]:
        """Called after RoomGrid.tick() completes.

        Returns list of tile IDs written this tick.
        """
        written: List[str] = []

        # 1. Diversity scores
        tile = self._write_diversity(grid, tick)
        if tile:
            written.append(tile.tile_id)

        # 2. Thermal snapshot
        tile = self._write_thermal(grid, tick)
        if tile:
            written.append(tile.tile_id)

        # 3. Room occupancy
        tile = self._write_occupancy(grid, tick)
        if tile:
            written.append(tile.tile_id)

        return written

    def on_agent_spawn(self, grid: RoomGrid, agent_id: str, room: str) -> Optional[str]:
        """Called when an agent is spawned into a room."""
        tile = self.bridge.write_lifecycle_event(
            agent_id=agent_id,
            phase=None,  # RoomGrid doesn't track agent phases
            reason=f"spawned into {room}",
        )
        return tile.tile_id

    def on_agent_sunset(
        self, grid: RoomGrid, agent_id: str, reason: str
    ) -> Optional[str]:
        """Called when an agent is sunset."""
        tile = self.bridge.write_lifecycle_event(
            agent_id=agent_id,
            phase=None,
            reason=f"sunset: {reason}",
        )
        return tile.tile_id

    # ------------------------------------------------------------------
    # Internal writers
    # ------------------------------------------------------------------

    def _write_diversity(self, grid: RoomGrid, tick: int):
        """Write diversity scores for the current room grid state."""
        try:
            score = grid.diversity(use_hdc=False)
        except Exception:
            return None  # diversity() may fail if no agents

        payload = {
            "tick": tick,
            "room_count": grid.n,
            "active_rooms": grid.agent_count(),
            "diversity_score": float(score),
        }
        return self.bridge._make_tile(
            tile_id=f"rg-diversity-{tick}",
            name=f"RoomGrid diversity @ tick {tick}",
            description=f"{grid.n} rooms, {grid.agent_count()} active, score={score:.4f}",
            payload=payload,
        )

    def _write_thermal(self, grid: RoomGrid, tick: int):
        """Write thermal snapshot if thermal manager is available."""
        thermal = getattr(grid, "thermal", None)
        if thermal is None:
            return None

        try:
            snap = thermal.snapshot()
        except Exception:
            return None

        payload = {
            "tick": tick,
            "timestamp": time.time(),
            **snap,
        }
        return self.bridge._make_tile(
            tile_id=f"rg-thermal-{tick}",
            name=f"Thermal snapshot @ tick {tick}",
            description=f"cpu={snap.get('cpu_percent', 0):.1f}% "
            f"mem={snap.get('memory_percent', 0):.1f}%",
            payload=payload,
        )

    def _write_occupancy(self, grid: RoomGrid, tick: int):
        """Write room occupancy counts."""
        active_count = grid.agent_count()

        payload = {
            "tick": tick,
            "total_rooms": grid.n,
            "active_rooms": active_count,
            "cold_rooms": grid.n - active_count,
        }
        return self.bridge._make_tile(
            tile_id=f"rg-occupancy-{tick}",
            name=f"Room occupancy @ tick {tick}",
            description=f"{active_count}/{grid.n} rooms active",
            payload=payload,
        )


__all__ = ["RoomGridPlatoObserver"]
