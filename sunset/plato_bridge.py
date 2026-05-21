"""Sunset-ecosystem ↔ PLATO bridge.

Persists sunset agent lifecycle data (trinity scores, epilogues, phase
transitions) as PLATO tiles so that the rest of the SuperInstance mesh
can observe and react to agent evolution.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from plato_core.types import (
    LamportClock,
    LifecycleEvent,
    TileLifecycle,
    TileType,
    TrainingTile,
    content_hash,
)

from sunset.agent import AgentPhase
from sunset.trinity_scorer import trinity_score
from sunset.sunset_documents import Epilogue


class PlatoBridge:
    """Wraps plato-core to persist sunset-ecosystem data as tiles.

    Each bridge owns a Lamport clock for causal ordering and a room
    namespace (default ``"sunset-ecosystem"``).  All tiles are standard
    ``TrainingTile`` instances so they're first-class citizens in the
    PLATO mesh.
    """

    def __init__(
        self,
        room: str = "sunset-ecosystem",
        clock: Optional[LamportClock] = None,
    ) -> None:
        self.room = room
        self._clock = clock or LamportClock()
        self._store: Dict[str, TrainingTile] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_tile(
        self,
        tile_id: str,
        name: str,
        description: str,
        payload: Dict[str, Any],
        tile_type: TileType = TileType.METRICS,
        parent_tile: str = "",
    ) -> TrainingTile:
        """Build a tile with content-hash and Lamport timestamp."""
        raw = json.dumps(payload, sort_keys=True).encode()
        tile = TrainingTile(
            tile_id=tile_id,
            room=self.room,
            tile_type=tile_type,
            state=TileLifecycle.ACTIVE,
            lamport=self._clock.tick(),
            name=name,
            description=description,
            content_hash=content_hash(raw),
            base_model="sunset-agent",
            source_room=self.room,
            parent_tile=parent_tile,
        )
        # Stash the payload in description metadata so round-trip works.
        # (TrainingTile doesn't have a generic payload field; we encode it.)
        tile._payload = payload  # type: ignore[attr-defined]
        self._store[tile_id] = tile
        return tile

    def _tile_prefix(self, agent_id: str, kind: str) -> str:
        return f"sunset-{kind}-{agent_id}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_trinity_score(
        self,
        agent_id: str,
        scores: Dict[str, float],
    ) -> TrainingTile:
        """Write trinity scores for *agent_id* as a PLATO tile.

        ``scores`` should contain ``ethos``, ``pathos``, and ``logos``
        keys (each in [0, 1]).
        """
        ethos = scores.get("ethos", 0.0)
        pathos = scores.get("pathos", 0.0)
        logos = scores.get("logos", 0.0)
        composite = trinity_score(ethos, pathos, logos)

        payload = {
            "ethos": ethos,
            "pathos": pathos,
            "logos": logos,
            "composite": composite,
        }
        tile_id = self._tile_prefix(agent_id, "trinity")
        tile = self._make_tile(
            tile_id=tile_id,
            name=f"Trinity score for {agent_id}",
            description=f"ethos={ethos:.4f} pathos={pathos:.4f} logos={logos:.4f} → {composite:.6f}",
            payload=payload,
        )
        return tile

    def read_trinity_scores(self, agent_id: str) -> Optional[Dict[str, float]]:
        """Read the last written trinity scores for *agent_id*."""
        tile_id = self._tile_prefix(agent_id, "trinity")
        tile = self._store.get(tile_id)
        if tile is None:
            return None
        return tile._payload  # type: ignore[attr-defined]

    def write_epilogue(
        self,
        agent_id: str,
        epilogue_text: str,
        *,
        generation: int = 0,
        peak_score: float = 0.0,
    ) -> TrainingTile:
        """Store a sunset epilogue as a PLATO tile."""
        payload = {
            "agent_id": agent_id,
            "epilogue_text": epilogue_text,
            "generation": generation,
            "peak_trinity_score": peak_score,
        }
        tile_id = self._tile_prefix(agent_id, "epilogue")
        tile = self._make_tile(
            tile_id=tile_id,
            name=f"Epilogue for {agent_id}",
            description=f"gen={generation} peak={peak_score:.4f}: {epilogue_text[:80]}",
            payload=payload,
            tile_type=TileType.EVALUATION,
        )
        return tile

    def read_epilogue(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored epilogue for *agent_id*."""
        tile_id = self._tile_prefix(agent_id, "epilogue")
        tile = self._store.get(tile_id)
        if tile is None:
            return None
        return tile._payload  # type: ignore[attr-defined]

    def write_lifecycle_event(
        self,
        agent_id: str,
        phase: AgentPhase,
        reason: str = "",
    ) -> TrainingTile:
        """Emit a lifecycle transition tile for *agent_id*."""
        tile_id = self._tile_prefix(agent_id, "lifecycle")

        # If a previous lifecycle tile exists, supersede it.
        parent = ""
        old = self._store.get(tile_id)
        if old is not None:
            parent = old.tile_id
            old.transition(
                TileLifecycle.SUPERSEDED,
                reason=f"Phase changed to {phase.value}",
                lamport=self._clock.tick(),
            )

        payload = {
            "agent_id": agent_id,
            "phase": phase.value,
            "reason": reason,
            "transition_time": time.time(),
        }
        tile = self._make_tile(
            tile_id=tile_id,
            name=f"Lifecycle: {agent_id} → {phase.value}",
            description=f"{agent_id} now in {phase.value}",
            payload=payload,
            tile_type=TileType.METRICS,
            parent_tile=parent,
        )
        return tile

    def read_lifecycle(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve current lifecycle state for *agent_id*."""
        tile_id = self._tile_prefix(agent_id, "lifecycle")
        tile = self._store.get(tile_id)
        if tile is None:
            return None
        return tile._payload  # type: ignore[attr-defined]

    def get_tile(self, tile_id: str) -> Optional[TrainingTile]:
        """Direct tile access for advanced consumers."""
        return self._store.get(tile_id)

    def all_tiles(self) -> List[TrainingTile]:
        """Return every tile managed by this bridge."""
        return list(self._store.values())


__all__ = ["PlatoBridge"]
