"""PLATO Bridge — wires sunset-ecosystem agents to the PLATO tile store.

Adapters:
- Trinity scores    → METRICS tiles
- Epilogues         → EVALUATION tiles
- Seed bank entries → CHECKPOINT tiles
- Lifecycle events  → lifecycle annotations on ACTIVE tiles

The bridge keeps an in-memory registry that can be persisted to JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from plato_core.types import (
    LamportClock,
    LifecycleEvent,
    TileLifecycle,
    TileType,
    TrainingTile,
)

from sunset.agent import Agent, AgentPhase
from sunset.seed_bank import SeedBank, SeedEntry
from sunset.sunset_documents import Epilogue, Onboarding
from sunset.trinity_scorer import trinity_score

__all__ = ["PlatoBridge", "AgentTileAdapter"]


class AgentTileAdapter:
    """Convert sunset agent concepts to PLATO TrainingTile objects."""

    @staticmethod
    def phase_to_lifecycle(phase: AgentPhase) -> TileLifecycle:
        mapping = {
            AgentPhase.INCUBATING: TileLifecycle.ACTIVE,
            AgentPhase.COMPETING: TileLifecycle.ACTIVE,
            AgentPhase.BREEDING: TileLifecycle.ACTIVE,
            AgentPhase.SUNSETTING: TileLifecycle.SUPERSEDED,
            AgentPhase.ASLEEP: TileLifecycle.SUPERSEDED,
        }
        return mapping.get(phase, TileLifecycle.ACTIVE)

    @staticmethod
    def trinity_tile(
        agent_id: str,
        ethos: float,
        pathos: float,
        logos: float,
        clock: LamportClock | None = None,
    ) -> TrainingTile:
        fitness = trinity_score(ethos, pathos, logos)
        return TrainingTile(
            tile_id=f"trinity:{agent_id}",
            room=agent_id,
            tile_type=TileType.METRICS,
            state=TileLifecycle.ACTIVE,
            lamport=clock.tick() if clock else 0,
            name="trinity_score",
            description=json.dumps({"ethos": ethos, "pathos": pathos, "logos": logos, "fitness": fitness}),
            content_hash=""
        )

    @staticmethod
    def epilogue_tile(epilogue: Epilogue, clock: LamportClock | None = None) -> TrainingTile:
        return TrainingTile(
            tile_id=f"epilogue:{epilogue.agent_id}",
            room=epilogue.agent_id,
            tile_type=TileType.EVALUATION,
            state=TileLifecycle.ACTIVE,
            lamport=clock.tick() if clock else 0,
            name="epilogue",
            description=json.dumps({
                "what_i_tried": epilogue.what_i_tried,
                "what_i_found": epilogue.what_i_found,
                "why_not_relevant": epilogue.why_not_relevant,
                "peak_trinity_score": epilogue.peak_trinity_score,
                "generation": epilogue.generation,
            }),
            content_hash=""
        )

    @staticmethod
    def seed_tile(entry: SeedEntry, clock: LamportClock | None = None) -> TrainingTile:
        onboarding = entry.onboarding
        return TrainingTile(
            tile_id=f"seed:{onboarding.agent_id}:{onboarding.variant}",
            room=onboarding.agent_id,
            tile_type=TileType.CHECKPOINT,
            state=TileLifecycle.ACTIVE,
            lamport=clock.tick() if clock else 0,
            name="seed_bank",
            description=json.dumps({
                "letter": onboarding.letter_to_children,
                "what_works": onboarding.what_works,
                "what_doesnt": onboarding.what_doesnt,
                "where_to_look": onboarding.where_to_look,
                "variant": onboarding.variant,
                "parent_id": onboarding.parent_id,
                "generation": onboarding.generation,
                "relevance": entry.relevance,
                "novelty": entry.novelty,
            }),
            content_hash=""
        )

    @staticmethod
    def lifecycle_tile(
        agent_id: str,
        from_phase: AgentPhase,
        to_phase: AgentPhase,
        reason: str = "",
        clock: LamportClock | None = None,
    ) -> TrainingTile:
        tile = TrainingTile(
            tile_id=f"lifecycle:{agent_id}",
            room=agent_id,
            tile_type=TileType.PREDICTION,
            state=TileLifecycle.ACTIVE,
            lamport=clock.tick() if clock else 0,
            name="lifecycle_transition",
            description=reason,
        )
        event = LifecycleEvent(
            from_state=AgentTileAdapter.phase_to_lifecycle(from_phase),
            to_state=AgentTileAdapter.phase_to_lifecycle(to_phase),
            reason=reason,
            lamport=tile.lamport,
        )
        tile.lifecycle_events.append(event)
        return tile


class PlatoBridge:
    """Adapter that lets sunset agents read/write PLATO tiles.

    Usage::

        bridge = PlatoBridge(store_path="/tmp/plato_store.json")
        bridge.write_trinity_score(agent_id="a1", ethos=0.9, pathos=0.8, logos=0.7)
        tiles = bridge.read_tiles(agent_id="a1")
    """

    def __init__(self, store_path: str | None = None) -> None:
        self._tiles: Dict[str, TrainingTile] = {}
        self._clock = LamportClock()
        self._store_path = Path(store_path) if store_path else None
        if self._store_path and self._store_path.exists():
            self._load()

    # ── write operations ────────────────────────────────────────

    def write_trinity_score(
        self,
        agent_id: str,
        ethos: float,
        pathos: float,
        logos: float,
    ) -> TrainingTile:
        tile = AgentTileAdapter.trinity_tile(agent_id, ethos, pathos, logos, self._clock)
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    def write_epilogue(self, epilogue: Epilogue) -> TrainingTile:
        tile = AgentTileAdapter.epilogue_tile(epilogue, self._clock)
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    def write_seed_bank(self, entry: SeedEntry) -> TrainingTile:
        tile = AgentTileAdapter.seed_tile(entry, self._clock)
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    def write_lifecycle_transition(
        self,
        agent_id: str,
        from_phase: AgentPhase,
        to_phase: AgentPhase,
        reason: str = "",
    ) -> TrainingTile:
        tile = AgentTileAdapter.lifecycle_tile(agent_id, from_phase, to_phase, reason, self._clock)
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    def write_agent_snapshot(self, agent: Agent) -> TrainingTile:
        """Write a full snapshot of an agent as a METRICS tile."""
        tile = TrainingTile(
            tile_id=f"agent:{agent.id}",
            room=agent.room,
            tile_type=TileType.METRICS,
            state=AgentTileAdapter.phase_to_lifecycle(agent.phase),
            lamport=self._clock.tick(),
            name="agent_snapshot",
            description=json.dumps({
                "generation": agent.generation,
                "parent_id": agent.parent_id,
                "phase": agent.phase.value,
                "trinity_score": agent.trinity_score,
                "max_tokens": agent.resource_budget.max_tokens,
                "max_time_seconds": agent.resource_budget.max_time_seconds,
                "parallel_slots": agent.resource_budget.parallel_slots,
            }),
            content_hash=""
        )
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    # ── read operations ─────────────────────────────────────────

    def get_tile(self, tile_id: str) -> TrainingTile | None:
        return self._tiles.get(tile_id)

    def read_tiles(
        self,
        agent_id: str | None = None,
        tile_type: TileType | None = None,
        state: TileLifecycle | None = None,
    ) -> List[TrainingTile]:
        results: List[TrainingTile] = []
        for tile in self._tiles.values():
            if agent_id is not None and tile.room != agent_id:
                continue
            if tile_type is not None and tile.tile_type != tile_type:
                continue
            if state is not None and tile.state != state:
                continue
            results.append(tile)
        return results

    def read_trinity_scores(self, agent_id: str | None = None) -> List[TrainingTile]:
        return self.read_tiles(agent_id=agent_id, tile_type=TileType.METRICS)

    def read_epilogues(self, agent_id: str | None = None) -> List[TrainingTile]:
        return self.read_tiles(agent_id=agent_id, tile_type=TileType.EVALUATION)

    def read_seed_bank(self, agent_id: str | None = None) -> List[TrainingTile]:
        return self.read_tiles(agent_id=agent_id, tile_type=TileType.CHECKPOINT)

    def read_lifecycle(self, agent_id: str | None = None) -> List[TrainingTile]:
        return self.read_tiles(agent_id=agent_id, tile_type=TileType.PREDICTION)

    # ── persistence ─────────────────────────────────────────────

    def _maybe_save(self) -> None:
        if self._store_path:
            self._save()

    def _save(self) -> None:
        data = {tid: tile.to_dict() for tid, tile in self._tiles.items()}
        self._store_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _load(self) -> None:
        raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        for tid, d in raw.items():
            try:
                self._tiles[tid] = TrainingTile.from_dict(d)
            except Exception:
                # Skip corrupt tiles
                continue

    def clear(self) -> None:
        self._tiles.clear()
        if self._store_path and self._store_path.exists():
            self._store_path.unlink()
