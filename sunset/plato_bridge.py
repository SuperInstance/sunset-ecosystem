"""Sunset-ecosystem ↔ PLATO bridge.

Persists sunset agent lifecycle data (trinity scores, epilogues, phase
transitions, seed bank entries) as PLATO tiles so that the rest of the
SuperInstance mesh can observe and react to agent evolution.

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
    content_hash,
)

from sunset.agent import Agent, AgentPhase
from sunset.seed_bank import SeedBank, SeedEntry
from sunset.sunset_documents import Epilogue, Onboarding
from sunset.trinity_scorer import trinity_score

__all__ = ["PlatoBridge", "AgentTileAdapter"]


# ---------------------------------------------------------------------------
# AgentTileAdapter — static helpers to convert sunset concepts to tiles
# ---------------------------------------------------------------------------


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
            content_hash="",
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
            content_hash="",
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
            content_hash="",
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


# ---------------------------------------------------------------------------
# PlatoBridge — main adapter for reading/writing PLATO tiles
# ---------------------------------------------------------------------------


class PlatoBridge:
    """Wraps plato-core to persist sunset-ecosystem data as tiles.

    Supports two construction modes:

    1. Legacy (room + optional clock) — original in-memory store keyed by
       ``sunset-{kind}-{agent_id}`` tile IDs.
    2. Store-path — persisted JSON store using ``AgentTileAdapter``-style
       tile IDs (``trinity:{agent_id}``, etc.).

    Both APIs are available regardless of construction mode.
    """

    def __init__(
        self,
        room: str = "sunset-ecosystem",
        clock: Optional[LamportClock] = None,
        store_path: Optional[str] = None,
    ) -> None:
        self.room = room
        self._clock = clock or LamportClock()
        self._store: Dict[str, TrainingTile] = {}  # legacy store
        self._tiles: Dict[str, TrainingTile] = {}   # adapter-style store
        self._store_path = Path(store_path) if store_path else None
        if self._store_path and self._store_path.exists():
            self._load()

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
        tile._payload = payload  # type: ignore[attr-defined]
        self._store[tile_id] = tile
        return tile

    def _tile_prefix(self, agent_id: str, kind: str) -> str:
        return f"sunset-{kind}-{agent_id}"

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
                continue

    # ------------------------------------------------------------------
    # Legacy write API (room-based, _store-backed)
    # ------------------------------------------------------------------

    def write_trinity_score(
        self,
        agent_id: str,
        scores_or_ethos: Any = None,
        pathos: Optional[float] = None,
        logos: Optional[float] = None,
        *,
        generation: int = 0,
        peak_score: float = 0.0,
    ) -> TrainingTile:
        """Write trinity scores for *agent_id* as a PLATO tile.

        Accepts two calling conventions:

        * Legacy dict: ``write_trinity_score("a1", {"ethos": 0.8, ...})``
        * Adapter args: ``write_trinity_score("a1", ethos=0.8, pathos=0.7, logos=0.9)``
        * Mixed: ``write_trinity_score("a1", 0.8, 0.7, 0.9)``
        """
        # Adapter-style: positional (agent_id, ethos, pathos, logos)
        if isinstance(scores_or_ethos, (int, float)) and pathos is not None and logos is not None:
            tile = AgentTileAdapter.trinity_tile(agent_id, float(scores_or_ethos), pathos, logos, self._clock)
            self._tiles[tile.tile_id] = tile
            self._maybe_save()
            return tile

        # Legacy dict-style
        if isinstance(scores_or_ethos, dict):
            scores = scores_or_ethos
            ethos = scores.get("ethos", 0.0)
            pathos_val = scores.get("pathos", 0.0)
            logos_val = scores.get("logos", 0.0)
            composite = trinity_score(ethos, pathos_val, logos_val)

            payload = {
                "ethos": ethos,
                "pathos": pathos_val,
                "logos": logos_val,
                "composite": composite,
            }
            tile_id = self._tile_prefix(agent_id, "trinity")
            tile = self._make_tile(
                tile_id=tile_id,
                name=f"Trinity score for {agent_id}",
                description=f"ethos={ethos:.4f} pathos={pathos_val:.4f} logos={logos_val:.4f} → {composite:.6f}",
                payload=payload,
            )
            return tile

        # Adapter-style keyword
        ethos_val = scores_or_ethos if isinstance(scores_or_ethos, (int, float)) else 0.0
        p = pathos if pathos is not None else 0.0
        l = logos if logos is not None else 0.0
        tile = AgentTileAdapter.trinity_tile(agent_id, float(ethos_val), float(p), float(l), self._clock)
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    def write_epilogue(
        self,
        agent_id_or_epilogue: Any,
        epilogue_text: Optional[str] = None,
        *,
        generation: int = 0,
        peak_score: float = 0.0,
    ) -> TrainingTile:
        """Store a sunset epilogue as a PLATO tile.

        Accepts:
        * ``write_epilogue("a1", "text", generation=2, peak_score=0.5)`` (legacy)
        * ``write_epilogue(Epilogue(...))`` (adapter-style)
        """
        # Adapter-style: single Epilogue object
        if isinstance(agent_id_or_epilogue, Epilogue):
            tile = AgentTileAdapter.epilogue_tile(agent_id_or_epilogue, self._clock)
            self._tiles[tile.tile_id] = tile
            self._maybe_save()
            return tile

        # Legacy: agent_id + epilogue_text
        agent_id = agent_id_or_epilogue
        payload = {
            "agent_id": agent_id,
            "epilogue_text": epilogue_text or "",
            "generation": generation,
            "peak_trinity_score": peak_score,
        }
        tile_id = self._tile_prefix(agent_id, "epilogue")
        tile = self._make_tile(
            tile_id=tile_id,
            name=f"Epilogue for {agent_id}",
            description=f"gen={generation} peak={peak_score:.4f}: {(epilogue_text or '')[:80]}",
            payload=payload,
            tile_type=TileType.EVALUATION,
        )
        return tile

    def write_seed_bank(self, entry: SeedEntry) -> TrainingTile:
        """Write a seed bank entry as a CHECKPOINT tile."""
        tile = AgentTileAdapter.seed_tile(entry, self._clock)
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    def write_lifecycle_event(
        self,
        agent_id: str,
        phase: Optional[Any] = None,
        reason: str = "",
    ) -> TrainingTile:
        """Emit a lifecycle transition tile for *agent_id* (legacy API)."""
        tile_id = self._tile_prefix(agent_id, "lifecycle")

        parent = ""
        old = self._store.get(tile_id)
        phase_str = phase.value if phase is not None else "unknown"
        if old is not None:
            parent = old.tile_id
            old.transition(
                TileLifecycle.SUPERSEDED,
                reason=f"Phase changed to {phase_str}",
                lamport=self._clock.tick(),
            )

        payload = {
            "agent_id": agent_id,
            "phase": phase_str,
            "reason": reason,
            "transition_time": time.time(),
        }
        tile = self._make_tile(
            tile_id=tile_id,
            name=f"Lifecycle: {agent_id} → {phase_str}",
            description=f"{agent_id} now in {phase_str}",
            payload=payload,
            tile_type=TileType.METRICS,
            parent_tile=parent,
        )
        return tile

    def write_lifecycle_transition(
        self,
        agent_id: str,
        from_phase: AgentPhase,
        to_phase: AgentPhase,
        reason: str = "",
    ) -> TrainingTile:
        """Write a lifecycle transition using AgentTileAdapter (adapter-style)."""
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
            content_hash="",
        )
        self._tiles[tile.tile_id] = tile
        self._maybe_save()
        return tile

    # ------------------------------------------------------------------
    # Legacy read API (room-based, _store-backed)
    # ------------------------------------------------------------------

    def read_trinity_scores(self, agent_id: Optional[str] = None) -> Any:
        """Read trinity scores.

        If *agent_id* is a string, tries the legacy store first (returns dict
        or None), then falls back to adapter-style read (returns list).
        """
        if agent_id is not None:
            # Legacy lookup
            tile_id = self._tile_prefix(agent_id, "trinity")
            tile = self._store.get(tile_id)
            if tile is not None:
                return tile._payload  # type: ignore[attr-defined]
            # Adapter-style fallback
            result = self.read_tiles(agent_id=agent_id, tile_type=TileType.METRICS)
            return result if result else None
        # No agent_id: return adapter-style
        return self.read_tiles(tile_type=TileType.METRICS)

    def read_epilogue(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored epilogue for *agent_id* (legacy)."""
        tile_id = self._tile_prefix(agent_id, "epilogue")
        tile = self._store.get(tile_id)
        if tile is None:
            return None
        return tile._payload  # type: ignore[attr-defined]

    def read_epilogues(self, agent_id: Optional[str] = None) -> List[TrainingTile]:
        """Read epilogue tiles (adapter-style)."""
        return self.read_tiles(agent_id=agent_id, tile_type=TileType.EVALUATION)

    def read_seed_bank(self, agent_id: Optional[str] = None) -> List[TrainingTile]:
        """Read seed bank tiles."""
        return self.read_tiles(agent_id=agent_id, tile_type=TileType.CHECKPOINT)

    def read_lifecycle(self, agent_id: Optional[str] = None) -> Any:
        """Retrieve lifecycle state.

        If *agent_id* is a string, tries legacy store first (returns dict or
        None), then falls back to adapter-style (returns list).
        """
        if agent_id is not None:
            tile_id = self._tile_prefix(agent_id, "lifecycle")
            tile = self._store.get(tile_id)
            if tile is not None:
                return tile._payload  # type: ignore[attr-defined]
            return self.read_tiles(agent_id=agent_id, tile_type=TileType.PREDICTION)
        return self.read_tiles(tile_type=TileType.PREDICTION)

    # ------------------------------------------------------------------
    # Adapter-style read API (_tiles-backed)
    # ------------------------------------------------------------------

    def get_tile(self, tile_id: str) -> Optional[TrainingTile]:
        """Direct tile access for advanced consumers."""
        return self._store.get(tile_id) or self._tiles.get(tile_id)

    def read_tiles(
        self,
        agent_id: Optional[str] = None,
        tile_type: Optional[TileType] = None,
        state: Optional[TileLifecycle] = None,
    ) -> List[TrainingTile]:
        """Read tiles with optional filters."""
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

    def all_tiles(self) -> List[TrainingTile]:
        """Return every tile managed by this bridge."""
        return list(self._store.values()) + list(self._tiles.values())

    # ------------------------------------------------------------------
    # Persistence & cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all tiles."""
        self._store.clear()
        self._tiles.clear()
        if self._store_path and self._store_path.exists():
            self._store_path.unlink()
