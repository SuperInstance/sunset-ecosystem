"""JepaGridMemory — temporal vector store for fleet room state.

JEPA (Joint Embedding Predictive Architecture) inspired: stores
room state vectors at each tick with temporal links, enabling
predictive queries like "what will room 42 look like in 3 ticks?"

Uses FluxVectorTable as the compressed storage backend.
"""

from __future__ import annotations

__all__ = ["JepaGridMemory", "TemporalSlice"]

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalSlice:
    """One room's state at a specific tick."""

    room_id: int
    tick: int
    vector: list[float]
    activity: float
    chaos: float
    metadata: dict[str, Any]


class JepaGridMemory:
    """Temporal room-state memory with predictive search.

    Stores per-tick room vectors and builds temporal edges between
    consecutive ticks. Supports:
        - Retrieve past state at tick T
        - Predict next state (linear extrapolation from last N ticks)
        - Find rooms with similar *trajectories* (not just current state)

    Args:
        dim: Vector dimensionality (must match FluxVectorTable).
        bit_width: Quantization bits per coordinate.
        history_ticks: How many ticks of history to keep per room.

    Example::

        mem = JepaGridMemory(dim=256, bit_width=4, history_ticks=10)
        mem.record(TemporalSlice(room_id=42, tick=100, vector=[...], activity=0.8))
        future = mem.predict(room_id=42, ticks_ahead=3)
        similar = mem.find_similar_trajectory(room_id=42, k=5)
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 4,
        history_ticks: int = 10,
    ) -> None:
        from swarm.vector_table import FluxVectorTable

        self.dim = dim
        self.bit_width = bit_width
        self.history_ticks = history_ticks

        # Two indices: one for raw state, one for temporal deltas
        self._state_table = FluxVectorTable(dim=dim, bit_width=bit_width)
        self._delta_table = FluxVectorTable(dim=dim, bit_width=bit_width)

        # Temporal history per room: list of (tick, vector, activity, chaos)
        self._history: dict[int, list[tuple[int, list[float], float, float]]] = {}

    def record(self, slice: TemporalSlice) -> None:
        """Record a room state slice.

        Automatically computes delta from previous tick and indexes both.
        """
        from swarm.vector_table import AgentVector

        # Encode as uint64: high 32 bits = room_id, low 32 bits = tick
        state_id = self._encode_id(slice.room_id, slice.tick)

        # Index raw state
        self._state_table.add(
            AgentVector(
                agent_id=state_id,
                vector=slice.vector,
                fitness=slice.activity,
                generation=slice.tick,
                extra={"room_id": slice.room_id, "chaos": slice.chaos},
            )
        )

        # Compute and index delta
        history = self._history.setdefault(slice.room_id, [])
        if history:
            prev_tick, prev_vec, _, _ = history[-1]
            delta = self._compute_delta(prev_vec, slice.vector)
            delta_id = self._encode_id(slice.room_id, slice.tick, delta=True)
            self._delta_table.add(
                AgentVector(
                    agent_id=delta_id,
                    vector=delta,
                    fitness=slice.activity,
                    generation=slice.tick,
                    extra={"room_id": slice.room_id, "prev_tick": prev_tick},
                )
            )

        # Update history
        history.append((slice.tick, slice.vector, slice.activity, slice.chaos))
        if len(history) > self.history_ticks:
            history.pop(0)

        logger.debug(
            "Recorded room %d tick %d (history depth=%d)",
            slice.room_id,
            slice.tick,
            len(history),
        )

    def predict(self, room_id: int, ticks_ahead: int = 1) -> list[float] | None:
        """Predict room state N ticks ahead.

        Uses simple linear extrapolation from the last 3 ticks' deltas.
        Returns None if insufficient history.
        """
        history = self._history.get(room_id, [])
        if len(history) < 2:
            return None

        # Compute average delta from last min(3, len-1) transitions
        deltas: list[list[float]] = []
        for i in range(max(0, len(history) - 3), len(history) - 1):
            _, v1, _, _ = history[i]
            _, v2, _, _ = history[i + 1]
            deltas.append(self._compute_delta(v1, v2))

        if not deltas:
            return None

        avg_delta = np.mean(deltas, axis=0).tolist()
        last_vec = history[-1][1]

        # Extrapolate
        predicted = [
            v + avg_delta[i] * ticks_ahead for i, v in enumerate(last_vec)
        ]
        return predicted

    def find_similar_trajectory(
        self,
        room_id: int,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find rooms with similar recent trajectories.

        Searches the delta table using the target room's most recent
        delta as query. Rooms moving in similar directions are matches.

        Returns:
            List of (room_id, score) sorted best-first.
        """
        history = self._history.get(room_id, [])
        if len(history) < 2:
            return []

        # Most recent delta as query
        _, v1, _, _ = history[-2]
        _, v2, _, _ = history[-1]
        delta = self._compute_delta(v1, v2)

        # Search delta table
        results = self._delta_table.search(delta, k=k * 3)

        # Group by room_id, take best score per room
        room_scores: dict[int, float] = {}
        for state_id, score, _meta in results:
            rid = self._decode_room_id(state_id)
            if rid == room_id:
                continue
            if rid not in room_scores or score > room_scores[rid]:
                room_scores[rid] = score

        sorted_rooms = sorted(room_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_rooms[:k]

    def get_state_at(self, room_id: int, tick: int) -> TemporalSlice | None:
        """Retrieve exact historical state."""
        state_id = self._encode_id(room_id, tick)
        if not self._state_table.contains(state_id):
            return None
        # Meta lookup would need AgentMeta exposure — placeholder
        return None

    def room_count(self) -> int:
        """Number of distinct rooms tracked."""
        return len(self._history)

    def __len__(self) -> int:
        return len(self._state_table)

    def __repr__(self) -> str:
        return (
            f"JepaGridMemory(dim={self.dim}, bit_width={self.bit_width}, "
            f"rooms={self.room_count()}, slices={len(self)})"
        )

    # ── internals ───────────────────────────────────────────

    @staticmethod
    def _encode_id(room_id: int, tick: int, delta: bool = False) -> int:
        """Pack room_id + tick into uint64.

        Layout:
            bit 63: delta flag
            bits 62-32: room_id (31 bits = ~2B rooms)
            bits 31-0: tick (32 bits = 4B ticks)
        """
        return (int(delta) << 63) | ((room_id & 0x7FFFFFFF) << 32) | (tick & 0xFFFFFFFF)

    @staticmethod
    def _decode_room_id(state_id: int) -> int:
        """Extract room_id from packed uint64."""
        return (state_id >> 32) & 0x7FFFFFFF

    @staticmethod
    def _compute_delta(a: list[float], b: list[float]) -> list[float]:
        """Element-wise difference."""
        return [b[i] - a[i] for i in range(len(a))]
