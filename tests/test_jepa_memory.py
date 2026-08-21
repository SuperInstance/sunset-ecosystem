"""Tests for JepaGridMemory — temporal room-state storage and prediction.

Uses a mock FluxVectorTable to avoid turbovec dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.jepa_memory import JepaGridMemory, TemporalSlice


class FakeFluxTable:
    """Minimal mock for JepaGridMemory tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self._index: dict[int, None] = {}

    def add(self, av) -> None:
        self._index[av.agent_id] = None

    def search(self, query, k=10, **kwargs):
        return []

    def contains(self, agent_id: int) -> bool:
        return agent_id in self._index

    def __len__(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# TemporalSlice
# ---------------------------------------------------------------------------


class TestTemporalSlice:
    def test_creation(self):
        ts = TemporalSlice(
            room_id=1,
            tick=10,
            vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            activity=0.8,
            chaos=0.1,
            metadata={"foo": "bar"},
        )
        assert ts.room_id == 1
        assert ts.tick == 10
        assert len(ts.vector) == 8
        assert ts.activity == 0.8
        assert ts.chaos == 0.1

    def test_immutability(self):
        ts = TemporalSlice(
            room_id=1,
            tick=10,
            vector=[0.0] * 8,
            activity=0.0,
            chaos=0.0,
            metadata={},
        )
        with pytest.raises(AttributeError):
            ts.room_id = 2


# ---------------------------------------------------------------------------
# JepaGridMemory
# ---------------------------------------------------------------------------


class TestJepaGridMemory:
    def test_init(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        assert mem.dim == 8
        assert mem.bit_width == 2
        assert mem.history_ticks == 3
        assert mem.room_count() == 0

    def test_record(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        ts = TemporalSlice(
            room_id=1,
            tick=10,
            vector=[0.1] * 8,
            activity=0.8,
            chaos=0.1,
            metadata={},
        )
        mem.record(ts)
        assert mem.room_count() == 1
        assert len(mem) == 1

    def test_history_pruning(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        for t in range(5):
            mem.record(
                TemporalSlice(
                    room_id=1,
                    tick=t,
                    vector=[float(t)] * 8,
                    activity=0.5,
                    chaos=0.0,
                    metadata={},
                )
            )
        assert len(mem._history[1]) == 3

    def test_predict_no_history(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        assert mem.predict(room_id=1, ticks_ahead=1) is None

    def test_predict_one_tick(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        mem.record(
            TemporalSlice(
                room_id=1,
                tick=0,
                vector=[0.0] * 8,
                activity=0.5,
                chaos=0.0,
                metadata={},
            )
        )
        mem.record(
            TemporalSlice(
                room_id=1,
                tick=1,
                vector=[1.0] * 8,
                activity=0.5,
                chaos=0.0,
                metadata={},
            )
        )
        pred = mem.predict(room_id=1, ticks_ahead=1)
        assert pred is not None
        assert len(pred) == 8
        assert all(v == pytest.approx(2.0) for v in pred)

    def test_predict_multiple_ticks(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        mem.record(
            TemporalSlice(
                room_id=1,
                tick=0,
                vector=[0.0] * 8,
                activity=0.5,
                chaos=0.0,
                metadata={},
            )
        )
        mem.record(
            TemporalSlice(
                room_id=1,
                tick=1,
                vector=[1.0] * 8,
                activity=0.5,
                chaos=0.0,
                metadata={},
            )
        )
        pred = mem.predict(room_id=1, ticks_ahead=3)
        assert pred is not None
        assert all(v == pytest.approx(4.0) for v in pred)

    def test_find_similar_trajectory_no_history(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        assert mem.find_similar_trajectory(room_id=1, k=5) == []

    def test_find_similar_trajectory(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        for t in range(3):
            mem.record(
                TemporalSlice(
                    room_id=1,
                    tick=t,
                    vector=[float(t)] * 8,
                    activity=0.5,
                    chaos=0.0,
                    metadata={},
                )
            )
        for t in range(3):
            mem.record(
                TemporalSlice(
                    room_id=2,
                    tick=t,
                    vector=[0.0] * 8,
                    activity=0.5,
                    chaos=0.0,
                    metadata={},
                )
            )
        results = mem.find_similar_trajectory(room_id=1, k=5)
        assert isinstance(results, list)

    def test_get_state_at_not_found(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        assert mem.get_state_at(room_id=1, tick=0) is None

    def test_room_count(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        mem.record(
            TemporalSlice(
                room_id=1,
                tick=0,
                vector=[0.0] * 8,
                activity=0.5,
                chaos=0.0,
                metadata={},
            )
        )
        mem.record(
            TemporalSlice(
                room_id=2,
                tick=0,
                vector=[0.0] * 8,
                activity=0.5,
                chaos=0.0,
                metadata={},
            )
        )
        assert mem.room_count() == 2

    def test_len(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        assert len(mem) == 0
        mem.record(
            TemporalSlice(
                room_id=1,
                tick=0,
                vector=[0.0] * 8,
                activity=0.5,
                chaos=0.0,
                metadata={},
            )
        )
        assert len(mem) == 1

    def test_repr(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        r = repr(mem)
        assert "JepaGridMemory" in r
        assert "dim=8" in r

    def test_encode_decode(self):
        sid = JepaGridMemory._encode_id(room_id=42, tick=100)
        assert JepaGridMemory._decode_room_id(sid) == 42

    def test_encode_delta(self):
        sid = JepaGridMemory._encode_id(room_id=1, tick=5, delta=True)
        assert JepaGridMemory._decode_room_id(sid) == 1

    def test_compute_delta(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        d = JepaGridMemory._compute_delta(a, b)
        assert d == [3.0, 3.0, 3.0]

    def test_multiple_rooms_history(self):
        mem = JepaGridMemory(dim=8, bit_width=2, history_ticks=3)
        for r in range(3):
            for t in range(4):
                mem.record(
                    TemporalSlice(
                        room_id=r,
                        tick=t,
                        vector=[float(r + t)] * 8,
                        activity=0.5,
                        chaos=0.0,
                        metadata={},
                    )
                )
        for r in range(3):
            assert len(mem._history[r]) == 3
        assert mem.room_count() == 3
        assert len(mem) == 12
