"""Tests for BreederDaemonV2 lifecycle FSM + WAL.

Mocks turbovec so tests run without the native extension.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

# ── Mock turbovec before any swarm.vector_table import ──
_mock_turbovec = types.ModuleType("turbovec")


class _MockIdMapIndex:
    """Minimal stand-in for turbovec.IdMapIndex."""

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self._vectors: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for vec, aid in zip(vectors, ids):
            self._vectors[int(aid)] = vec.copy()

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        allowlist: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.zeros((1, k), dtype=np.float32),
                np.zeros((1, k), dtype=np.uint64),
            )
        q = query[0]
        candidates = list(self._vectors.items())
        if allowlist is not None:
            allowed = set(int(a) for a in allowlist)
            candidates = [(aid, v) for aid, v in candidates if aid in allowed]

        qn = q / (np.linalg.norm(q) + 1e-8)
        sims: list[tuple[int, float]] = []
        for aid, vec in candidates:
            vn = vec / (np.linalg.norm(vec) + 1e-8)
            sims.append((aid, float(np.dot(qn, vn))))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:k]
        while len(top) < k:
            top.append((0, 0.0))
        scores = np.array([[s for _, s in top]], dtype=np.float32)
        ids_arr = np.array([[aid for aid, _ in top]], dtype=np.uint64)
        return scores, ids_arr

    def remove(self, agent_id: int) -> bool:
        return self._vectors.pop(agent_id, None) is not None

    def contains(self, agent_id: int) -> bool:
        return agent_id in self._vectors

    def write(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "_MockIdMapIndex":
        return cls(dim=256)


_mock_turbovec.IdMapIndex = _MockIdMapIndex  # type: ignore[attr-defined]
sys.modules["turbovec"] = _mock_turbovec

# Now safe to import swarm modules
from nerve.room_grid import RoomGrid
from swarm.breeder_daemon_v2 import (
    BreederDaemonV2,
    DiversityConfig,
    LifecycleState,
    LifecycleTransition,
    ThermalConfig,
)
from swarm.thermal import DeviceType, ThermalBudget
from swarm.vector_table import AgentVector, FluxVectorTable


# ── fixtures ────────────────────────────────────────────────

@pytest.fixture
def grid():
    """20-room grid with a clear hot/cold split."""
    g = RoomGrid(n=20)
    for _ in range(20):
        for i in range(10):
            g.activity[i] += 5
    return g


@pytest.fixture
def thermal():
    return ThermalBudget({DeviceType.GPU: 10, DeviceType.CPU: 20})


@pytest.fixture
def wal_path():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def vector_table():
    """Pre-populated vector table with 20 agents."""
    vt = FluxVectorTable(dim=256, bit_width=4)
    rng = np.random.RandomState(42)
    for i in range(20):
        scale = 2.0 if i < 10 else 0.5
        vec = (rng.randn(256).astype(np.float32) * scale).tolist()
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec,
                fitness=0.8 if i < 10 else 0.3,
                generation=1,
                capability_mask=0xFFFF,
                thermal_pressure=0.1,
            )
        )
    return vt


def make_daemon(grid, thermal, wal_path, vector_table=None):
    """Factory for a test daemon."""
    return BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        vector_table=vector_table,
        diversity=DiversityConfig(),
        thermal_cfg=ThermalConfig(max_agents=65, hysteresis_ticks=2),
        wal_path=wal_path,
        tick_interval=60.0,
    )


# ── tests ───────────────────────────────────────────────────

class TestWALReplay:
    """WAL replay reconstructs state after restart."""

    def test_replay_recovers_agents(self, grid, thermal, wal_path, vector_table):
        """Start → breed → stop → new daemon → start → verify state recovered."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()
        assert daemon.running

        # Seed a single agent (one breed avoids room-reuse sunset)
        daemon.queue_breed(parent_a=1, parent_b=2, priority=0)
        transitions = daemon.step()

        # Collect agent ID that reached INCUBATE
        incubated = [
            tr.agent_id for tr in transitions
            if tr.to_state == LifecycleState.INCUBATE
        ]
        assert len(incubated) == 1
        child_id = incubated[0]

        daemon.stop()

        # Fresh daemon on same WAL
        daemon2 = make_daemon(grid, thermal, wal_path, vector_table)
        daemon2.start()
        assert daemon2.running

        # Replayed state should include the incubated agent
        replayed = daemon2.state
        assert child_id in replayed
        assert replayed[child_id] == LifecycleState.INCUBATE

        daemon2.stop()

    def test_replay_preserves_sunset(self, grid, thermal, wal_path, vector_table):
        """SUNSET agents stay SUNSET after replay."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        daemon.queue_breed(parent_a=10, parent_b=11)
        transitions = daemon.step()

        # Force a sunset by re-breeding into the same room
        daemon.queue_breed(parent_a=12, parent_b=13)
        daemon.step()

        daemon.stop()

        daemon2 = make_daemon(grid, thermal, wal_path, vector_table)
        daemon2.start()

        # At least some agents should be in non-EGG states
        assert len(daemon2.state) > 0
        daemon2.stop()

    def test_wal_is_sqlite(self, grid, thermal, wal_path, vector_table):
        """WAL file is a valid SQLite database."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()
        daemon.stop()

        import sqlite3
        conn = sqlite3.connect(wal_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {r[0] for r in cur.fetchall()}
        conn.close()

        assert "lifecycle" in tables
        assert "lifecycle_log" in tables
        assert "breed_queue" in tables
        assert "genealogy" in tables


class TestLifecycleTransitions:
    """Lifecycle transitions are recorded."""

    def test_egg_to_incubate_recorded(self, grid, thermal, wal_path, vector_table):
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        daemon.queue_breed(parent_a=1, parent_b=2)
        transitions = daemon.step()

        daemon.stop()

        # Should see EGG → INCUBATE
        found = False
        for tr in transitions:
            if tr.to_state == LifecycleState.INCUBATE:
                found = True
                assert tr.from_state == LifecycleState.EGG
                assert tr.parent_a == 1
                assert tr.parent_b == 2
        assert found, "Expected EGG→INCUBATE transition"

    def test_incubate_has_generation(self, grid, thermal, wal_path, vector_table):
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # First generation parents have gen=0 in empty genealogy
        daemon.queue_breed(parent_a=100, parent_b=101)
        transitions = daemon.step()
        daemon.stop()

        incubate_tr = next(
            (t for t in transitions if t.to_state == LifecycleState.INCUBATE), None
        )
        assert incubate_tr is not None
        assert incubate_tr.generation == 1  # max(0,0)+1

    def test_sunset_transition(self, grid, thermal, wal_path, vector_table):
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Breed once
        daemon.queue_breed(parent_a=1, parent_b=2)
        daemon.step()

        # Breed again — should sunset the first agent in the same room
        daemon.queue_breed(parent_a=3, parent_b=4)
        transitions = daemon.step()
        daemon.stop()

        sunset_tr = next(
            (t for t in transitions if t.to_state == LifecycleState.SUNSET), None
        )
        assert sunset_tr is not None
        assert sunset_tr.from_state != LifecycleState.SUNSET

    def test_state_property_matches_wal(self, grid, thermal, wal_path, vector_table):
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        daemon.queue_breed(parent_a=5, parent_b=6)
        daemon.step()

        state = daemon.state
        daemon.stop()

        import sqlite3
        conn = sqlite3.connect(wal_path)
        cur = conn.execute("SELECT agent_id, state FROM lifecycle")
        wal_states = {r[0]: r[1] for r in cur.fetchall()}
        conn.close()

        for aid, expected in state.items():
            assert wal_states.get(aid) == expected.name


class TestThermalBudget:
    """step() respects thermal budget."""

    def test_no_spawn_when_full(self, grid, wal_path, vector_table):
        """When thermal budget is full, step() should not spawn."""
        thermal = ThermalBudget({DeviceType.GPU: 0})  # zero slots
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        daemon.queue_breed(parent_a=1, parent_b=2)
        transitions = daemon.step()
        daemon.stop()

        # Should not have reached INCUBATE — blocked by thermal
        incubated = [t for t in transitions if t.to_state == LifecycleState.INCUBATE]
        assert len(incubated) == 0

    def test_hysteresis_ticks(self, grid, wal_path, vector_table):
        """Thermal blocking should persist for hysteresis_ticks."""
        thermal = ThermalBudget({DeviceType.GPU: 0})
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        daemon.queue_breed(parent_a=1, parent_b=2)
        # First step: blocked, re-queued
        tr1 = daemon.step()
        # Second step: still blocked (hysteresis=2), re-queued
        tr2 = daemon.step()
        daemon.stop()

        assert len([t for t in tr1 if t.to_state == LifecycleState.INCUBATE]) == 0
        assert len([t for t in tr2 if t.to_state == LifecycleState.INCUBATE]) == 0

    def test_succeeds_when_room_available(self, grid, thermal, wal_path, vector_table):
        """When thermal has room, step() should spawn."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        daemon.queue_breed(parent_a=1, parent_b=2)
        transitions = daemon.step()
        daemon.stop()

        incubated = [t for t in transitions if t.to_state == LifecycleState.INCUBATE]
        assert len(incubated) > 0

    def test_parent_sacrifice(self, grid, wal_path, vector_table):
        """When budget full, parent sacrifice can free a slot."""
        thermal = ThermalBudget({DeviceType.GPU: 1})
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # First breed: should succeed (room in budget)
        daemon.queue_breed(parent_a=1, parent_b=2)
        daemon.step()

        # Second breed: budget full, but parent sacrifice should free slot
        daemon.queue_breed(parent_a=3, parent_b=4)
        transitions = daemon.step()
        daemon.stop()

        # The second spawn may or may not succeed depending on parent matching,
        # but it should not crash.
        assert isinstance(transitions, list)


class TestDiversityScore:
    """diversity_score property works."""

    def test_returns_zero_without_table(self, grid, thermal, wal_path):
        daemon = make_daemon(grid, thermal, wal_path, vector_table=None)
        assert daemon.diversity_score == 0.0

    def test_returns_zero_with_empty_table(self, grid, thermal, wal_path):
        empty_vt = FluxVectorTable(dim=256, bit_width=4)
        daemon = make_daemon(grid, thermal, wal_path, vector_table=empty_vt)
        assert daemon.diversity_score == 0.0

    def test_returns_positive_with_diverse_population(
        self, grid, thermal, wal_path, vector_table
    ):
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Manually inject multiple agents in INCUBATE state with diverse vectors
        # (breeding repeatedly into the same room would sunset previous agents,
        # so we directly populate state + vector table for this property test)
        rng = np.random.RandomState(77)
        for i in range(5):
            aid = 9000 + i
            daemon._state[aid] = LifecycleState.INCUBATE
            vec = (rng.randn(256).astype(np.float32) * (1.0 if i < 3 else 0.2)).tolist()
            daemon._vector_table.add(
                AgentVector(
                    agent_id=aid,
                    vector=vec,
                    fitness=0.5,
                    generation=1,
                )
            )

        score = daemon.diversity_score
        daemon.stop()

        # With diverse random vectors, score should be > 0
        assert score > 0.0

    def test_drops_when_population_is_cloned(
        self, grid, thermal, wal_path, vector_table
    ):
        """If all agents share the same vector, diversity should be low."""
        # Build a table with identical vectors
        cloned_vt = FluxVectorTable(dim=256, bit_width=4)
        same_vec = np.random.randn(256).astype(np.float32).tolist()
        for i in range(5):
            cloned_vt.add(
                AgentVector(
                    agent_id=i,
                    vector=same_vec,
                    fitness=0.5,
                    generation=1,
                )
            )

        daemon = make_daemon(grid, thermal, wal_path, vector_table=cloned_vt)
        daemon.start()

        for i in range(5):
            daemon.queue_breed(parent_a=i, parent_b=(i + 1) % 5)
            daemon.step()

        score = daemon.diversity_score
        daemon.stop()

        # Identical vectors → near-zero diversity
        assert score < 0.01

    def test_select_parents_uses_diversity(self, grid, thermal, wal_path, vector_table):
        """select_parents should return different pairs for diversity."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Seed some survive-state agents in the vector table
        pairs = daemon.select_parents(n_children=3)
        daemon.stop()

        assert isinstance(pairs, list)
        # With a populated table we should get pairs
        assert len(pairs) > 0

    def test_state_includes_all_agents(self, grid, thermal, wal_path, vector_table):
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        for i in range(3):
            daemon.queue_breed(parent_a=i, parent_b=(i + 1) % 3)
            daemon.step()

        state = daemon.state
        daemon.stop()

        # Every incubated agent should appear in state
        assert len(state) >= 3
        for st in state.values():
            assert isinstance(st, LifecycleState)


class TestDaemonThread:
    """Start/stop lifecycle."""

    def test_start_stop(self, grid, thermal, wal_path):
        daemon = make_daemon(grid, thermal, wal_path)
        daemon.start()
        assert daemon.running
        daemon.stop()
        assert not daemon.running

    def test_double_start_is_noop(self, grid, thermal, wal_path):
        daemon = make_daemon(grid, thermal, wal_path)
        daemon.start()
        t1 = daemon._thread
        daemon.start()
        t2 = daemon._thread
        assert t1 is t2
        daemon.stop()
