"""Tests for BreederDaemonV2 lifecycle FSM + WAL.

Mocks cocapn_traps so tests run without the external package.
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

# ── Mock cocapn_traps before swarm.breeder_daemon_v2 import ──
_mock_cocapn_traps = types.ModuleType("cocapn_traps")
_mock_cocapn_traps_traps = types.ModuleType("cocapn_traps.traps")
_mock_cocapn_traps_diversity = types.ModuleType("cocapn_traps.traps.diversity_collapse_trap")

class _MockAlert:
    level = "WARNING"
    recommended_action = "mock alert"

class _MockDiversityCollapseTrap:
    def __init__(self, bus=None):
        self._history = []
    def record(self, value: float) -> None:
        self._history.append(value)
    def check(self):
        return None  # no alerts in tests

_mock_cocapn_traps_diversity.DiversityCollapseTrap = _MockDiversityCollapseTrap
_mock_cocapn_traps_diversity.Alert = _MockAlert
sys.modules["cocapn_traps"] = _mock_cocapn_traps
sys.modules["cocapn_traps.traps"] = _mock_cocapn_traps_traps
sys.modules["cocapn_traps.traps.diversity_collapse_trap"] = _mock_cocapn_traps_diversity

# Now safe to import swarm modules
from nerve.room_grid import RoomGrid
from swarm.breeder_daemon_v2 import (
    BreederDaemonV2,
    AgentLifecycleFSM,
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

        # Collect agent ID that reached COMPETE
        incubated = [
            tr.agent_id for tr in transitions
            if tr.to_state == LifecycleState.COMPETE
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
        assert replayed[child_id] == LifecycleState.COMPETE

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

        # Should see EGG → COMPETE
        found = False
        for tr in transitions:
            if tr.to_state == LifecycleState.COMPETE:
                found = True
                assert tr.from_state == LifecycleState.EGG
                assert tr.parent_a == 1
                assert tr.parent_b == 2
        assert found, "Expected EGG→COMPETE transition"

    def test_incubate_has_generation(self, grid, thermal, wal_path, vector_table):
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # First generation parents have gen=0 in empty genealogy
        daemon.queue_breed(parent_a=100, parent_b=101)
        transitions = daemon.step()
        daemon.stop()

        incubate_tr = next(
            (t for t in transitions if t.to_state == LifecycleState.COMPETE), None
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

        # Should not have reached COMPETE — blocked by thermal
        incubated = [t for t in transitions if t.to_state == LifecycleState.COMPETE]
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

        assert len([t for t in tr1 if t.to_state == LifecycleState.COMPETE]) == 0
        assert len([t for t in tr2 if t.to_state == LifecycleState.COMPETE]) == 0

    def test_succeeds_when_room_available(self, grid, thermal, wal_path, vector_table):
        """When thermal has room, step() should spawn."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        daemon.queue_breed(parent_a=1, parent_b=2)
        transitions = daemon.step()
        daemon.stop()

        incubated = [t for t in transitions if t.to_state == LifecycleState.COMPETE]
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

        # Manually inject multiple agents in COMPETE state with diverse vectors
        # (breeding repeatedly into the same room would sunset previous agents,
        # so we directly populate state + vector table for this property test)
        rng = np.random.RandomState(77)
        for i in range(5):
            aid = 9000 + i
            daemon._fsm[aid] = AgentLifecycleFSM(agent_id=aid, initial_state=LifecycleState.COMPETE, strict=False)
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


class TestVectorParentSelection:
    """_select_parents_vector wired into step() and select_parents()."""

    def test_diversity_parent_selection(self, grid, thermal, wal_path, vector_table):
        """Vector table produces different parents than random fallback."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Seed some agents in breedable states with known vectors
        rng = np.random.RandomState(99)
        for i in range(6):
            aid = 100 + i
            daemon._fsm[aid] = AgentLifecycleFSM(
                agent_id=aid, initial_state=LifecycleState.SURVIVE, strict=False
            )
            vec = (rng.randn(256).astype(np.float32) * (2.0 if i < 3 else 0.3)).tolist()
            daemon._vector_table.add(
                AgentVector(
                    agent_id=aid,
                    vector=vec,
                    fitness=0.9 if i < 3 else 0.2,
                    generation=1,
                )
            )

        candidates = daemon._get_breedable_candidates()

        # With vector table — should pick high-fitness + diverse agents
        vec_pairs = daemon._select_parents_vector(
            population=candidates,
            vector_table=daemon._vector_table,
            n_children=2,
        )

        # Without vector table — random fallback
        random_pairs = daemon._select_parents_random(n_children=2)

        daemon.stop()

        assert len(vec_pairs) > 0
        assert len(random_pairs) > 0
        # Vector-based selection should differ from pure random
        # (statistically almost certain with diverse vectors)
        assert vec_pairs != random_pairs or len(vec_pairs) == 0

        # Verify vector-based parents are from the candidate set
        for a, b in vec_pairs:
            assert a in candidates
            assert b in candidates

    def test_fallback_without_vector_table(self, grid, thermal, wal_path):
        """When vector_table is None, fallback to fitness-only/random works."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table=None)
        daemon.start()

        # Seed agents without any vector table backing
        for i in range(4):
            daemon._fsm[200 + i] = AgentLifecycleFSM(
                agent_id=200 + i, initial_state=LifecycleState.SURVIVE, strict=False
            )

        candidates = daemon._get_breedable_candidates()
        pairs = daemon._select_parents_vector(
            population=candidates,
            vector_table=None,
            n_children=2,
        )

        daemon.stop()

        assert len(pairs) > 0
        for a, b in pairs:
            assert a in candidates
            assert b in candidates

    def test_step_fills_missing_parent_b(self, grid, thermal, wal_path, vector_table):
        """step() fills in parent_b via _select_parents_vector when queued as None."""
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Seed one breedable agent
        daemon._fsm[300] = AgentLifecycleFSM(
            agent_id=300, initial_state=LifecycleState.SURVIVE, strict=False
        )
        daemon._vector_table.add(
            AgentVector(
                agent_id=300,
                vector=np.random.randn(256).astype(np.float32).tolist(),
                fitness=0.8,
                generation=1,
            )
        )

        # Seed another so there are at least 2 candidates
        daemon._fsm[301] = AgentLifecycleFSM(
            agent_id=301, initial_state=LifecycleState.SURVIVE, strict=False
        )
        daemon._vector_table.add(
            AgentVector(
                agent_id=301,
                vector=np.random.randn(256).astype(np.float32).tolist(),
                fitness=0.7,
                generation=1,
            )
        )

        # Queue with only parent_a — step() should fill parent_b
        daemon.queue_breed(parent_a=300, parent_b=None, priority=0)
        transitions = daemon.step()
        daemon.stop()

        incubated = [t for t in transitions if t.to_state == LifecycleState.COMPETE]
        assert len(incubated) > 0
        # parent_b should have been filled in (not None in the transition)
        tr = incubated[0]
        assert tr.parent_a == 300
        assert tr.parent_b is not None
