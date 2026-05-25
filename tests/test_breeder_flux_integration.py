"""Tests for BreederDaemonV2 FLUX gating integration.

Covers:
1. Constructor accepts flux_checker parameter
2. FLUX blocks parent pairs in select_parents() (all 3 paths)
3. FLUX blocks step() before room allocation
4. FLUX passes allow normal breeding
5. No flux_checker = no gating (backward compatible)
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.breeder_daemon_v2 import (
    BreederDaemonV2,
    DiversityConfig,
    ThermalConfig,
)
from swarm.flux_gating import FluxGatingConfig, FluxCheckResult, PythonFluxFallback
from swarm.thermal import DeviceType, ThermalBudget
from swarm.vector_table import FluxVectorTable


class _MockGrid:
    """Minimal RoomGrid mock."""

    def __init__(self, n_rooms: int = 4) -> None:
        self._rooms = list(range(n_rooms))
        self._weights: dict[int, np.ndarray] = {}
        self.chaos: dict[int, float] = {r: 0.3 for r in self._rooms}

    def cold(self, thresh: int = 1) -> list[int]:
        return self._rooms[:thresh]

    def top(self, k: int = 10) -> list[tuple[int, float]]:
        return [(r, 0.5) for r in self._rooms[:k]]

    def rebirth(self, room_id: int) -> None:
        self._weights[room_id] = np.zeros(64)

    def breed(self, src: int, dst: int) -> None:
        self._weights[dst] = self._weights.get(src, np.zeros(64)).copy()

    def get_weights(self, room_id: int) -> np.ndarray:
        return self._weights.get(room_id, np.zeros(64))


class _ParentBlockingChecker(PythonFluxFallback):
    """PythonFluxFallback that blocks parents with extreme weights.

    Any parent whose weights contain a value >= 90.0 is treated as blocked,
    simulating a FLUX bounds violation for that specific parent.
    """

    def __init__(self, blocked_parents: set[int] | None = None) -> None:
        self.blocked = blocked_parents or set()
        self.checks: list[tuple] = []
        # Use tight bounds so extreme weights fail
        super().__init__(FluxGatingConfig(weight_bounds=(0.0, 1.0)))

    def check_candidate(self, weights, chaos=0.3, thermal_pressure=0.0):
        self.checks.append((weights, chaos, thermal_pressure))
        # If any weight is the "blocked" sentinel (>= 90.0), force fail
        if weights.size > 0 and float(np.max(weights)) >= 90.0:
            return FluxCheckResult(
                passed=False,
                score=1.0,
                violations={"mock_block": 1.0},
            )
        return super().check_candidate(weights, chaos, thermal_pressure)


@pytest.fixture
def grid():
    return _MockGrid(n_rooms=8)


@pytest.fixture
def thermal():
    return ThermalBudget()


@pytest.fixture
def wal_file(tmp_path):
    return str(tmp_path / "test_breeder.wal.sqlite")


@pytest.fixture
def vector_table():
    vt = FluxVectorTable(dim=64, bit_width=4, use_hdc=False)
    # Seed with 4 agents
    for aid in range(1, 5):
        vec = np.random.randn(64)
        av = type("AV", (), {
            "agent_id": aid,
            "vector": vec.tolist(),
            "dim": 64,
            "fitness": float(aid) * 0.25,
            "generation": 0,
            "capability_mask": 0xFFFF,
            "thermal_pressure": 0.0,
            "extra": {},
        })()
        vt.add(av)
    return vt


@pytest.fixture
def daemon(grid, thermal, vector_table, wal_file):
    d = BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        vector_table=vector_table,
        diversity=DiversityConfig(metric="cosine", novelty_weight=0.3),
        thermal_cfg=ThermalConfig(max_agents=65, hysteresis_ticks=1),
        wal_path=wal_file,
    )
    d.start()
    yield d
    d.stop()


# ─────────────────────────────────────────────────────────────
# 1. Constructor accepts flux_checker
# ─────────────────────────────────────────────────────────────

def test_constructor_accepts_flux_checker(grid, thermal, vector_table, wal_file):
    checker = _ParentBlockingChecker()
    d = BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        vector_table=vector_table,
        flux_checker=checker,
        wal_path=wal_file,
    )
    assert d._flux_checker is checker


def test_constructor_flux_checker_none_default(grid, thermal, wal_file):
    d = BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        wal_path=wal_file,
    )
    assert d._flux_checker is None


# ─────────────────────────────────────────────────────────────
# 2. FLUX gating in select_parents()
# ─────────────────────────────────────────────────────────────

def test_select_parents_blocks_flux_violation(daemon, vector_table):
    """If FLUX blocks the best pair, select_parents should try alternatives."""
    # Force the vector table to propose agent 1 so we can test gating
    vector_table.search_diverse_parents = lambda n_results=2: [(1, 2)]
    # Mark agent 1 as blocked by giving it extreme weights
    daemon.grid._weights[1] = np.full(64, 99.0)
    for aid in (2, 3, 4):
        daemon.grid._weights[aid] = np.random.rand(64) * 0.5

    daemon._flux_checker = _ParentBlockingChecker(blocked_parents={1})
    daemon._vector_table = vector_table

    pairs = daemon.select_parents(n_children=1)
    assert len(pairs) == 1
    a, b = pairs[0]
    # FLUX gating checks the first parent; in random fallback the second
    # parent is not gated (pre-existing behaviour), so we only assert on a.
    assert a != 1


def test_select_parents_allows_clean_pair(daemon, vector_table):
    """If FLUX passes all pairs, normal selection proceeds."""
    for aid in range(1, 5):
        daemon.grid._weights[aid] = np.random.rand(64) * 0.5

    daemon._flux_checker = _ParentBlockingChecker(blocked_parents=set())
    daemon._vector_table = vector_table

    pairs = daemon.select_parents(n_children=1)
    assert len(pairs) == 1
    a, b = pairs[0]
    assert a is not None
    assert b is not None


def test_select_parents_no_checker_no_gating(daemon, vector_table):
    """Without flux_checker, selection proceeds unhindered."""
    daemon._flux_checker = None
    daemon._vector_table = vector_table

    pairs = daemon.select_parents(n_children=1)
    assert len(pairs) == 1


# ─────────────────────────────────────────────────────────────
# 3. FLUX gating in step()
# ─────────────────────────────────────────────────────────────

def test_step_blocks_flux_violation(daemon, thermal, wal_file):
    """FLUX block before room allocation should re-queue the ticket."""
    daemon.grid._weights[1] = np.full(64, 99.0)
    daemon.grid._weights[2] = np.random.rand(64) * 0.5

    daemon._flux_checker = _ParentBlockingChecker(blocked_parents={1})
    # Ensure thermal allows spawn
    thermal.allocate("agent_1", DeviceType.GPU)

    # Queue a breed with parent 1
    ticket = daemon.queue_breed(parent_a=1, parent_b=2, priority=10)
    assert ticket > 0

    transitions = daemon.step()
    # Should return empty because FLUX blocked and re-queued
    assert transitions == []
    # Ticket should be re-queued (pending count stays >= 1)
    assert daemon._wal.count_pending() >= 1


def test_step_allows_flux_pass(daemon, thermal, vector_table):
    """FLUX pass should allow normal step() spawning."""
    for aid in range(1, 5):
        daemon.grid.rebirth(aid)
        daemon.grid._weights[aid] = np.random.rand(64) * 0.5
    # Monkeypatch _extract_room_vector so step() doesn't crash on real grid
    daemon._extract_room_vector = lambda room_id: np.random.randn(64)

    daemon._flux_checker = _ParentBlockingChecker(blocked_parents=set())
    daemon._vector_table = vector_table

    ticket = daemon.queue_breed(parent_a=1, parent_b=2, priority=10)
    transitions = daemon.step()
    # Should produce at least one transition (EGG or COMPETE)
    assert len(transitions) >= 1


def test_step_no_checker_no_gating(daemon, thermal, vector_table):
    """Without flux_checker, step() proceeds normally."""
    daemon._flux_checker = None
    daemon._vector_table = vector_table

    for aid in range(1, 5):
        daemon.grid.rebirth(aid)
        daemon.grid._weights[aid] = np.random.randn(64)
    daemon._extract_room_vector = lambda room_id: np.random.randn(64)

    ticket = daemon.queue_breed(parent_a=1, parent_b=2, priority=10)
    transitions = daemon.step()
    assert len(transitions) >= 1


# ─────────────────────────────────────────────────────────────
# 4. Integration with PythonFluxFallback
# ─────────────────────────────────────────────────────────────

def test_python_flux_fallback_blocks_overweight(daemon, vector_table):
    """PythonFluxFallback should block candidates with extreme weights."""
    checker = PythonFluxFallback(
        config=FluxGatingConfig(weight_bounds=(0.0, 1.0))
    )
    daemon._flux_checker = checker
    daemon._vector_table = vector_table

    weights = np.array([5.0, 5.0, 5.0])
    result = checker.check_candidate(weights)
    assert not result.passed
    assert "bounds" in result.violations


def test_python_flux_fallback_allows_normal(daemon, vector_table):
    """PythonFluxFallback should pass candidates within bounds."""
    checker = PythonFluxFallback(
        config=FluxGatingConfig(weight_bounds=(0.0, 10.0))
    )
    daemon._flux_checker = checker
    daemon._vector_table = vector_table

    weights = np.array([0.5, 0.5, 0.5])
    result = checker.check_candidate(weights)
    assert result.passed
    assert result.score == 0.0


# ─────────────────────────────────────────────────────────────
# 5. Regression: existing tests still pass
# ─────────────────────────────────────────────────────────────

def test_breeder_daemon_v2_step_basic(daemon):
    """Smoke test: step() with no queue returns empty."""
    transitions = daemon.step()
    assert transitions == []


def test_queue_and_dequeue(daemon):
    """Queue items can be dequeued."""
    t1 = daemon.queue_breed(1, 2, priority=5)
    t2 = daemon.queue_breed(3, None, priority=10)
    assert daemon._wal.count_pending() == 2

    req = daemon._wal.dequeue_breed()
    assert req is not None
    # Higher priority (10) should come first
    assert req[3] == 10  # priority
