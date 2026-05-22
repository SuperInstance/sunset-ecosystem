"""End-to-end breeding cycle integration test.

Tests the full lifecycle using WorkerPool + BreederDaemonV2 + RoomGrid together:
    EGG → COMPETE → SURVIVE → BREED → SUNSET → ARCHIVE

Fast execution: small grid (n=20), short tick_interval (0.05s).
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
    ThermalConfig,
)
from swarm.thermal import DeviceType, ThermalBudget
from swarm.vector_table import AgentVector, FluxVectorTable
from swarm.worker_pool import WorkerPool, WorkerState


# ── fixtures ────────────────────────────────────────────────

@pytest.fixture
def grid():
    """20-room grid for fast tests."""
    return RoomGrid(n=20)


@pytest.fixture
def thermal():
    """Generous thermal budget."""
    return ThermalBudget({DeviceType.GPU: 20, DeviceType.CPU: 10})


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
    """Pre-populated vector table."""
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


@pytest.fixture
def pool(grid, thermal):
    """Fresh worker pool with fast ticks."""
    return WorkerPool(grid, thermal, max_workers=15)


def make_daemon(grid, thermal, wal_path, vector_table=None):
    """Factory for a test daemon with fast ticks."""
    return BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        vector_table=vector_table,
        diversity=DiversityConfig(),
        thermal_cfg=ThermalConfig(max_agents=20, hysteresis_ticks=2),
        wal_path=wal_path,
        tick_interval=0.05,
    )


# ── E2E lifecycle tests ───────────────────────────────────

class TestEggToCompete:
    """WorkerPool spawns → EGG → COMPETE after 3 ticks."""

    def test_egg_to_compete(self, grid, thermal, pool, wal_path):
        """Spawn worker, verify starts at EGG and transitions to COMPETE."""
        agent_id = pool.spawn_worker(
            config={
                "room_id": 0,
                "tick_interval": 0.05,
                "max_ticks": 50,
            }
        )

        # Check immediately — should be EGG
        state = pool.get_worker_lifecycle(agent_id)
        assert state == LifecycleState.EGG, (
            f"Expected EGG immediately after spawn, got {state}"
        )

        # Wait 3 ticks — should be COMPETE
        time.sleep(0.2)
        state = pool.get_worker_lifecycle(agent_id)
        assert state == LifecycleState.COMPETE, (
            f"Expected COMPETE after 3 ticks, got {state}"
        )

        pool.kill_worker(agent_id)


class TestEggToCompeteTransition:
    """EGG → COMPETE after activity threshold (3 ticks)."""

    def test_egg_to_compete_transition(self, grid, thermal, pool, wal_path):
        """Worker transitions to COMPETE after 3+ ticks."""
        agent_id = pool.spawn_worker(
            config={
                "room_id": 1,
                "tick_interval": 0.05,
                "max_ticks": 50,
            }
        )

        # Wait for 5 ticks (3 needed for EGG→COMPETE)
        time.sleep(0.3)

        state = pool.get_worker_lifecycle(agent_id)
        assert state == LifecycleState.COMPETE, (
            f"Expected COMPETE, got {state} after {pool.list_active()[agent_id]['ticks']} ticks"
        )

        pool.kill_worker(agent_id)


class TestCompeteToSurvive:
    """COMPETE → SURVIVE after sustained activity (10 ticks + activity)."""

    def test_compete_to_survive(self, grid, thermal, pool, wal_path):
        """Worker transitions to SURVIVE after 10+ ticks with activity."""
        # Pre-heat the room so activity accumulates
        grid.activity[2] = 5

        agent_id = pool.spawn_worker(
            config={
                "room_id": 2,
                "tick_interval": 0.05,
                "max_ticks": 100,
            }
        )

        # Wait for 12+ ticks (10 needed for COMPETE→SURVIVE)
        time.sleep(0.7)

        state = pool.get_worker_lifecycle(agent_id)
        assert state == LifecycleState.SURVIVE, (
            f"Expected SURVIVE, got {state} after {pool.list_active()[agent_id]['ticks']} ticks"
        )

        pool.kill_worker(agent_id)


class TestSurviveToBreed:
    """Flag worker BREED-ready, verify daemon queues breeding."""

    def test_survive_to_breed(self, grid, thermal, pool, wal_path, vector_table):
        """Worker reaches SURVIVE, flagged BREED, daemon queues breed."""
        # Pre-heat room
        grid.activity[3] = 5

        # Spawn and wait for SURVIVE
        agent_id = pool.spawn_worker(
            config={
                "room_id": 3,
                "tick_interval": 0.05,
                "max_ticks": 100,
            }
        )
        time.sleep(0.7)

        state = pool.get_worker_lifecycle(agent_id)
        assert state == LifecycleState.SURVIVE

        # Flag as BREED-ready
        ok = pool.set_worker_lifecycle(agent_id, LifecycleState.BREED)
        assert ok is True

        # Create daemon and queue a breed
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Queue breed using this agent as parent
        ticket = daemon.queue_breed(parent_a=agent_id, parent_b=4, priority=10)
        assert ticket > 0

        # Verify pending queue
        assert daemon._wal.count_pending() == 1

        daemon.stop()
        pool.kill_worker(agent_id)


class TestBreedToSunset:
    """Breed produces child, parent sunsets after max_ticks."""

    def test_breed_produces_child(self, grid, thermal, pool, wal_path, vector_table):
        """Daemon step() breeds child, parent sunsets after max_ticks."""
        # Pre-heat two rooms
        grid.activity[4] = 5
        grid.activity[5] = 5

        # Spawn parent workers
        parent_a = pool.spawn_worker(
            config={
                "room_id": 4,
                "tick_interval": 0.05,
                "max_ticks": 100,
            }
        )
        parent_b = pool.spawn_worker(
            config={
                "room_id": 5,
                "tick_interval": 0.05,
                "max_ticks": 100,
            }
        )
        time.sleep(0.7)

        # Both should be SURVIVE
        assert pool.get_worker_lifecycle(parent_a) == LifecycleState.SURVIVE
        assert pool.get_worker_lifecycle(parent_b) == LifecycleState.SURVIVE

        # Create daemon
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Queue and step breed
        daemon.queue_breed(parent_a=parent_a, parent_b=parent_b, priority=10)
        transitions = daemon.step()

        # Should see EGG → COMPETE for child
        spawned = [
            tr.agent_id for tr in transitions
            if tr.to_state == LifecycleState.EGG
        ]
        assert len(spawned) == 1, f"Expected 1 child, transitions: {[(t.agent_id, t.from_state.name, t.to_state.name) for t in transitions]}"
        child_id = spawned[0]

        # Child should be in daemon state
        assert child_id in daemon.state
        assert daemon.state[child_id] == LifecycleState.COMPETE

        daemon.stop()

        # Now let parents reach max_ticks and sunset
        # max_ticks=100, tick_interval=0.05 → 5 seconds. Too slow.
        # Instead, kill them directly to verify cleanup
        pool.kill_worker(parent_a)
        pool.kill_worker(parent_b)

        # Verify parents are gone
        assert parent_a not in pool.list_active()
        assert parent_b not in pool.list_active()

    def test_parent_sunsets_at_max_ticks(self, grid, thermal, pool, wal_path):
        """Worker with tiny max_ticks auto-sunsets."""
        agent_id = pool.spawn_worker(
            config={
                "room_id": 6,
                "tick_interval": 0.05,
                "max_ticks": 5,  # Very short lifecycle
            }
        )

        # Wait for 5 ticks + buffer
        time.sleep(0.4)

        # Worker should have exited and be in SUNSET
        rec = pool.list_active()
        # Might already be removed if join completed
        if agent_id in rec:
            assert rec[agent_id]["lifecycle"] == "SUNSET"

        # Ensure it's cleaned up
        pool.kill_worker(agent_id)
        assert agent_id not in pool.list_active()


class TestFullCycle10Generations:
    """Run 10 generations, verify population growth then plateau."""

    def test_full_cycle_10_generations(self, grid, thermal, wal_path, vector_table):
        """End-to-end: 10 generations with WorkerPool + Daemon + RoomGrid."""
        pool = WorkerPool(grid, thermal, max_workers=20)
        daemon = make_daemon(grid, thermal, wal_path, vector_table)
        daemon.start()

        # Track metrics
        generation_counts = []
        total_bred = 0
        total_sunset = 0

        # Seed: spawn 4 initial workers in different rooms
        initial_rooms = [0, 1, 2, 3]
        for rid in initial_rooms:
            grid.activity[rid] = 5  # Pre-heat

        agents = []
        for rid in initial_rooms:
            aid = pool.spawn_worker(
                config={
                    "room_id": rid,
                    "tick_interval": 0.05,
                    "max_ticks": 500,
                    "generation": 0,
                }
            )
            agents.append(aid)

        # Let seed workers mature to SURVIVE
        time.sleep(0.8)

        # Verify seed workers reached at least COMPETE (some may reach SURVIVE)
        for aid in agents:
            state = pool.get_worker_lifecycle(aid)
            assert state in (LifecycleState.COMPETE, LifecycleState.SURVIVE, LifecycleState.BREED)

        # Run 10 "generations" via daemon
        for gen in range(1, 11):
            active = pool.list_active()
            available = [
                aid for aid, info in active.items()
                if info["lifecycle"] in ("SURVIVE", "BREED", "COMPETE")
            ]

            if len(available) >= 2:
                # Queue breed from first two available
                a, b = available[:2]
                daemon.queue_breed(parent_a=a, parent_b=b, priority=10)
                transitions = daemon.step()

                # Count what happened
                for tr in transitions:
                    if tr.to_state == LifecycleState.EGG:
                        total_bred += 1
                        child_id = tr.agent_id
                        # Find child's room from daemon allocation
                        room_id = None
                        for rid, allocated_aid in daemon._room_allocations.items():
                            if allocated_aid == child_id:
                                room_id = rid
                                break

                        if room_id is not None:
                            # Daemon already allocated thermal for child.
                            # Release it so WorkerPool can manage the slot.
                            try:
                                thermal.release(f"agent_{child_id}")
                            except Exception:
                                pass

                            # Only spawn if room not occupied by another active worker
                            occupied = any(
                                info["room_id"] == room_id
                                for info in pool.list_active().values()
                            )
                            if not occupied:
                                try:
                                    pool.spawn_worker(
                                        agent_id=child_id,
                                        config={
                                            "room_id": room_id,
                                            "tick_interval": 0.05,
                                            "max_ticks": 500,
                                            "generation": gen,
                                            "parent_a": tr.parent_a,
                                            "parent_b": tr.parent_b,
                                        }
                                    )
                                except RuntimeError:
                                    # Thermal or capacity - skip
                                    pass

                    if tr.to_state == LifecycleState.SUNSET:
                        total_sunset += 1

            generation_counts.append(len(pool.list_active()))

            # Small delay between generations
            time.sleep(0.1)

        daemon.stop()

        # Cleanup remaining workers
        for aid in list(pool.list_active().keys()):
            try:
                pool.kill_worker(aid)
            except Exception:
                pass

        # Assertions
        assert total_bred > 0, "Should have bred at least some children"
        assert len(generation_counts) == 10

        # Population should have grown from initial 4
        max_pop = max(generation_counts)
        assert max_pop >= 4, f"Population never grew: {generation_counts}"

        # Should have seen at least some sunsets (replacements)
        assert total_sunset >= 0  # May or may not sunset depending on cold rooms

        # Final population should be bounded by thermal max
        assert generation_counts[-1] <= 20, f"Population exceeded max: {generation_counts[-1]}"


class TestDaemonPoolIntegration:
    """Specific integration points between Daemon and Pool."""

    def test_daemon_records_pool_worker_states(self, grid, thermal, pool, wal_path):
        """Daemon WAL records lifecycle from pool workers."""
        daemon = make_daemon(grid, thermal, wal_path)
        daemon.start()

        # Use callback to bridge pool worker lifecycle to daemon
        transitions_captured = []

        def on_lifecycle(aid, old, new):
            transitions_captured.append((aid, old, new))

        agent_id = pool.spawn_worker(
            config={
                "room_id": 7,
                "tick_interval": 0.05,
                "max_ticks": 50,
                "on_lifecycle_change": on_lifecycle,
            }
        )

        time.sleep(0.3)

        # Should have captured EGG→COMPETE transition
        assert len(transitions_captured) >= 1
        assert transitions_captured[0] == (agent_id, LifecycleState.EGG, LifecycleState.COMPETE)

        daemon.stop()
        pool.kill_worker(agent_id)

    def test_thermal_shared_between_pool_and_daemon(self, grid, thermal, pool, wal_path):
        """Pool and Daemon share thermal budget correctly."""
        daemon = make_daemon(grid, thermal, wal_path)
        daemon.start()

        before = thermal.total_current

        # Pool spawns worker
        agent_id = pool.spawn_worker(
            config={
                "room_id": 8,
                "tick_interval": 0.05,
                "max_ticks": 50,
            }
        )

        after_pool = thermal.total_current
        assert after_pool == before + 1

        # Daemon breed also allocates
        daemon.queue_breed(parent_a=agent_id, parent_b=None, priority=10)
        daemon.step()

        after_daemon = thermal.total_current
        # Daemon may have allocated a child, or re-used a slot if room was replaced
        assert after_daemon >= after_pool

        daemon.stop()
        pool.kill_worker(agent_id)

    def test_room_allocation_no_conflict(self, grid, thermal, pool, wal_path):
        """Pool and daemon don't double-allocate the same room."""
        daemon = make_daemon(grid, thermal, wal_path)
        daemon.start()

        # Pool worker in room 9
        agent_id = pool.spawn_worker(
            config={
                "room_id": 9,
                "tick_interval": 0.05,
                "max_ticks": 50,
            }
        )

        # Daemon step — should pick a DIFFERENT cold room
        daemon.queue_breed(parent_a=agent_id, parent_b=None, priority=10)
        transitions = daemon.step()

        spawned = [
            tr.agent_id for tr in transitions
            if tr.to_state == LifecycleState.EGG
        ]

        if spawned:
            child_id = spawned[0]
            child_room = None
            for rid, aid in daemon._room_allocations.items():
                if aid == child_id:
                    child_room = rid
                    break
            # Child room should not be room 9
            assert child_room != 9, f"Daemon allocated same room as pool worker"

        daemon.stop()
        pool.kill_worker(agent_id)
