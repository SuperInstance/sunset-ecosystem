"""WorkerPool — manages breeding worker threads with lifecycle FSM integration.

Each worker represents an agent going through the BreederDaemonV2 lifecycle:
    EGG → COMPETE → SURVIVE → BREED → SUNSET → ARCHIVE

The pool is thermal-aware: it refuses to spawn workers when the thermal
budget is exhausted. Workers run in daemon threads and can be gracefully
terminated via ``kill_worker()``.

Usage::

    pool = WorkerPool(grid, thermal, max_workers=65)
    agent_id = pool.spawn_worker(config={"room_id": 5})
    # ... later:
    pool.kill_worker(agent_id)
    active = pool.list_active()
"""

from __future__ import annotations

__all__ = [
    "WorkerPool",
    "BreedingWorker",
    "WorkerConfig",
    "WorkerState",
]

import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

import numpy as np

from nerve.room_grid import RoomGrid
from swarm.breeder_daemon_v2 import LifecycleState
from swarm.thermal import DeviceType, ThermalBudget

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Runtime state of an individual breeding worker thread."""

    PENDING = auto()  # Spawn requested, waiting for thermal slot
    RUNNING = auto()  # Thread alive, worker loop executing
    PAUSED = auto()  # Thermally throttled, loop sleeping
    STOPPING = auto()  # Kill signal sent, awaiting join
    DEAD = auto()  # Thread joined, resources released


@dataclass
class WorkerConfig:
    """Configuration for a single breeding worker."""

    room_id: int
    agent_id: int | None = None
    generation: int = 0
    parent_a: int | None = None
    parent_b: int | None = None
    tick_interval: float = 1.0
    max_ticks: int = 1000
    capability_mask: int = 0xFFFF
    on_lifecycle_change: (
        Callable[[int, LifecycleState, LifecycleState], None] | None
    ) = None
    on_tick: Callable[[int, dict[str, Any]], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _WorkerRecord:
    """Internal book-keeping for one worker."""

    agent_id: int
    config: WorkerConfig
    thread: threading.Thread
    stop_event: threading.Event
    lifecycle: LifecycleState
    worker_state: WorkerState
    worker_ref: BreedingWorker | None = None
    tick_count: int = 0
    start_time: float = 0.0
    device: DeviceType = DeviceType.GPU
    worker: "BreedingWorker | None" = None


class BreedingWorker:
    """A single worker thread that simulates an agent lifecycle in a room.

    The worker transitions through lifecycle states as it runs:
        EGG      → at thread start, before room allocation
        INCUBATE → first tick after room is assigned
        COMPETE  → chaos decays, activity accumulates
        SURVIVE  → after threshold ticks with sustained activity
        BREED    → flagged ready for breeding (external daemon decides)
        SUNSET   → killed or replaced

    The worker is designed to be lightweight: it calls ``grid.tick()`` on
    its assigned room and reports state transitions. It does NOT perform
    the actual breeding — that is the daemon's responsibility.
    """

    def __init__(
        self,
        agent_id: int,
        config: WorkerConfig,
        grid: RoomGrid,
        thermal: ThermalBudget,
        stop_event: threading.Event,
    ) -> None:
        self.agent_id = agent_id
        self.config = config
        self.grid = grid
        self.thermal = thermal
        self._stop_event = stop_event
        self._lifecycle = LifecycleState.EGG
        self._tick_count = 0
        self._activity_window: list[int] = []
        self._lock = threading.Lock()

    @property
    def lifecycle(self) -> LifecycleState:
        with self._lock:
            return self._lifecycle

    @lifecycle.setter
    def lifecycle(self, value: LifecycleState) -> None:
        with self._lock:
            old = self._lifecycle
            self._lifecycle = value
        if old != value and self.config.on_lifecycle_change:
            try:
                self.config.on_lifecycle_change(self.agent_id, old, value)
            except Exception:
                logger.exception(
                    "Lifecycle callback failed for agent %d", self.agent_id
                )

    def run(self) -> None:
        """Main worker loop."""
        logger.debug(
            "Worker %d starting in room %d", self.agent_id, self.config.room_id
        )

        # EGG → COMPETE: signal we have a room
        self.lifecycle = LifecycleState.EGG

        # Small random offset to desynchronize workers
        time.sleep(random.random() * self.config.tick_interval)

        while (
            not self._stop_event.is_set() and self._tick_count < self.config.max_ticks
        ):
            self._tick()
            self._tick_count += 1

            # Lifecycle transitions based on tick count / activity
            if self._lifecycle == LifecycleState.EGG and self._tick_count >= 3:
                self.lifecycle = LifecycleState.COMPETE

            if self._lifecycle == LifecycleState.COMPETE:
                recent_activity = sum(self._activity_window[-5:])
                if recent_activity >= 3 and self._tick_count >= 10:
                    self.lifecycle = LifecycleState.SURVIVE

            # BREED state is externally triggered, not auto

            # Report tick
            if self.config.on_tick:
                try:
                    tick_info = {
                        "tick": self._tick_count,
                        "activity": int(self.grid.activity[self.config.room_id]),
                        "chaos": float(self.grid.chaos[self.config.room_id]),
                        "lifecycle": self._lifecycle.name,
                    }
                    self.config.on_tick(self.agent_id, tick_info)
                except Exception:
                    logger.exception("Tick callback failed for agent %d", self.agent_id)

            # Wait for next tick or stop signal
            self._stop_event.wait(self.config.tick_interval)

        # Loop exit → SUNSET (unless already set)
        if self._lifecycle != LifecycleState.SUNSET:
            self.lifecycle = LifecycleState.SUNSET

        logger.debug(
            "Worker %d exiting after %d ticks", self.agent_id, self._tick_count
        )

    def _tick(self) -> None:
        """Execute one work tick on the assigned room."""
        room_id = self.config.room_id
        # Signal the room — in a real deployment this would be grid.tick()
        # Here we just ensure the room has some activity perturbation
        signal = np.random.randn(64).astype(np.float32) * 0.1
        # We don't call grid.tick() directly to avoid thundering herd;
        # instead we just bump activity slightly to simulate work
        self.grid.activity[room_id] += np.random.randint(0, 2)
        self._activity_window.append(int(self.grid.activity[room_id]))


class WorkerPool:
    """Manages a pool of ``BreedingWorker`` threads.

    Thermal-aware: ``spawn_worker()`` refuses to allocate a new worker
    if the thermal budget has no headroom. The pool also releases
    thermal slots when workers are killed.

    Args:
        grid: RoomGrid used by all workers for room allocation.
        thermal: ThermalBudget for slot management.
        max_workers: Hard ceiling on concurrent workers (beyond thermal).
        default_device: Which device type to allocate slots on.
    """

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        max_workers: int = 65,
        default_device: DeviceType = DeviceType.GPU,
    ) -> None:
        self.grid = grid
        self.thermal = thermal
        self.max_workers = max_workers
        self.default_device = default_device

        self._workers: dict[int, _WorkerRecord] = {}
        self._lock = threading.Lock()
        self._agent_counter = 0

    # ── public API ──────────────────────────────────────────

    def spawn_worker(
        self,
        agent_id: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Launch a new breeding worker.

        Args:
            agent_id: Explicit agent ID. If None, a monotonic ID is assigned.
            config: Dict merged into ``WorkerConfig``. Must contain ``room_id``.

        Returns:
            The assigned agent ID.

        Raises:
            RuntimeError: If thermal budget is exhausted or max_workers reached.
            ValueError: If config does not contain ``room_id``.
        """
        config = config or {}
        if "room_id" not in config:
            raise ValueError("config must contain 'room_id'")

        with self._lock:
            if len(self._workers) >= self.max_workers:
                raise RuntimeError(
                    f"WorkerPool at capacity ({self.max_workers} workers)"
                )

            # Assign agent ID
            if agent_id is None:
                self._agent_counter += 1
                agent_id = self._agent_counter
            elif agent_id in self._workers:
                raise ValueError(f"Agent ID {agent_id} already active")

            # Thermal check before allocating
            if not self.thermal.can_spawn(self.default_device):
                # Try parent sacrifice if we know a parent
                parent_id = config.get("parent_a")
                if parent_id is not None:
                    parent_str = f"agent_{parent_id}"
                    ok = self.thermal.parent_sacrifice_before_spawn(
                        parent_id=parent_str,
                        child_device=self.default_device,
                    )
                    if not ok:
                        raise RuntimeError(
                            f"Thermal budget exhausted on {self.default_device.value}, "
                            "cannot spawn worker"
                        )
                else:
                    raise RuntimeError(
                        f"Thermal budget exhausted on {self.default_device.value}, "
                        "cannot spawn worker"
                    )

            # Allocate thermal slot
            agent_str = f"agent_{agent_id}"
            allocated = self.thermal.allocate(agent_str, self.default_device)
            if not allocated:
                raise RuntimeError(
                    f"Thermal allocation failed for {agent_str} despite can_spawn=True"
                )

            # Build worker config — inject pool-internal lifecycle tracking
            def _track_lifecycle(aid, old, new):
                with self._lock:
                    rec = self._workers.get(aid)
                    if rec is not None:
                        rec.lifecycle = new
                # Also call user-provided callback if any
                user_cb = config.get("on_lifecycle_change")
                if user_cb is not None:
                    try:
                        user_cb(aid, old, new)
                    except Exception:
                        logger.exception(
                            "User lifecycle callback failed for agent %d", aid
                        )

            wc = WorkerConfig(
                room_id=config["room_id"],
                agent_id=agent_id,
                generation=config.get("generation", 0),
                parent_a=config.get("parent_a"),
                parent_b=config.get("parent_b"),
                tick_interval=config.get("tick_interval", 1.0),
                max_ticks=config.get("max_ticks", 1000),
                capability_mask=config.get("capability_mask", 0xFFFF),
                on_lifecycle_change=_track_lifecycle,
                on_tick=config.get("on_tick"),
                extra=config.get("extra", {}),
            )

            # Create and start thread
            stop_event = threading.Event()
            worker = BreedingWorker(
                agent_id=agent_id,
                config=wc,
                grid=self.grid,
                thermal=self.thermal,
                stop_event=stop_event,
            )
            thread = threading.Thread(
                target=worker.run,
                name=f"breeding-worker-{agent_id}",
                daemon=True,
            )

            record = _WorkerRecord(
                agent_id=agent_id,
                config=wc,
                thread=thread,
                stop_event=stop_event,
                lifecycle=LifecycleState.EGG,
                worker_state=WorkerState.PENDING,
                device=self.default_device,
            )

            self._workers[agent_id] = record
            record.worker = worker
            thread.start()
            record.start_time = time.time()
            record.worker_state = WorkerState.RUNNING

            logger.info(
                "Spawned worker %d in room %d (total workers: %d/%d)",
                agent_id,
                wc.room_id,
                len(self._workers),
                self.max_workers,
            )
            return agent_id

    def kill_worker(self, agent_id: int, timeout: float = 5.0) -> bool:
        """Gracefully terminate a worker.

        Args:
            agent_id: The worker to kill.
            timeout: Seconds to wait for thread join.

        Returns:
            True if the worker was found and stopped, False otherwise.
        """
        with self._lock:
            record = self._workers.get(agent_id)
            if record is None:
                return False

            record.worker_state = WorkerState.STOPPING
            record.stop_event.set()
            record.lifecycle = LifecycleState.SUNSET

        # Join outside the lock to avoid blocking other operations
        record.thread.join(timeout=timeout)

        with self._lock:
            if record.thread.is_alive():
                logger.warning(
                    "Worker %d thread did not join within %fs", agent_id, timeout
                )
            else:
                record.worker_state = WorkerState.DEAD

            # Release thermal slot
            agent_str = f"agent_{agent_id}"
            self.thermal.release(agent_str)

            # Clean up from pool
            del self._workers[agent_id]

            logger.info("Killed worker %d (released thermal slot)", agent_id)
            return True

    def list_active(self) -> dict[int, dict[str, Any]]:
        """Return metadata for all active workers.

        Returns:
            Mapping ``agent_id → {state, lifecycle, room_id, ticks, uptime}``.
        """
        with self._lock:
            result: dict[int, dict[str, Any]] = {}
            for agent_id, record in self._workers.items():
                uptime = time.time() - record.start_time
                ticks = (
                    record.worker._tick_count if record.worker else record.tick_count
                )
                result[agent_id] = {
                    "worker_state": record.worker_state.name,
                    "lifecycle": (
                        record.worker.lifecycle.name
                        if record.worker
                        else record.lifecycle.name
                    ),
                    "room_id": record.config.room_id,
                    "ticks": ticks,
                    "uptime_sec": round(uptime, 2),
                }
            return result

    def get_worker_lifecycle(self, agent_id: int) -> LifecycleState | None:
        """Get the current lifecycle state of a specific worker."""
        with self._lock:
            record = self._workers.get(agent_id)
            return record.lifecycle if record else None

    def set_worker_lifecycle(self, agent_id: int, state: LifecycleState) -> bool:
        """Externally set a worker's lifecycle state (e.g. BREED flag).

        Returns True if the worker was found and updated.
        """
        with self._lock:
            record = self._workers.get(agent_id)
            if record is None:
                return False
            record.lifecycle = state
            # Do NOT write back to worker.lifecycle — the worker thread's
            # callback would deadlock trying to acquire self._lock again.
            return True

    def kill_all(self, timeout: float = 5.0) -> list[int]:
        """Kill all active workers. Returns list of killed agent IDs."""
        with self._lock:
            ids = list(self._workers.keys())
        killed: list[int] = []
        for aid in ids:
            if self.kill_worker(aid, timeout=timeout):
                killed.append(aid)
        return killed

    @property
    def count(self) -> int:
        """Number of currently active workers."""
        with self._lock:
            return len(self._workers)

    @property
    def at_capacity(self) -> bool:
        """True if the pool has reached max_workers."""
        with self._lock:
            return len(self._workers) >= self.max_workers

    @property
    def thermal_headroom(self) -> float:
        """Current thermal utilization [0, 1]."""
        return self.thermal.thermal_headroom()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"WorkerPool(workers={len(self._workers)}/{self.max_workers}, "
                f"thermal={self.thermal})"
            )
