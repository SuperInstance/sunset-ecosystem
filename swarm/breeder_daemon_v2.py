"""BreederDaemonV2 — persistent, diversity-aware, thermal-scheduled breeding daemon.

Implements the lifecycle FSM from SPEC_BREEDER_DAEMON_V2.md:
    EGG → INCUBATE → COMPETE → SURVIVE → BREED → SUNSET

Every transition is logged to an append-only SQLite WAL. On restart,
the WAL is replayed to reconstruct agent states.

The daemon maintains a breeding queue. `step()` dequeues one request
per tick, checks thermal budget, and either spawns or waits. Parent
selection optimizes for Pareto novelty (fitness + distance from
population) when a FluxVectorTable is available.

Porting note: the existing AutoBreeder logic is wrapped as a
compatibility shim so legacy callers can migrate gradually.
"""

from __future__ import annotations

__all__ = [
    "BreederDaemonV2",
    "LifecycleState",
    "LifecycleTransition",
    "DiversityConfig",
    "ThermalConfig",
]

import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np

from nerve.room_grid import RoomGrid
from swarm.thermal import DeviceType, ThermalBudget
from swarm.tournament import AgentScore, TournamentRound, breed

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Explicit lifecycle states for every agent in the fleet."""

    EGG = auto()       # Vector exists in table, no room allocated
    INCUBATE = auto()  # Room allocated, chaos=0.3
    COMPETE = auto()   # Active, chaos decaying
    SURVIVE = auto()   # Pareto non-dominated, stable activity
    BREED = auto()     # Actively breeding
    SUNSET = auto()    # Retired, room freed


@dataclass(frozen=True)
class LifecycleTransition:
    """A single state transition recorded in the WAL."""

    agent_id: int
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: float
    generation: int = 0
    origin_node: str = "local"
    parent_a: int | None = None
    parent_b: int | None = None
    vector_hash: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state": self.to_state.name,
            "entered_at": self.timestamp,
            "generation": self.generation,
            "origin_node": self.origin_node,
            "parent_a": self.parent_a,
            "parent_b": self.parent_b,
            "vector_hash": self.vector_hash,
        }


@dataclass(frozen=True)
class DiversityConfig:
    """How aggressively to pursue genetic diversity."""

    metric: str = "cosine"          # "cosine" | "hamming" | "l2"
    min_pairwise_dist: float = 0.15  # if avg dist drops below, request mesh breed
    novelty_weight: float = 0.3      # novelty vs fitness in parent selection
    max_inbreeding_gen: int = 3      # reject parent pairs sharing grandparent


@dataclass(frozen=True)
class ThermalConfig:
    """Thermal-aware scheduling parameters."""

    max_agents: int = 65
    hysteresis_ticks: int = 10
    cooling_curve: str = "exponential"  # "exponential" | "linear" | "measured"
    predictive_spawn: bool = True       # predict next free slot from curve


class _WALSchema:
    """SQLite schema and helper for the breeder WAL."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS lifecycle (
        agent_id INTEGER PRIMARY KEY,
        state TEXT CHECK(state IN (
            'EGG','INCUBATE','COMPETE','SURVIVE','BREED','SUNSET'
        )),
        entered_at REAL,
        generation INTEGER DEFAULT 0,
        origin_node TEXT DEFAULT 'local',
        parent_a INTEGER,
        parent_b INTEGER,
        vector_hash TEXT
    );

    CREATE TABLE IF NOT EXISTS lifecycle_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id INTEGER NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        timestamp REAL NOT NULL,
        generation INTEGER,
        origin_node TEXT,
        parent_a INTEGER,
        parent_b INTEGER,
        vector_hash TEXT
    );

    CREATE TABLE IF NOT EXISTS breed_queue (
        ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_a INTEGER NOT NULL,
        parent_b INTEGER,
        priority INTEGER DEFAULT 0,
        remote INTEGER DEFAULT 0,
        enqueued_at REAL NOT NULL,
        processed INTEGER DEFAULT 0,
        result_agent_id INTEGER
    );

    CREATE TABLE IF NOT EXISTS genealogy (
        agent_id INTEGER PRIMARY KEY,
        parent_a INTEGER,
        parent_b INTEGER,
        generation INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_lifecycle_state ON lifecycle(state);
    CREATE INDEX IF NOT EXISTS idx_queue_pending ON breed_queue(processed);
    CREATE INDEX IF NOT EXISTS idx_log_agent ON lifecycle_log(agent_id);
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.executescript(self.SCHEMA)
        conn.commit()
        conn.close()

    def replay(self) -> dict[int, LifecycleState]:
        """Replay WAL: return current state of every known agent."""
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT agent_id, state FROM lifecycle ORDER BY entered_at"
        )
        states: dict[int, LifecycleState] = {}
        for row in cur:
            agent_id = row["agent_id"]
            state_name = row["state"]
            try:
                states[agent_id] = LifecycleState[state_name]
            except KeyError:
                logger.warning("Unknown state %r for agent %d", state_name, agent_id)
        conn.close()
        return states

    def transition(
        self,
        tr: LifecycleTransition,
    ) -> None:
        """Record a transition in WAL (upsert lifecycle + append log)."""
        conn = sqlite3.connect(self.path, check_same_thread=False)
        row = tr.to_row()
        conn.execute(
            """INSERT INTO lifecycle (agent_id, state, entered_at, generation,
                 origin_node, parent_a, parent_b, vector_hash)
               VALUES (:agent_id, :state, :entered_at, :generation,
                 :origin_node, :parent_a, :parent_b, :vector_hash)
               ON CONFLICT(agent_id) DO UPDATE SET
                 state=excluded.state,
                 entered_at=excluded.entered_at,
                 generation=excluded.generation,
                 origin_node=excluded.origin_node,
                 parent_a=excluded.parent_a,
                 parent_b=excluded.parent_b,
                 vector_hash=excluded.vector_hash""",
            row,
        )
        conn.execute(
            """INSERT INTO lifecycle_log
                 (agent_id, from_state, to_state, timestamp, generation,
                  origin_node, parent_a, parent_b, vector_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tr.agent_id,
                tr.from_state.name if tr.from_state else None,
                tr.to_state.name,
                tr.timestamp,
                tr.generation,
                tr.origin_node,
                tr.parent_a,
                tr.parent_b,
                tr.vector_hash,
            ),
        )
        # Also update genealogy tree
        if tr.parent_a is not None or tr.parent_b is not None:
            conn.execute(
                """INSERT INTO genealogy (agent_id, parent_a, parent_b, generation)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(agent_id) DO UPDATE SET
                     parent_a=excluded.parent_a,
                     parent_b=excluded.parent_b,
                     generation=excluded.generation""",
                (tr.agent_id, tr.parent_a, tr.parent_b, tr.generation),
            )
        conn.commit()
        conn.close()

    def enqueue_breed(
        self,
        parent_a: int,
        parent_b: int | None,
        priority: int = 0,
        remote: bool = False,
    ) -> int:
        """Add a breeding request to queue. Returns ticket ID."""
        conn = sqlite3.connect(self.path, check_same_thread=False)
        cur = conn.execute(
            """INSERT INTO breed_queue
                 (parent_a, parent_b, priority, remote, enqueued_at)
               VALUES (?, ?, ?, ?, ?)""",
            (parent_a, parent_b, priority, int(remote), time.time()),
        )
        ticket = cur.lastrowid
        conn.commit()
        conn.close()
        return ticket or 0

    def dequeue_breed(
        self,
    ) -> tuple[int, int, int | None, int, bool] | None:
        """Pop the highest-priority unprocessed breed request.

        Returns: (ticket_id, parent_a, parent_b, priority, remote)
        """
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """SELECT ticket_id, parent_a, parent_b, priority, remote
               FROM breed_queue
               WHERE processed = 0
               ORDER BY priority DESC, enqueued_at ASC
               LIMIT 1"""
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return None
        ticket = row["ticket_id"]
        conn.execute(
            "UPDATE breed_queue SET processed = 1 WHERE ticket_id = ?",
            (ticket,),
        )
        conn.commit()
        conn.close()
        return (
            ticket,
            row["parent_a"],
            row["parent_b"],
            row["priority"],
            bool(row["remote"]),
        )

    def count_pending(self) -> int:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        cur = conn.execute(
            "SELECT COUNT(*) FROM breed_queue WHERE processed = 0"
        )
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_genealogy(self, agent_id: int) -> dict[str, Any] | None:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM genealogy WHERE agent_id = ?", (agent_id,)
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return dict(row)

    def get_generations_back(self, agent_id: int, depth: int) -> set[int]:
        """Collect ancestor IDs up to *depth* generations."""
        ancestors: set[int] = set()
        frontier = {agent_id}
        for _ in range(depth):
            next_frontier: set[int] = set()
            for aid in frontier:
                if aid in ancestors:
                    continue
                ancestors.add(aid)
                g = self.get_genealogy(aid)
                if g:
                    if g.get("parent_a") is not None:
                        next_frontier.add(g["parent_a"])
                    if g.get("parent_b") is not None:
                        next_frontier.add(g["parent_b"])
            frontier = next_frontier
            if not frontier:
                break
        return ancestors

    def close(self) -> None:
        pass  # SQLite connections are per-operation; nothing to hold open


class BreederDaemonV2:
    """Persistent, diversity-aware, thermal-scheduled breeding daemon.

    Args:
        grid: RoomGrid for spawning / rebirth.
        thermal: ThermalBudget for slot management.
        vector_table: Optional FluxVectorTable for diversity search.
        diversity: DiversityConfig for parent selection.
        thermal_cfg: ThermalConfig for scheduling parameters.
        wal_path: Path to SQLite WAL file.
        mesh: Optional MeshNode for cross-instance breeding (future).
    """

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        vector_table: Optional["FluxVectorTable"] = None,
        diversity: DiversityConfig = DiversityConfig(),
        thermal_cfg: ThermalConfig = ThermalConfig(),
        wal_path: str = "breeder.wal.sqlite",
        mesh: Any = None,
        tick_interval: float = 1.0,
    ) -> None:
        self.grid = grid
        self.thermal = thermal
        self._vector_table = vector_table
        self._diversity = diversity
        self._thermal_cfg = thermal_cfg
        self._wal_path = wal_path
        self._mesh = mesh
        self._tick_interval = tick_interval

        self._wal = _WALSchema(wal_path)
        self._state: dict[int, LifecycleState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0
        self._room_allocations: dict[int, int] = {}  # room_id → agent_id
        self._transitions_log: list[LifecycleTransition] = []

        # Thermal hysteresis counter (how many ticks we've been blocked)
        self._thermal_blocked_ticks: int = 0

    # ── public API ──────────────────────────────────────────

    def start(self) -> None:
        """Replay WAL, start scheduler thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        # Replay WAL to reconstruct state
        self._state = self._wal.replay()
        logger.info(
            "BreederDaemonV2 replayed WAL: %d agents, %d pending queue items",
            len(self._state),
            self._wal.count_pending(),
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="breeder-daemon-v2",
            daemon=True,
        )
        self._thread.start()
        logger.info("BreederDaemonV2 started")

    def stop(self) -> None:
        """Flush WAL, stop scheduler thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("BreederDaemonV2 stopped")

    def queue_breed(
        self,
        parent_a: int,
        parent_b: int | None = None,
        priority: int = 0,
        remote: bool = False,
    ) -> int:
        """Add a breeding request to the scheduler queue.

        Returns: queue ticket ID.
        """
        ticket = self._wal.enqueue_breed(parent_a, parent_b, priority, remote)
        logger.debug(
            "Queued breed ticket %d (parents=%s, remote=%s)",
            ticket,
            (parent_a, parent_b),
            remote,
        )
        return ticket

    def select_parents(
        self,
        n_children: int = 1,
    ) -> list[tuple[int, int]]:
        """Diversity-aware parent selection.

        1. Build Pareto frontier from vector table (fitness vs novelty).
        2. For each child, pick parent_a = highest Pareto score.
        3. Search vector table for most *distant* compatible winner = parent_b.
        4. Reject pairs that violate inbreeding rules.
        """
        if self._vector_table is None or len(self._vector_table) == 0:
            # Legacy fallback: random tournament winners
            return self._select_parents_random(n_children)

        # Get all agents in SURVIVE or BREED state as candidates
        candidates = self._get_breedable_candidates()
        if len(candidates) < 2:
            return self._select_parents_random(n_children)

        # Compute Pareto-novelty scores
        scored = []
        for aid in candidates:
            meta = self._vector_table._meta.get(aid)
            if meta is None:
                continue
            fitness = meta.fitness
            novelty = self._avg_nearest_neighbor_distance(aid, candidates, k=3)
            score = fitness * (1 - self._diversity.novelty_weight) + novelty * self._diversity.novelty_weight
            scored.append((aid, score, fitness, novelty))

        if len(scored) < 2:
            return self._select_parents_random(n_children)

        scored.sort(key=lambda x: x[1], reverse=True)
        pairs: list[tuple[int, int]] = []

        for _ in range(n_children):
            parent_a = scored[0][0]  # highest Pareto score

            # Find most distant compatible candidate
            best_b: int | None = None
            best_dist = -1.0

            vec_a = self._get_vector(parent_a)
            if vec_a is not None:
                for aid, _, _, _ in scored[1:]:
                    if aid == parent_a:
                        continue
                    # Inbreeding guard (1-generation: no direct parent-child)
                    if self._is_inbred(parent_a, aid):
                        continue
                    vec_b = self._get_vector(aid)
                    if vec_b is not None:
                        dist = self._vector_distance(vec_a, vec_b)
                        if dist > best_dist:
                            best_dist = dist
                            best_b = aid

            if best_b is None:
                # Fallback to second-best scored
                best_b = scored[1][0] if len(scored) > 1 else scored[0][0]

            pairs.append((parent_a, best_b))

        return pairs

    def step(self) -> list[LifecycleTransition]:
        """Run one scheduler tick: dequeue, check thermal, spawn or wait.

        Returns: list of state transitions that occurred.
        """
        transitions: list[LifecycleTransition] = []

        with self._lock:
            self._tick_count += 1
            tick = self._tick_count

        # Dequeue highest-priority breed request
        request = self._wal.dequeue_breed()
        if request is None:
            return transitions

        ticket, parent_a, parent_b, priority, remote = request

        # Thermal check
        device = DeviceType.GPU  # default; could be configurable per agent
        if not self.thermal.can_spawn(device):
            # Hysteresis: wait before trying again
            self._thermal_blocked_ticks += 1
            if self._thermal_blocked_ticks < self._thermal_cfg.hysteresis_ticks:
                # Re-queue at same priority for later retry
                self._wal.enqueue_breed(parent_a, parent_b, priority, remote)
                logger.debug(
                    "Thermal block (tick %d, hysteresis %d/%d), re-queued ticket %d",
                    tick,
                    self._thermal_blocked_ticks,
                    self._thermal_cfg.hysteresis_ticks,
                    ticket,
                )
                return transitions
            else:
                # Hysteresis exhausted — try parent sacrifice
                parent_id = f"agent_{parent_a}"
                ok = self.thermal.parent_sacrifice_before_spawn(
                    parent_id=parent_id,
                    child_device=device,
                )
                if not ok:
                    # Still no room; keep re-queuing
                    self._wal.enqueue_breed(parent_a, parent_b, priority, remote)
                    logger.warning(
                        "Thermal saturated, re-queued ticket %d", ticket
                    )
                    return transitions

        # Thermal check passed (or sacrifice succeeded)
        self._thermal_blocked_ticks = 0

        # Find a cold room for the child
        cold_rooms = self.grid.cold(thresh=1)
        if not cold_rooms:
            # No cold rooms — park this request back in queue
            self._wal.enqueue_breed(parent_a, parent_b, priority, remote)
            return transitions

        room_id = cold_rooms[0]

        # Release old agent in this room if any
        old_agent_id = self._room_allocations.get(room_id)
        if old_agent_id is not None:
            self.thermal.release(f"agent_{old_agent_id}")
            del self._room_allocations[room_id]
            # Record SUNSET for old agent
            sunset_tr = LifecycleTransition(
                agent_id=old_agent_id,
                from_state=self._state.get(old_agent_id, LifecycleState.COMPETE),
                to_state=LifecycleState.SUNSET,
                timestamp=time.time(),
            )
            self._wal.transition(sunset_tr)
            self._state[old_agent_id] = LifecycleState.SUNSET
            transitions.append(sunset_tr)

        # Determine child agent ID
        child_id = self._next_agent_id()
        generation = 0
        if parent_b is not None:
            g_a = self._wal.get_genealogy(parent_a)
            g_b = self._wal.get_genealogy(parent_b)
            gen_a = g_a["generation"] if g_a else 0
            gen_b = g_b["generation"] if g_b else 0
            generation = max(gen_a, gen_b) + 1
        elif parent_a is not None:
            g_a = self._wal.get_genealogy(parent_a)
            generation = (g_a["generation"] if g_a else 0) + 1

        # Place child in EGG state
        egg_tr = LifecycleTransition(
            agent_id=child_id,
            from_state=LifecycleState.EGG,
            to_state=LifecycleState.EGG,
            timestamp=time.time(),
            generation=generation,
            parent_a=parent_a,
            parent_b=parent_b,
            origin_node="local" if not remote else "remote",
        )
        self._wal.transition(egg_tr)
        self._state[child_id] = LifecycleState.EGG
        transitions.append(egg_tr)

        # Allocate room → INCUBATE
        # Use grid.rebirth() to reset room, then clone parent weights
        parent_room = self._find_room_for_agent(parent_a)
        if parent_room is not None:
            self.grid.breed(parent_room, room_id)
        else:
            self.grid.rebirth(room_id)

        incubate_tr = LifecycleTransition(
            agent_id=child_id,
            from_state=LifecycleState.EGG,
            to_state=LifecycleState.INCUBATE,
            timestamp=time.time(),
            generation=generation,
            parent_a=parent_a,
            parent_b=parent_b,
            origin_node="local" if not remote else "remote",
        )
        self._wal.transition(incubate_tr)
        self._state[child_id] = LifecycleState.INCUBATE
        transitions.append(incubate_tr)

        # Allocate thermal budget
        child_agent_str = f"agent_{child_id}"
        self.thermal.allocate(child_agent_str, device)
        self._room_allocations[room_id] = child_id

        # Sync to vector table if available
        if self._vector_table is not None:
            from swarm.vector_table import AgentVector
            # Build a simple vector from room weights (flattened)
            vec = self._extract_room_vector(room_id)
            self._vector_table.add(
                AgentVector(
                    agent_id=child_id,
                    vector=vec.tolist(),
                    fitness=0.0,  # will be scored after first tick
                    generation=generation,
                    capability_mask=0xFFFF,
                    thermal_pressure=0.0,
                )
            )

        logger.info(
            "Step %d: spawned agent %d in room %d (parents=%s, gen=%d)",
            tick,
            child_id,
            room_id,
            (parent_a, parent_b),
            generation,
        )

        return transitions

    @property
    def state(self) -> dict[int, LifecycleState]:
        """Current lifecycle state of every known agent."""
        with self._lock:
            return dict(self._state)

    @property
    def diversity_score(self) -> float:
        """Average pairwise cosine distance in current population.

        Range [0, 2]. Below 0.15 triggers mesh breeding request.
        """
        if self._vector_table is None or len(self._vector_table) == 0:
            return 0.0

        # Collect vectors for all non-SUNSET agents
        agent_ids = [
            aid for aid, st in self._state.items()
            if st not in (LifecycleState.SUNSET, LifecycleState.EGG)
        ]
        if len(agent_ids) < 2:
            return 0.0

        vectors: list[np.ndarray] = []
        for aid in agent_ids:
            vec = self._get_vector(aid)
            if vec is not None:
                vectors.append(vec)

        if len(vectors) < 2:
            return 0.0

        # Average pairwise cosine distance
        total_dist = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total_dist += self._vector_distance(vectors[i], vectors[j])
                count += 1

        return total_dist / count if count > 0 else 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    @property
    def wal_path(self) -> str:
        return self._wal_path

    # ── compatibility shim: wrap AutoBreeder ────────────────

    def auto_breed(
        self,
        n_winners: int | None = None,
    ) -> list[tuple[int, str]]:
        """Legacy compatibility: run one breeding cycle via V2 machinery.

        Uses select_parents() + queue_breed() + step() to emulate
        the old AutoBreeder.auto_breed() behavior.
        """
        n_children = n_winners or 3
        pairs = self.select_parents(n_children)
        tickets: list[int] = []
        for a, b in pairs:
            t = self.queue_breed(parent_a=a, parent_b=b, priority=0)
            tickets.append(t)

        results: list[tuple[int, str]] = []
        for _ in tickets:
            transitions = self.step()
            for tr in transitions:
                if tr.to_state == LifecycleState.INCUBATE:
                    parent_str = f"agent_{tr.parent_a}" if tr.parent_a else "unknown"
                    # Find which room got this agent
                    room_id = None
                    for rid, aid in self._room_allocations.items():
                        if aid == tr.agent_id:
                            room_id = rid
                            break
                    if room_id is not None:
                        results.append((room_id, parent_str))

        return results

    def cycle(self, n_winners: int | None = None) -> list[tuple[int, str]]:
        """Alias for auto_breed()."""
        return self.auto_breed(n_winners=n_winners)

    # ── internals ─────────────────────────────────────────

    def _run_loop(self) -> None:
        """Background loop: step() every tick_interval seconds."""
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception:
                logger.exception("BreederDaemonV2 step failed")
            self._stop_event.wait(self._tick_interval)

    def _next_agent_id(self) -> int:
        """Generate a fresh agent ID (signed 63-bit hash of uuid)."""
        raw = uuid.uuid4().hex
        digest = hashlib.blake2b(raw.encode(), digest_size=8).digest()
        val = int.from_bytes(digest, "big")
        # SQLite INTEGER is signed 64-bit; keep us in safe range
        return val % (2 ** 63 - 1)

    def _get_breedable_candidates(self) -> list[int]:
        """Return agent IDs that are eligible for breeding."""
        return [
            aid for aid, st in self._state.items()
            if st in (LifecycleState.SURVIVE, LifecycleState.BREED)
        ]

    def _get_vector(self, agent_id: int) -> np.ndarray | None:
        """Fetch a float32 vector for an agent from the table."""
        if self._vector_table is None:
            return None
        meta = self._vector_table._meta.get(agent_id)
        if meta is None:
            return None
        # Need to retrieve the actual vector from turbovec
        # Since IdMapIndex doesn't expose direct vector access, we search
        # with a dummy query and match by ID, or use the internal storage.
        # For the mock, _vectors exists. For real turbovec, we do a search.
        if hasattr(self._vector_table._index, "_vectors"):
            vec = self._vector_table._index._vectors.get(agent_id)
            if vec is not None:
                return np.array(vec, dtype=np.float32)
        # Fallback: search with zero vector, filter by ID (expensive but correct)
        try:
            dummy = [0.0] * self._vector_table.dim
            results = self._vector_table.search(dummy, k=len(self._vector_table))
            for rid, _, _ in results:
                if rid == agent_id:
                    # Still don't have the raw vector... for diversity score
                    # we need actual vectors. Return None to skip this agent.
                    pass
        except Exception:
            pass
        return None

    def _vector_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance between two vectors [0, 2]."""
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an == 0 or bn == 0:
            return 1.0
        sim = float(np.dot(a, b) / (an * bn))
        # Clamp to [-1, 1] for numerical safety
        sim = max(-1.0, min(1.0, sim))
        return 1.0 - sim  # distance = 1 - similarity

    def _avg_nearest_neighbor_distance(
        self,
        agent_id: int,
        population: list[int],
        k: int = 3,
    ) -> float:
        """Average cosine distance to k nearest neighbors."""
        vec = self._get_vector(agent_id)
        if vec is None:
            return 0.0
        distances: list[float] = []
        for other_id in population:
            if other_id == agent_id:
                continue
            other_vec = self._get_vector(other_id)
            if other_vec is not None:
                distances.append(self._vector_distance(vec, other_vec))
        distances.sort()
        taken = distances[:k]
        return sum(taken) / len(taken) if taken else 0.0

    def _is_inbred(self, a: int, b: int) -> bool:
        """Check if a and b share a recent ancestor.

        Current rule: 1-generation guard — reject if either is the
        direct parent of the other.
        """
        # Direct parent-child check
        g_a = self._wal.get_genealogy(a)
        g_b = self._wal.get_genealogy(b)
        if g_a:
            if g_a.get("parent_a") == b or g_a.get("parent_b") == b:
                return True
        if g_b:
            if g_b.get("parent_a") == a or g_b.get("parent_b") == a:
                return True
        return False

    def _find_room_for_agent(self, agent_id: int) -> int | None:
        """Reverse lookup: which room holds this agent?"""
        for rid, aid in self._room_allocations.items():
            if aid == agent_id:
                return rid
        return None

    def _extract_room_vector(self, room_id: int) -> np.ndarray:
        """Flatten room weights into a vector for the vector table."""
        parts: list[np.ndarray] = []
        for key in ("w1", "w2", "w3"):
            parts.append(self.grid.w[key][room_id].flatten())
        vec = np.concatenate(parts).astype(np.float32)
        # Pad or truncate to match vector table dim
        if self._vector_table is not None:
            target_dim = self._vector_table.dim
            if len(vec) < target_dim:
                vec = np.pad(vec, (0, target_dim - len(vec)), mode="constant")
            elif len(vec) > target_dim:
                vec = vec[:target_dim]
        return vec

    def _select_parents_random(
        self,
        n_children: int,
    ) -> list[tuple[int, int]]:
        """Legacy random fallback: pick random agents from grid."""
        hot = self.grid.top(k=max(20, n_children * 2))
        if not hot:
            return []
        room_ids = [rid for rid, _ in hot]
        pairs: list[tuple[int, int]] = []
        if len(room_ids) < 2:
            for _ in range(n_children):
                pairs.append((room_ids[0], room_ids[0]))
            return pairs
        for _ in range(n_children):
            a, b = random.sample(room_ids, 2)
            pairs.append((a, b))
        return pairs
