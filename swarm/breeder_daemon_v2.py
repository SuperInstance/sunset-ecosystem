"""BreederDaemonV2 — persistent, diversity-aware, thermal-scheduled breeding daemon.

Implements the lifecycle FSM from SPEC_BREEDER_DAEMON_V2.md:
    EGG → COMPETE → SURVIVE → BREED → SUNSET → ARCHIVE

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
    "AgentLifecycleFSM",
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
from pathlib import Path
from typing import Any, Optional

import numpy as np

from nerve.room_grid import RoomGrid
from swarm.lifecycle_fsm import AgentLifecycleFSM, LifecycleState
from swarm.thermal import DeviceType, ThermalBudget
from swarm.tournament import AgentScore, TournamentRound, breed
from swarm.inheritance_tax import InheritanceTax
from swarm.crdt_merge import CRDTMergeEngine, Agent as CRDTAgent
from swarm.lineage_checker import LineageSanityChecker, Agent as LineageAgent
from swarm.trajectory_monitor import TrajectoryMonitor, SecurityEvent
# Make cocapn_traps optional — graceful degrade when unavailable
try:
    from cocapn_traps.traps.diversity_collapse_trap import DiversityCollapseTrap
except ImportError:
    # Graceful fallback: no-op diversity monitoring
    class DiversityCollapseTrap:  # type: ignore[no-redef]
        """No-op fallback when cocapn_traps is not installed."""
        def __init__(self, *args, **kwargs):
            self._history = []
        def record(self, diversity_score):
            self._history.append(diversity_score)
        def check(self):
            if len(self._history) >= 3:
                return type("DiversityAlert", (), {"level": "CRITICAL", "recommended_action": "CROSS_SHIP_INJECTION"})()
            if len(self._history) >= 2:
                return type("DiversityAlert", (), {"level": "WARNING", "recommended_action": "EMERGENCY_MUTATE"})()
            return None
from nerve.distributed_metronome_bridge import MetronomeBridge
from swarm.mesh_vector_tables import FleetVectorIndex, VectorTableEntry
from fleet.operational_trap import TrapRegistry
from sunset.flux_preset_library import FluxPresetLibrary
from logos.a2a_identity import AgentIdentity
from logos.signed_wal import SignedWAL, WALEntry
from logos.decision_journal import log_spawn, log_sunset, log_breed

# FLUX Path A gating — optional, graceful degrade when unavailable
try:
    from swarm.flux_gating import FluxGatingChecker, FluxGatingConfig
except ImportError:
    # Graceful fallback: no-op gating
    class FluxGatingConfig:  # type: ignore[no-redef]
        """No-op fallback when swarm.flux_gating is not available."""
        pass

    class FluxGatingChecker:  # type: ignore[no-redef]
        """No-op fallback when swarm.flux_gating is not available."""
        def __init__(self, *args, **kwargs):
            pass

# FLUX Path A gating — optional, graceful degrade when unavailable
try:
    from swarm.flux_gating import FluxGatingChecker, FluxGatingConfig
except ImportError:
    # Graceful fallback: no-op gating
    class FluxGatingConfig:  # type: ignore[no-redef]
        """No-op fallback when swarm.flux_gating is not available."""
        pass

    class FluxGatingChecker:  # type: ignore[no-redef]
        """No-op fallback when swarm.flux_gating is not available."""
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


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
            'EGG','COMPETE','SURVIVE','BREED','SUNSET','ARCHIVE'
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

    def replay(self) -> dict[int, AgentLifecycleFSM]:
        """Replay WAL: return FSM for every known agent."""
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT agent_id, state FROM lifecycle ORDER BY entered_at"
        )
        fsm_map: dict[int, AgentLifecycleFSM] = {}
        for row in cur:
            agent_id = row["agent_id"]
            state_name = row["state"]
            try:
                state = LifecycleState[state_name]
            except KeyError:
                logger.warning("Unknown state %r for agent %d", state_name, agent_id)
                continue
            fsm_map[agent_id] = AgentLifecycleFSM(
                agent_id=agent_id,
                initial_state=state,
                strict=False,  # replay must not crash on historic transitions
            )
        conn.close()
        return fsm_map

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
            If omitted and *use_hdc* is True, a default HDC-enabled table
            is created automatically.
        diversity: DiversityConfig for parent selection.
        thermal_cfg: ThermalConfig for scheduling parameters.
        wal_path: Path to SQLite WAL file.
        mesh: Optional MeshNode for cross-instance breeding (future).
        use_hdc: When True, diversity matrix uses HDC (XOR+POPCNT)
            Hamming distance instead of cosine. ~100-1000× faster on
            AVX-512 hardware with 0.943 correlation.
    """

    def __init__(
        self,
        grid: RoomGrid,
        thermal: ThermalBudget,
        vector_table: Optional["FluxVectorTable"] = None,
        diversity: DiversityConfig = DiversityConfig(),
        thermal_cfg: ThermalConfig = ThermalConfig(),
        wal_path: str = "breeder.wal.sqlite",
        signed_wal_path: str | None = None,
        signed_wal_algorithm: str = "ed25519",
        mesh: Any = None,
        tick_interval: float = 1.0,
        trajectory_monitor: TrajectoryMonitor | None = None,
        inheritance_tax: InheritanceTax | None = None,
        decision_journal_path: str | None = None,
        use_hdc: bool = False,
        interval: float | None = None,
        metronome_bridge: Optional[MetronomeBridge] = None,
        fleet_vector_index: Optional[FleetVectorIndex] = None,
        trap_registry: Optional[TrapRegistry] = None,
        flux_preset_library: Optional[FluxPresetLibrary] = None,
        agent_identity: Optional[AgentIdentity] = None,
        flux_checker: Optional[Any] = None,
        compiled_checker: Optional[Any] = None,
        flux_config: FluxGatingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.grid = grid
        self.thermal = thermal
        self._vector_table = vector_table
        self._diversity = diversity
        self._thermal_cfg = thermal_cfg
        self._wal_path = wal_path
        self._mesh = mesh
        self._tick_interval = tick_interval
        self._trajectory_monitor = trajectory_monitor or TrajectoryMonitor()
        self._inheritance_tax = inheritance_tax
        self._decision_journal_path = decision_journal_path
        self._use_hdc = use_hdc
        self._metronome_bridge = metronome_bridge
        self._fleet_vector_index = fleet_vector_index
        self._trap_registry = trap_registry
        self._flux_preset_library = flux_preset_library
        self._agent_identity = agent_identity
        self._flux_checker = flux_checker
        self._compiled_checker = compiled_checker
        self._breed_signatures: dict[int, str] = {}
        self._flux_config = flux_config

        # Backward-compatible alias: interval overrides tick_interval
        if interval is not None:
            self._tick_interval = interval

        self._signed_wal_path = signed_wal_path
        self._signed_wal_algorithm = signed_wal_algorithm
        self._signed_wal: SignedWAL | None = None
        self._safe_mode: bool = False

        self._wal = _WALSchema(wal_path)
        self._fsm: dict[int, AgentLifecycleFSM] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0

        self._room_allocations: dict[int, int] = {}  # room_id → agent_id
        self._transitions_log: list[LifecycleTransition] = []
        self._slot_registry: dict[int, int] = {}  # agent_id → economic slots

        # Thermal hysteresis counter (how many ticks we've been blocked)
        self._thermal_blocked_ticks: int = 0

        # Diversity collapse monitor (hooked after each breeding round)
        self._diversity_trap = DiversityCollapseTrap(bus=mesh)

    def _check_flux(self, parent_idx: int, mutation_plan: dict[str, Any]) -> Any:
        """Check FLUX constraints using compiled checker first, then fallback.

        Extracts weights from the grid (or mutation_plan) and calls the
        checker with the standard (weights, chaos, thermal_pressure) signature.

        Returns the first non-passing result, or a passing result if both pass.
        If compiled_checker is available it takes priority; flux_checker is fallback.
        If both block, returns the compiled checker's result (stricter VM path).
        """
        # Extract weights
        if "weights" in mutation_plan:
            weights = np.asarray(mutation_plan["weights"], dtype=np.float32)
        elif hasattr(self.grid, "get_weights"):
            weights = np.asarray(self.grid.get_weights(parent_idx), dtype=np.float32)
        else:
            # parent_idx may be an agent ID or a room ID.
            room_id = self._find_room_for_agent(parent_idx)
            if room_id is None:
                room_id = parent_idx
            try:
                weights = self._extract_room_vector(room_id)
            except Exception:
                # Agent not allocated / room out of bounds — can't extract
                # weights so skip FLUX gating for this parent.
                return None

        chaos = float(mutation_plan.get("chaos", 0.3))
        thermal = float(mutation_plan.get("thermal_pressure", 0.0))

        # Try compiled checker first (Path B)
        if self._compiled_checker is not None:
            result = self._compiled_checker.check_candidate(weights, chaos, thermal)
            if not result.passed:
                return result
        # Fallback to Python checker (Path A)
        if self._flux_checker is not None:
            result = self._flux_checker.check_candidate(weights, chaos, thermal)
            if not result.passed:
                return result
        # Both passed (or neither configured)
        return result if (self._compiled_checker is not None or self._flux_checker is not None) else None

    def _flux_passed(self, result: Any | None) -> bool:
        """Return True if the FLUX check passed (or no checker configured)."""
        if result is None:
            return True  # no checker configured → pass
        return getattr(result, "passed", True)

    def attach_flux_gating(
        self,
        checker: FluxGatingChecker | None = None,
        config: FluxGatingConfig | None = None,
    ) -> None:
        """Attach or replace the FLUX constraint gating checker.

        Args:
            checker: Pre-built FluxGatingChecker instance.  If None, one is
                created from *config* (or from ``self._flux_config``).
            config: Override configuration.  Only used when *checker* is None.
        """
        if checker is not None:
            self._flux_checker = checker
            logger.info("FLUX gating checker attached (external instance)")
            return

        cfg = config or self._flux_config
        if cfg is None:
            cfg = FluxGatingConfig()
        self._flux_checker = FluxGatingChecker(config=cfg)
        logger.info("FLUX gating checker attached (config=%s)", cfg)

        # FLUX Path A gating — initialized on first attach_flux_gating() call
        self._flux_checker: FluxGatingChecker | None = None

    def attach_flux_gating(
        self,
        checker: FluxGatingChecker | None = None,
        config: FluxGatingConfig | None = None,
    ) -> None:
        """Attach or replace the FLUX constraint gating checker.

        Args:
            checker: Pre-built FluxGatingChecker instance.  If None, one is
                created from *config* (or from ``self._flux_config``).
            config: Override configuration.  Only used when *checker* is None.
        """
        if checker is not None:
            self._flux_checker = checker
            logger.info("FLUX gating checker attached (external instance)")
            return

        cfg = config or self._flux_config
        if cfg is None:
            cfg = FluxGatingConfig()
        self._flux_checker = FluxGatingChecker(config=cfg)
        logger.info("FLUX gating checker attached (config=%s)", cfg)

    # ── public API ──────────────────────────────────────────

    def start(self) -> None:
        """Replay WAL, start scheduler thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        # Replay WAL to reconstruct state
        self._fsm = self._wal.replay()

        # Seed slot registry for recovered agents
        if self._inheritance_tax is not None:
            for agent_id in self._fsm:
                if agent_id not in self._slot_registry:
                    self._slot_registry[agent_id] = InheritanceTax.DEFAULT_SLOTS

        logger.info(
            "BreederDaemonV2 replayed WAL: %d agents, %d pending queue items",
            len(self._fsm),
            self._wal.count_pending(),
        )

        # ── Signed WAL initialization and integrity check ──────
        if self._signed_wal_path:
            self._signed_wal = SignedWAL(
                algorithm=self._signed_wal_algorithm,
                log_path=self._signed_wal_path,
            )
            ok, first_bad = self._signed_wal.verify_chain()
            if not ok:
                logger.error(
                    "Signed WAL tampering detected at index %d! Entering safe mode.",
                    first_bad,
                )
                self._safe_mode = True
            else:
                logger.info(
                    "Signed WAL verified: %d entries, all valid",
                    len(self._signed_wal),
                )
                self._safe_mode = False

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

        Delegates to _select_parents_vector when a FluxVectorTable is
        available, otherwise falls back to random selection.

        Fleet-aware: if fleet_vector_index is attached, cross-node
        candidates are merged with local candidates and deduplicated.
        """
        # ── Local candidates ────────────────────────────────────
        local_candidates = self._get_breedable_candidates()

        # ── Fleet-wide candidates ───────────────────────────────
        fleet_candidates: list[int] = []
        if self._fleet_vector_index is not None:
            try:
                entries = self._fleet_vector_index.get_breedable_pool()
                for entry in entries:
                    aid_str = entry.agent_id
                    if "::agent_" in aid_str:
                        try:
                            aid = int(aid_str.split("::agent_")[-1])
                            fleet_candidates.append(aid)
                        except ValueError:
                            pass
                    else:
                        try:
                            aid = int(aid_str)
                            fleet_candidates.append(aid)
                        except ValueError:
                            pass
            except Exception:
                logger.exception("FleetVectorIndex.get_breedable_pool failed, using local only")

        # Merge and deduplicate (local first, then fleet)
        merged = list(dict.fromkeys(local_candidates + fleet_candidates))

        pairs = self._select_parents_vector(
            population=merged,
            vector_table=self._vector_table,
            n_children=n_children,
        )
        # Pad with random if we didn't get enough — but respect FLUX gating
        attempts = 0
        max_attempts = n_children * 10
        while len(pairs) < n_children and attempts < max_attempts:
            extra = self._select_parents_random(n_children - len(pairs))
            for a, b in extra:
                plan = {"parents": (a, b)}
                result = self._check_flux(a, plan)
                if not self._flux_passed(result):
                    logger.debug("FLUX blocked random pair (%d, %d)", a, b)
                    attempts += 1
                    continue
                pairs.append((a, b))
                attempts += 1
                if len(pairs) >= n_children:
                    break
        return pairs[:n_children]
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

        # If only one parent was specified, select a diverse mate
        if parent_b is None:
            candidates = self._get_breedable_candidates()
            if parent_a in candidates:
                pairs = self._select_parents_vector(
                    population=candidates,
                    vector_table=self._vector_table,
                    n_children=1,
                )
                if pairs:
                    _, parent_b = pairs[0]

        # ── TrajectoryMonitor circuit breaker ───────────────────
        parents_to_check = [p for p in (parent_a, parent_b) if p is not None]
        flagged_parents = self._trajectory_monitor.circuit_breaker(parents_to_check)
        if flagged_parents:
            logger.warning(
                "Step %d: breeding ticket %d aborted — "
                "anomalous trajectory detected in parent(s) %s",
                tick, ticket, flagged_parents,
            )
            # Do NOT re-queue — a flagged parent is a security event.
            # Return empty transitions; the ticket is consumed.
            return transitions

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

        # ── FLUX gating: check candidate before room allocation ───
        mutation_plan = {
            "parents": (parent_a, parent_b),
            "room_id": None,  # not yet known
        }
        flux_result = self._check_flux(parent_a, mutation_plan)
        if not self._flux_passed(flux_result):
            logger.warning(
                "Step %d: FLUX blocked ticket %d (parents=%s): %s",
                tick, ticket, (parent_a, parent_b),
                getattr(flux_result, "violations", "unknown"),
            )
            # Re-queue for later retry — constraint may loosen
            self._wal.enqueue_breed(parent_a, parent_b, priority, remote)
            return transitions

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
            old_fsm = self._fsm.get(old_agent_id)
            old_state = old_fsm.get_state() if old_fsm else LifecycleState.COMPETE
            sunset_tr = LifecycleTransition(
                agent_id=old_agent_id,
                from_state=old_state,
                to_state=LifecycleState.SUNSET,
                timestamp=time.time(),
            )
            self._wal.transition(sunset_tr)
            self._log_transition_to_signed_wal(sunset_tr)
            if self._decision_journal_path:
                log_sunset(
                    agent_id=old_agent_id,
                    reason="room_reuse",
                    generation=0,
                    journal_path=self._decision_journal_path,
                )
            self._fsm[old_agent_id] = AgentLifecycleFSM(
                agent_id=old_agent_id, initial_state=LifecycleState.SUNSET, strict=False
            )
            transitions.append(sunset_tr)

        # Determine child agent ID
        child_id = self._next_agent_id()

        # ── Inheritance Tax ─────────────────────────────────────
        if self._inheritance_tax is not None and parent_a is not None:
            parent_slots = self._slot_registry.get(
                parent_a, InheritanceTax.DEFAULT_SLOTS
            )
            # Child inherits a portion of parent's slots
            child_slots = int(parent_slots * 0.8)

            # Fitness from vector table (0.0 default for unscored agents)
            parent_fitness = 0.0
            child_fitness = 0.0
            if self._vector_table is not None:
                meta = self._vector_table._meta.get(parent_a)
                if meta is not None:
                    parent_fitness = meta.fitness

            parent_after, child_after = self._inheritance_tax.apply_tax(
                parent_slots, child_slots, parent_fitness, child_fitness
            )
            self._slot_registry[parent_a] = parent_after
            self._slot_registry[child_id] = child_after

            # Social welfare: grant bonus slots from global pool
            bonus = self._inheritance_tax.fund_new_agent(
                InheritanceTax.DEFAULT_SLOTS
            )
            if bonus > 0:
                self._slot_registry[child_id] += bonus
                logger.info(
                    "InheritanceTax: agent %d granted %d bonus slots "
                    "from global pool (pool=%d)",
                    child_id, bonus, self._inheritance_tax.global_pool
                )

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

        # ── FLUX Path A: candidate gating ───────────────────────
        if self._flux_checker is not None:
            # Gather parent room weights and chaos for FLUX check
            flux_results: list[tuple[int, Any]] = []
            for pid in (parent_a, parent_b):
                if pid is None:
                    continue
                pr = self._find_room_for_agent(pid)
                if pr is not None:
                    wvec = self._extract_room_vector(pr)
                    chaos_val = float(self.grid.chaos[pr])
                    thermal_val = self.thermal.thermal_headroom()
                    if hasattr(self.thermal, "get_device") and aid is not None:
                        dev = self.thermal.get_device(f"agent_{aid}")
                        if dev is not None:
                            db = self.thermal.device_budget(dev)
                            if db.max_agents > 0:
                                thermal_val = db.current_agents / db.max_agents
                    res = self._flux_checker.check_candidate(
                        weights=wvec,
                        chaos=chaos_val,
                        thermal_pressure=thermal_val,
                    )
                    flux_results.append((pid, res))

            flux_fails = [(pid, r) for pid, r in flux_results if not r.passed]
            if flux_fails:
                fail_str = ", ".join(
                    f"agent_{pid}({r.violations})" for pid, r in flux_fails
                )
                logger.warning(
                    "Step %d: breeding ticket %d FLUX-gated — parent(s) %s",
                    tick, ticket, fail_str,
                )
                # Re-queue with slightly lower priority to avoid spin-lock
                self._wal.enqueue_breed(parent_a, parent_b, max(0, priority - 1), remote)
                return transitions

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
        self._log_transition_to_signed_wal(egg_tr)
        self._fsm[child_id] = AgentLifecycleFSM(
            agent_id=child_id, initial_state=LifecycleState.EGG, strict=False
        )
        transitions.append(egg_tr)

        if self._decision_journal_path:
            log_breed(
                parent_a=parent_a,
                parent_b=parent_b,
                child_id=child_id,
                generation=generation,
                journal_path=self._decision_journal_path,
            )
            log_spawn(
                agent_id=child_id,
                parents=(parent_a, parent_b),
                generation=generation,
                reason="breeder_daemon_v2 step",
                journal_path=self._decision_journal_path,
            )

        # Allocate room → COMPETE
        # Use grid.rebirth() to reset room, then clone parent weights
        parent_room = self._find_room_for_agent(parent_a)
        if parent_room is not None:
            self.grid.breed(parent_room, room_id)
        else:
            self.grid.rebirth(room_id)

        incubate_tr = LifecycleTransition(
            agent_id=child_id,
            from_state=LifecycleState.EGG,
            to_state=LifecycleState.COMPETE,
            timestamp=time.time(),
            generation=generation,
            parent_a=parent_a,
            parent_b=parent_b,
            origin_node="local" if not remote else "remote",
        )
        self._wal.transition(incubate_tr)
        self._log_transition_to_signed_wal(incubate_tr)
        self._fsm[child_id] = AgentLifecycleFSM(
            agent_id=child_id, initial_state=LifecycleState.COMPETE, strict=False
        )
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
            # Record trajectory for security monitoring
            self._trajectory_monitor.record(child_id, vec)

            # ── LineageSanityChecker integration ────────────────────
            lineage_checker = LineageSanityChecker(max_depth=5)
            population_agents = self._build_lineage_population()
            child_agent = LineageAgent(
                id=child_id,
                vector=vec.tolist(),
                generation=generation,
                parent_a=parent_a,
                parent_b=parent_b,
            )
            is_valid, reason = lineage_checker.verify_lineage(
                child_id, population_agents + [child_agent]
            )
            if not is_valid:
                logger.warning(
                    "LineageSanityChecker failed for agent %d: %s",
                    child_id, reason,
                )
                # Log security event
                event = SecurityEvent(
                    agent_id=child_id,
                    z_score=0.0,
                    threshold=0.0,
                    generation_count=generation,
                    message=(
                        f"Lineage tamper detected for agent {child_id}: {reason}"
                    ),
                )
                self._trajectory_monitor._events.append(event)
                # Sunset the child immediately
                sunset_tr = LifecycleTransition(
                    agent_id=child_id,
                    from_state=LifecycleState.COMPETE,
                    to_state=LifecycleState.SUNSET,
                    timestamp=time.time(),
                    generation=generation,
                    parent_a=parent_a,
                    parent_b=parent_b,
                    origin_node="local" if not remote else "remote",
                )
                self._wal.transition(sunset_tr)
                self._log_transition_to_signed_wal(sunset_tr)
                if self._decision_journal_path:
                    log_sunset(
                        agent_id=child_id,
                        reason=f"lineage_tamper: {reason}",
                        generation=generation,
                        journal_path=self._decision_journal_path,
                    )
                self._fsm[child_id] = AgentLifecycleFSM(
                    agent_id=child_id, initial_state=LifecycleState.SUNSET, strict=False
                )
                transitions.append(sunset_tr)
                # Release resources
                self.thermal.release(child_agent_str)
                del self._room_allocations[room_id]
                if self._vector_table is not None:
                    try:
                        self._vector_table.remove(child_id)
                    except Exception:
                        pass
                return transitions

        logger.info(
            "Step %d: spawned agent %d in room %d (parents=%s, gen=%d)",
            tick,
            child_id,
            room_id,
            (parent_a, parent_b),
            generation,
        )

        # ── Diversity collapse hook ───────────────────────────────
        if self._vector_table is not None:
            diversity = self.diversity_score
            self._diversity_trap.record(diversity)
            alert = self._diversity_trap.check()
            if alert is not None:
                logger.warning(
                    "DIVERSITY %s (tick=%d): %s",
                    alert.level,
                    tick,
                    alert.recommended_action,
                )

        # ── FLUX Path A: batch check top-k active rooms ───────────
        if self._flux_checker is not None:
            topk = self.grid.top(k=self._flux_checker.config.top_k_batch)
            if topk:
                weights_batch: list[np.ndarray] = []
                chaos_vec: list[float] = []
                thermal_vec: list[float] = []
                room_ids_checked: list[int] = []
                for rid, _ in topk:
                    wvec = self._extract_room_vector(rid)
                    weights_batch.append(wvec)
                    chaos_vec.append(float(self.grid.chaos[rid]))
                    aid = self._room_allocations.get(rid)
                    thermal_val = self.thermal.thermal_headroom()
                    if hasattr(self.thermal, "get_device") and aid is not None:
                        dev = self.thermal.get_device(f"agent_{aid}")
                        if dev is not None:
                            db = self.thermal.device_budget(dev)
                            if db.max_agents > 0:
                                thermal_val = db.current_agents / db.max_agents
                    thermal_vec.append(thermal_val)
                    room_ids_checked.append(rid)

                if weights_batch:
                    batch_results = self._flux_checker.check_batch(
                        np.stack(weights_batch),
                        np.array(chaos_vec, dtype=np.float32),
                        np.array(thermal_vec, dtype=np.float32),
                    )
                    for rid, br in zip(room_ids_checked, batch_results):
                        if not br.passed:
                            # Increase chaos for violating rooms
                            self.grid.chaos[rid] += 0.1
                            logger.debug(
                                "FLUX batch: room %d failed (%s), chaos bumped to %.3f",
                                rid, br.violations, self.grid.chaos[rid],
                            )

        return transitions

    def _build_lineage_population(self) -> list[LineageAgent]:
        """Construct Agent records for every known agent (state + vector table + genealogy)."""
        # Collect all known agent IDs from multiple sources
        all_ids: set[int] = set()

        # 1. Agents in lifecycle state (WAL)
        for aid, fsm in self._fsm.items():
            st = fsm.get_state()
            if st != LifecycleState.SUNSET:
                all_ids.add(aid)

        # 2. Agents in vector table metadata
        if self._vector_table is not None:
            all_ids.update(self._vector_table._meta.keys())

        # 3. Agents in genealogy table
        conn = sqlite3.connect(self._wal_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT agent_id, parent_a, parent_b, generation FROM genealogy")
        genealogy_rows: dict[int, dict[str, Any]] = {}
        for row in cur:
            aid = row["agent_id"]
            all_ids.add(aid)
            if row["parent_a"] is not None:
                all_ids.add(row["parent_a"])
            if row["parent_b"] is not None:
                all_ids.add(row["parent_b"])
            genealogy_rows[aid] = dict(row)
        conn.close()

        agents: list[LineageAgent] = []
        for aid in all_ids:
            g = genealogy_rows.get(aid) or self._wal.get_genealogy(aid)
            vec: list[float] = []
            if self._vector_table is not None:
                # Try to get raw vector from index
                if hasattr(self._vector_table._index, "_vectors"):
                    v = self._vector_table._index._vectors.get(aid)
                    if v is not None:
                        vec = v.tolist() if hasattr(v, "tolist") else list(v)
            gen = g["generation"] if g else 0
            pa = g.get("parent_a") if g else None
            pb = g.get("parent_b") if g else None
            agents.append(
                LineageAgent(
                    id=aid,
                    vector=vec,
                    generation=gen,
                    parent_a=pa,
                    parent_b=pb,
                )
            )
        return agents


    def merge_remote_population(
        self, remote_agents: list[CRDTAgent]
    ) -> list[CRDTAgent]:
        """Merge a remote population into local state before accepting new agents.

        Called when a network partition heals and a remote node sends its
        divergent population.  Runs CRDT merge, logs a divergence report,
        and returns the merged agent list (which the caller may use to update
        its own vector table / WAL).
        """
        local_agents = self._build_crdt_population()

        # Use the local vector table as the merge target; if none, create a
        # transient one so the engine can still perform lineage checks.
        from swarm.vector_table import FluxVectorTable
        if self._vector_table is None:
            self._vector_table = FluxVectorTable(dim=64, bit_width=4, use_hdc=self._use_hdc)
        vt = self._vector_table
        engine = CRDTMergeEngine(vt)

        merged = engine.merge_populations(local_agents, remote_agents)
        report = engine.detect_divergence(local_agents, remote_agents)

        logger.info(
            "Remote merge report: local_only=%d remote_only=%d "
            "common_diverged=%d lineage_conflicts=%d fitness_delta=%.4f",
            len(report.local_only),
            len(report.remote_only),
            len(report.common_diverged),
            len(report.lineage_conflicts),
            report.fitness_delta,
        )

        # Update our vector table reference if we created a transient one
        if self._vector_table is None and vt is not None:
            self._vector_table = vt

        return merged

    def _build_crdt_population(self) -> list[CRDTAgent]:
        """Construct CRDT Agent records from current non-SUNSET WAL state."""
        agents: list[CRDTAgent] = []
        for aid, fsm in self._fsm.items():
            st = fsm.get_state()
            if st == LifecycleState.SUNSET:
                continue
            g = self._wal.get_genealogy(aid)
            vec: list[float] = []
            last_updated = 0.0
            fitness = 0.0
            capability = 0xFFFF
            if self._vector_table is not None:
                meta = self._vector_table._meta.get(aid)
                if meta is not None:
                    last_updated = float(meta.extra.get("last_updated", 0.0))
                    fitness = meta.fitness
                    capability = meta.capability_mask
                if hasattr(self._vector_table._index, "_vectors"):
                    v = self._vector_table._index._vectors.get(aid)
                    if v is not None:
                        vec = v.tolist() if hasattr(v, "tolist") else list(v)
            gen = g["generation"] if g else 0
            pa = g.get("parent_a") if g else None
            pb = g.get("parent_b") if g else None
            agents.append(
                CRDTAgent(
                    agent_id=aid,
                    fitness=fitness,
                    generation=gen,
                    parent_a=pa,
                    parent_b=pb,
                    vector=vec,
                    last_updated=last_updated,
                    capability_mask=capability,
                )
            )
        return agents

    @property
    def state(self) -> dict[int, LifecycleState]:
        """Current lifecycle state of every known agent."""
        with self._lock:
            return {aid: fsm.get_state() for aid, fsm in self._fsm.items()}

    @property
    def diversity_score(self) -> float:
        """Average pairwise cosine distance in current population.

        Range [0, 2]. Below 0.15 triggers mesh breeding request.
        """
        if self._vector_table is None or len(self._vector_table) == 0:
            return 0.0

        # Collect vectors for all non-SUNSET agents
        agent_ids = [
            aid for aid, fsm in self._fsm.items()
            if fsm.get_state() not in (LifecycleState.SUNSET, LifecycleState.EGG)
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

    def _log_transition_to_signed_wal(self, tr: LifecycleTransition) -> None:
        """Mirror a lifecycle transition into the cryptographically signed WAL."""
        if self._signed_wal is None:
            return

        # Map lifecycle state to operation name
        op_map = {
            LifecycleState.EGG: "spawn",
            LifecycleState.COMPETE: "spawn",
            LifecycleState.COMPETE: "mutate",
            LifecycleState.SURVIVE: "mutate",
            LifecycleState.BREED: "breed",
            LifecycleState.SUNSET: "sunset",
        }
        operation = op_map.get(tr.to_state, "signal")

        parent_ids: list[int] = []
        if tr.parent_a is not None:
            parent_ids.append(tr.parent_a)
        if tr.parent_b is not None:
            parent_ids.append(tr.parent_b)

        entry = WALEntry(
            timestamp=tr.timestamp,
            agent_id=tr.agent_id,
            operation=operation,
            vector_hash=tr.vector_hash or "0" * 64,
            parent_ids=parent_ids,
            generation=tr.generation,
        )
        self._signed_wal.append(entry)

    # ── compatibility shim: wrap AutoBreeder ────────────────

    def auto_breed(
        self,
        n_winners: int | None = None,
    ) -> list[tuple[int, str]]:
        """Legacy compatibility: run one breeding cycle via V2 machinery.

        Uses select_parents() + queue_breed() + step() to emulate
        the old AutoBreeder.auto_breed() behavior.

        Fleet-aware wiring:
        - Ticks metronome_bridge before breeding if attached.
        - Queries fleet_vector_index for cross-node parents if attached.
        - Applies flux_preset_library preset if attached.
        - Signs breed records with agent_identity if attached.
        - Runs trap_registry after breeding if attached.
        """
        # ── Fleet beat sync ─────────────────────────────────────
        if self._metronome_bridge is not None:
            try:
                self._metronome_bridge.tick()
            except Exception:
                logger.exception("MetronomeBridge tick failed")

        # ── FLUX preset gating ──────────────────────────────────
        preset_name: str | None = None
        if self._flux_preset_library is not None:
            try:
                preset_name = self._flux_preset_library.suggest_preset_for_task("breeding")
                self._flux_preset_library.apply_preset(preset_name, {"daemon": self, "cycle": "auto_breed"})
            except Exception:
                logger.exception("FluxPresetLibrary apply_preset failed for %s", preset_name)

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
                if tr.to_state == LifecycleState.COMPETE:
                    parent_str = f"agent_{tr.parent_a}" if tr.parent_a else "unknown"
                    # Find which room got this agent
                    room_id = None
                    for rid, aid in self._room_allocations.items():
                        if aid == tr.agent_id:
                            room_id = rid
                            break
                    if room_id is not None:
                        results.append((room_id, parent_str))
                    # ── Sign breed records ──────────────────────────
                    if self._agent_identity is not None:
                        payload = {
                            "task": "breed",
                            "agent_id": tr.agent_id,
                            "parent_a": tr.parent_a,
                            "parent_b": tr.parent_b,
                            "generation": tr.generation,
                            "preset": preset_name,
                        }
                        try:
                            sig = self._agent_identity.sign_task(payload)
                            self._breed_signatures[tr.agent_id] = sig
                        except Exception:
                            logger.exception("AgentIdentity sign_task failed for agent %d", tr.agent_id)

        # ── Operational traps ───────────────────────────────────
        if self._trap_registry is not None:
            try:
                self._trap_registry.run_all()
            except Exception:
                logger.exception("TrapRegistry run_all failed")

        return results

    def get_fleet_status(self) -> dict[str, Any]:
        """Unified status from all attached fleet modules."""
        status: dict[str, Any] = {
            "running": self.running,
            "tick_count": self.tick_count,
            "agent_count": len(self._fsm),
            "diversity_score": self.diversity_score,
            "thermal_blocked_ticks": getattr(self, "_thermal_blocked_ticks", 0),
        }
        status["metronome_bridge"] = self._metronome_bridge is not None
        status["fleet_vector_index"] = self._fleet_vector_index is not None
        status["trap_registry"] = self._trap_registry is not None
        status["flux_preset_library"] = self._flux_preset_library is not None
        status["agent_identity"] = self._agent_identity is not None
        if self._metronome_bridge is not None:
            try:
                status["metronome_bridge_beat"] = getattr(self._metronome_bridge, "_beat_count", 0)
            except Exception:
                status["metronome_bridge_beat"] = -1
        if self._fleet_vector_index is not None:
            try:
                pool = self._fleet_vector_index.get_breedable_pool()
                status["fleet_vector_index_pool_size"] = len(pool)
            except Exception:
                status["fleet_vector_index_pool_size"] = -1
        if self._trap_registry is not None:
            try:
                status["trap_registry_results"] = len(self._trap_registry.run_all())
            except Exception:
                status["trap_registry_results"] = -1
        if self._flux_preset_library is not None:
            try:
                status["flux_preset_suggested"] = self._flux_preset_library.suggest_preset_for_task("breeding")
            except Exception:
                status["flux_preset_suggested"] = None
        if self._agent_identity is not None:
            try:
                status["agent_identity_name"] = getattr(self._agent_identity, "agent_id", "unknown")
            except Exception:
                status["agent_identity_name"] = None
        return status

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
            aid for aid, fsm in self._fsm.items()
            if fsm.get_state() in (LifecycleState.SURVIVE, LifecycleState.BREED)
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

    def _select_parents_vector(
        self,
        population: list[int],
        vector_table: Optional["FluxVectorTable"] = None,
        n_children: int = 1,
    ) -> list[tuple[int, int]]:
        """Diversity-aware parent selection using FLUX vector table methods.

        Priority when *vector_table* is provided:
        1. ``search_diverse_parents`` — maximally diverse pairs via novelty
           + tournament results, filtered to breedable population.
        2. ``recommend_breed_pair`` — niche-crossing high-fitness parents.
        3. Legacy fitness + distance-from-centroid fallback.
        4. Random fallback.

        When *vector_table* is None, falls back to fitness-only selection
        from the provided population.
        """
        if len(population) < 2:
            return self._select_parents_random(n_children)

        pairs: list[tuple[int, int]] = []

        # ── Fitness-only fallback when no vector table ─────
        if vector_table is None or len(vector_table) == 0:
            scored: list[tuple[int, float]] = []
            for aid in population:
                fitness = 0.0
                if self._vector_table is not None:
                    meta = self._vector_table._meta.get(aid)
                    if meta is not None:
                        fitness = meta.fitness
                scored.append((aid, fitness))

            scored.sort(key=lambda x: x[1], reverse=True)
            for _ in range(n_children):
                parent_a = scored[0][0]
                parent_b = scored[1][0] if len(scored) > 1 else scored[0][0]
                pairs.append((parent_a, parent_b))
            return pairs

        # ── Path 1: FluxVectorTable.search_diverse_parents ─────
        if hasattr(vector_table, "search_diverse_parents"):
            try:
                diverse_pairs = vector_table.search_diverse_parents(
                    n_results=n_children * 2
                )
                for a, b in diverse_pairs:
                    if a in population and b in population and not self._is_inbred(a, b):
                        # FLUX gating: check candidate before accepting
                        plan = {"parents": (a, b)}
                        result = self._check_flux(a, plan)
                        if not self._flux_passed(result):
                            logger.debug("FLUX blocked diverse pair (%d, %d): %s", a, b, getattr(result, "violations", "unknown"))
                            continue
                        pairs.append((a, b))
                    if len(pairs) >= n_children:
                        return pairs[:n_children]
            except Exception:
                logger.exception("search_diverse_parents failed, falling back")

        # ── Path 2: FluxVectorTable.recommend_breed_pair ───────
        if len(pairs) < n_children and hasattr(vector_table, "recommend_breed_pair"):
            try:
                rec = vector_table.recommend_breed_pair()
                if rec is not None:
                    a, b = rec
                    if a in population and b in population and not self._is_inbred(a, b):
                        if (a, b) not in pairs and (b, a) not in pairs:
                            # FLUX gating
                            plan = {"parents": (a, b)}
                            result = self._check_flux(a, plan)
                            if not self._flux_passed(result):
                                logger.debug("FLUX blocked recommend pair (%d, %d): %s", a, b, getattr(result, "violations", "unknown"))
                            else:
                                pairs.append((a, b))
                        else:
                            pairs.append((a, b))
                    if len(pairs) >= n_children:
                        return pairs[:n_children]
            except Exception:
                logger.exception("recommend_breed_pair failed, falling back")

        # ── Path 3: Legacy fitness + distance-from-centroid ────
        scored = []
        for aid in population:
            meta = vector_table._meta.get(aid)
            if meta is None:
                continue
            fitness = meta.fitness

            vec = self._get_vector(aid)
            if vec is None:
                continue

            # Build population vectors excluding self
            pop_vectors: list[list[float]] = []
            for other_id in population:
                if other_id == aid:
                    continue
                other_vec = self._get_vector(other_id)
                if other_vec is not None:
                    pop_vectors.append(other_vec.tolist())

            novelty = vector_table.compute_novelty(aid, vec.tolist(), pop_vectors)
            score = fitness + novelty
            scored.append((aid, score, fitness, novelty))

        if len(scored) >= 2:
            scored.sort(key=lambda x: x[1], reverse=True)
            while len(pairs) < n_children:
                parent_a = scored[0][0]

                # Find most distant compatible candidate
                best_b: int | None = None
                best_dist = -1.0

                vec_a = self._get_vector(parent_a)
                if vec_a is not None:
                    for aid, _, _, _ in scored[1:]:
                        if aid == parent_a:
                            continue
                        if self._is_inbred(parent_a, aid):
                            continue
                        vec_b = self._get_vector(aid)
                        if vec_b is not None:
                            dist = self._vector_distance(vec_a, vec_b)
                            # ── FLUX Path A: tiebreak ──────────────────
                            if self._flux_checker is not None:
                                pr_a = self._find_room_for_agent(parent_a)
                                pr_b = self._find_room_for_agent(aid)
                                if pr_a is not None and pr_b is not None:
                                    flux_score = self._flux_checker.score_for_breeding(
                                        self._extract_room_vector(pr_a),
                                        self._extract_room_vector(pr_b),
                                        chaos_a=float(self.grid.chaos[pr_a]),
                                        chaos_b=float(self.grid.chaos[pr_b]),
                                    )
                                    # Blend distance with FLUX compliance
                                    # FLUX score ∈ [0,1]; small weight so it
                                    # only breaks ties, not dominates
                                    dist = dist + 0.05 * flux_score

                            if dist > best_dist:
                                best_dist = dist
                                best_b = aid

                if best_b is None:
                    best_b = scored[1][0] if len(scored) > 1 else scored[0][0]

                # FLUX gating on legacy fallback pair
                plan = {"parents": (parent_a, best_b)}
                result = self._check_flux(parent_a, plan)
                if not self._flux_passed(result):
                    logger.debug("FLUX blocked legacy pair (%d, %d): %s", parent_a, best_b, getattr(result, "violations", "unknown"))
                    # Try next best candidate
                    for alt_aid, _, _, _ in scored[1:]:
                        if alt_aid == parent_a or alt_aid == best_b:
                            continue
                        if self._is_inbred(parent_a, alt_aid):
                            continue
                        alt_plan = {"parents": (parent_a, alt_aid)}
                        alt_result = self._check_flux(parent_a, alt_plan)
                        if self._flux_passed(alt_result):
                            best_b = alt_aid
                            break
                    else:
                        # No alt passed — skip this child
                        continue

                pairs.append((parent_a, best_b))
                if len(pairs) >= n_children:
                    return pairs[:n_children]

        # ── Path 4: Random fallback ────────────────────────────
        attempts = 0
        max_attempts = n_children * 10
        while len(pairs) < n_children and attempts < max_attempts:
            extra = self._select_parents_random(n_children - len(pairs))
            for a, b in extra:
                plan = {"parents": (a, b)}
                result = self._check_flux(a, plan)
                if not self._flux_passed(result):
                    logger.debug("FLUX blocked random pair (%d, %d)", a, b)
                    attempts += 1
                    continue
                pairs.append((a, b))
                attempts += 1
                if len(pairs) >= n_children:
                    break
        return pairs[:n_children]

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
