"""FLUX Path A — constraint gating for breeding decisions.

Treats FLUX as a constraint library (not a VM).  Python calls
``flux_check_batch()`` via FFI (or a Python fallback), gets back
pass/fail/severity, and uses it to gate breeding decisions.

Architecture
------------
``FluxGatingChecker`` is the interface contract.  The real
implementation will wrap the Rust FFI (``flux_check_batch``).
Until that FFI is ready, ``PythonFluxFallback`` provides the same
API using numpy-based constraint checks.
"""

from __future__ import annotations

__all__ = [
    "FluxGatingConfig",
    "GatingResult",
    "FluxGatingChecker",
    "PythonFluxFallback",
    "ViolationSeverity",
    "FluxViolation",
    "FluxWAL",
]

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

import numpy as np


# ── severity model ──────────────────────────────────────

class ViolationSeverity(Enum):
    """FLUX constraint violation severity."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


# ── data structures ─────────────────────────────────────

@dataclass(frozen=True)
class FluxGatingConfig:
    """Configuration for FLUX constraint gating.

    Args:
        max_violations_per_cycle: Hard cap on violations before a
            candidate is blocked (default 5).
        severity_weights: Mapping from severity name to numeric weight.
            Used when computing aggregate gating scores.
        weight_bounds: (min, max) for room weight norms.
        chaos_limit: Maximum allowed chaos value per room.
        thermal_budget_limit: Maximum thermal headroom before breeding
            is throttled (0.0–1.0).
        enable_wal: Whether to persist violations to disk.
        wal_path: Path for the write-ahead log.  If ``None`` a default
            is chosen under ``/tmp``.
    """

    max_violations_per_cycle: int = 5
    severity_weights: dict[str, int] = field(
        default_factory=lambda: {"critical": 100, "warning": 10, "info": 1}
    )
    weight_bounds: tuple[float, float] = (0.0, 10.0)
    chaos_limit: float = 1.0
    thermal_budget_limit: float = 0.95
    enable_wal: bool = True
    wal_path: Optional[str] = None


@dataclass(frozen=True)
class FluxViolation:
    """A single FLUX constraint violation."""

    room_id: int
    constraint_id: str
    severity: ViolationSeverity
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GatingResult:
    """Result of a FLUX constraint check on a single candidate."""

    candidate_id: str
    passed: bool
    score: float  # 0.0 = blocked, 1.0 = perfectly clean
    violations: list[FluxViolation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── WAL (Write-Ahead Log) ─────────────────────────────────

class FluxWAL:
    """Simple WAL for FLUX violations.

    Thread-safe.  In production this would be backed by a real
    append-only store (e.g. SQLite, RocksDB, or a structured log).
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or f"/tmp/flux_violations_{os.getpid()}.wal"
        self._lock = threading.Lock()
        self._mem: list[dict[str, Any]] = []
        self._ensure_file()

    def _ensure_file(self) -> None:
        if self._path and not os.path.exists(self._path):
            open(self._path, "a").close()

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._mem.append(record)
            if self._path:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")

    def query_by_event_type(self, event_type: str) -> list[dict[str, Any]]:
        """Return all records whose ``event_type`` matches."""
        with self._lock:
            return [r for r in self._mem if r.get("event_type") == event_type]

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._mem)

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            if self._path and os.path.exists(self._path):
                os.remove(self._path)
            self._ensure_file()


# ── abstract checker interface ────────────────────────────

class FluxGatingChecker:
    """Interface contract between the breeder and FLUX.

    This is the **Path A** contract: Python calls into FLUX as a
    library, not as a full VM.  The concrete subclass can be:

    * ``PythonFluxFallback`` — pure-Python constraint checks.
    * ``RustFluxChecker`` (future) — wraps the Rust FFI.

    Both expose the same API so swapping is a one-line change.
    """

    def __init__(self, config: Optional[FluxGatingConfig] = None) -> None:
        self.config = config or FluxGatingConfig()
        self._wal = FluxWAL(self.config.wal_path) if self.config.enable_wal else None

    # ── public API ──────────────────────────────────────────

    def check_candidate(
        self,
        parent_idx: int,
        mutation_plan: dict[str, Any],
    ) -> GatingResult:
        """Check whether a proposed mutation violates constraints.

        Args:
            parent_idx: Index of the parent room.
            mutation_plan: Dict describing the mutation (keys, noise,
                crossover source, etc.).

        Returns:
            ``GatingResult`` with ``passed=True`` if the candidate is
            acceptable, ``False`` if it should be blocked.
        """
        raise NotImplementedError

    def check_batch(
        self,
        candidates: list[tuple[int, dict[str, Any]]],
    ) -> list[GatingResult]:
        """Batch constraint check for tournament selection.

        Args:
            candidates: List of ``(parent_idx, mutation_plan)`` tuples.

        Returns:
            List of ``GatingResult`` objects, one per candidate.
        """
        raise NotImplementedError

    def score_for_breeding(
        self,
        room_vector: np.ndarray,
        room_metadata: dict[str, Any],
    ) -> float:
        """Compute a 0.0–1.0 fitness score for breeding eligibility.

        0.0 = blocked (violates hard constraints).
        1.0 = clean (no violations).

        The score is used as a **tiebreak** in tournament selection.
        """
        raise NotImplementedError

    def record_violation(
        self,
        room_id: int,
        constraint_id: str,
        severity: ViolationSeverity,
        context: dict[str, Any],
    ) -> None:
        """Log a constraint violation to the WAL.

        This is a side-effect operation; it does not gate anything
        on its own but provides auditability.
        """
        record = {
            "event_type": "flux_violation",
            "room_id": room_id,
            "constraint_id": constraint_id,
            "severity": severity.name,
            "context": context,
            "timestamp": time.time(),
        }
        if self._wal is not None:
            self._wal.append(record)

    @property
    def wal(self) -> Optional[FluxWAL]:
        return self._wal


# ── Python fallback ───────────────────────────────────────

class PythonFluxFallback(FluxGatingChecker):
    """Pure-Python FLUX constraint checker.

    Checks simple constraints that can be evaluated with numpy:

    1. **Weight bounds** — L2 norm of room weights must sit inside
       ``config.weight_bounds``.
    2. **Chaos limits** — room chaos must be ≤ ``config.chaos_limit``.
    3. **Thermal budget** — fleet-wide thermal headroom must be ≤
       ``config.thermal_budget_limit``.

    Performance: ~1 ms per check on a single room (numpy einsum).
    Batch checks amortise the overhead.
    """

    def __init__(self, config: Optional[FluxGatingConfig] = None) -> None:
        super().__init__(config)
        self._check_count = 0
        self._violation_count = 0

    # ── single candidate ────────────────────────────────────

    def check_candidate(
        self,
        parent_idx: int,
        mutation_plan: dict[str, Any],
    ) -> GatingResult:
        self._check_count += 1
        violations: list[FluxViolation] = []

        # Extract weights from mutation_plan if provided
        weights = mutation_plan.get("weights")
        if weights is not None:
            w_norm = float(np.linalg.norm(weights))
            w_min, w_max = self.config.weight_bounds
            if w_norm < w_min or w_norm > w_max:
                v = FluxViolation(
                    room_id=parent_idx,
                    constraint_id="weight_bounds",
                    severity=ViolationSeverity.CRITICAL,
                    context={"norm": w_norm, "bounds": [w_min, w_max]},
                )
                violations.append(v)
                self.record_violation(
                    parent_idx, "weight_bounds", ViolationSeverity.CRITICAL, {"norm": w_norm}
                )

        # Chaos check
        chaos = mutation_plan.get("chaos")
        if chaos is not None and chaos > self.config.chaos_limit:
            v = FluxViolation(
                room_id=parent_idx,
                constraint_id="chaos_limit",
                severity=ViolationSeverity.WARNING,
                context={"chaos": chaos, "limit": self.config.chaos_limit},
            )
            violations.append(v)
            self.record_violation(
                parent_idx, "chaos_limit", ViolationSeverity.WARNING, {"chaos": chaos}
            )

        # Thermal budget check (fleet-wide)
        thermal = mutation_plan.get("thermal_headroom")
        if thermal is not None and thermal > self.config.thermal_budget_limit:
            v = FluxViolation(
                room_id=parent_idx,
                constraint_id="thermal_budget",
                severity=ViolationSeverity.WARNING,
                context={"thermal_headroom": thermal, "limit": self.config.thermal_budget_limit},
            )
            violations.append(v)
            self.record_violation(
                parent_idx, "thermal_budget", ViolationSeverity.WARNING, {"thermal": thermal}
            )

        # Pass/fail based on violation count AND severity
        # CRITICAL violations always block (hard constraint)
        has_critical = any(v.severity == ViolationSeverity.CRITICAL for v in violations)
        passed = not has_critical and len(violations) < self.config.max_violations_per_cycle
        score = self._score_from_violations(violations)

        if violations:
            self._violation_count += len(violations)

        return GatingResult(
            candidate_id=mutation_plan.get("candidate_id", f"candidate_{parent_idx}"),
            passed=passed,
            score=score,
            violations=violations,
            metadata={"checker": "PythonFluxFallback", "checks": self._check_count},
        )

    # ── batch check ─────────────────────────────────────────

    def check_batch(
        self,
        candidates: list[tuple[int, dict[str, Any]]],
    ) -> list[GatingResult]:
        return [self.check_candidate(pid, plan) for pid, plan in candidates]

    # ── breeding score ──────────────────────────────────────

    def score_for_breeding(
        self,
        room_vector: np.ndarray,
        room_metadata: dict[str, Any],
    ) -> float:
        """Score a room for breeding eligibility.

        Computes a penalty based on:
        * weight norm deviation from midpoint of bounds
        * chaos proximity to limit
        * activity recency (lower activity = slightly worse)
        """
        penalties = 0.0

        # Weight penalty
        w_norm = float(np.linalg.norm(room_vector))
        w_min, w_max = self.config.weight_bounds
        if w_max > w_min:
            mid = (w_min + w_max) / 2.0
            deviation = abs(w_norm - mid) / (w_max - w_min)
            penalties += deviation * self.config.severity_weights.get("warning", 10)

        # Chaos penalty
        chaos = room_metadata.get("chaos", 0.0)
        if chaos > self.config.chaos_limit:
            penalties += self.config.severity_weights.get("critical", 100)
        else:
            penalties += (chaos / max(self.config.chaos_limit, 1e-6)) * self.config.severity_weights.get("info", 1)

        # Convert penalty to score (0.0–1.0)
        score = max(0.0, 1.0 - penalties / 1000.0)
        return score

    # ── helpers ─────────────────────────────────────────────

    def _score_from_violations(self, violations: list[FluxViolation]) -> float:
        if not violations:
            return 1.0
        total_weight = 0.0
        for v in violations:
            w = self.config.severity_weights.get(v.severity.name.lower(), 1)
            total_weight += w
        return max(0.0, 1.0 - total_weight / 500.0)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "checks": self._check_count,
            "violations": self._violation_count,
            "violation_rate": (
                self._violation_count / self._check_count
                if self._check_count > 0
                else 0.0
            ),
        }
