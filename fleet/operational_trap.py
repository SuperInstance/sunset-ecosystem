"""Operational Trap — fleet health monitoring and alerting base class.

Provides a unified trap framework for thermal overcommit, FLUX constraint
violations, and agent process crashes.  Designed to be run by FleetConductor
on every beat, attached to BreederDaemonV2 for FLUX gating, and wired into
MeshVectorGossip for thermal routing decisions.

Reference: docs/OPERATIONAL_TRAP.md
"""

from __future__ import annotations

__all__ = [
    "TrapSeverity",
    "TrapResult",
    "OperationalTrap",
    "TrapRegistry",
    "ThermalTrap",
    "FluxViolationTrap",
    "AgentCrashTrap",
    "TrapDashboard",
]

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── severity model ──────────────────────────────────────


class TrapSeverity(Enum):
    """Operational trap severity levels."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


# ── data structures ─────────────────────────────────────


@dataclass(frozen=True)
class TrapResult:
    """Result of a single trap check.

    Attributes
    ----------
    condition:
        Short identifier for the condition that fired (e.g. "thermal_overcommit").
    severity:
        ``TrapSeverity`` of the detected condition.
    message:
        Human-readable description.
    metadata:
        Arbitrary key/value context (temperatures, violation counts, PIDs, etc.).
    timestamp:
        Unix timestamp when the result was produced.
    """

    condition: str
    severity: TrapSeverity
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ── base trap ───────────────────────────────────────────


class OperationalTrap(ABC):
    """Abstract base for all fleet health traps.

    Subclasses implement ``check()`` and may override ``escalate()`` / ``notify()``
    for custom routing.  Rate limiting is provided at the base level to prevent
    alert storms.
    """

    def __init__(
        self,
        name: str,
        notify_channels: list[str] | None = None,
        rate_limit_interval: float = 60.0,
    ) -> None:
        self.name = name
        self.notify_channels = notify_channels or ["log"]
        self.rate_limit_interval = rate_limit_interval
        self._rate_limit_store: dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_result: TrapResult | None = None
        self._total_checks: int = 0
        self._total_fired: int = 0

    # ── public API ──────────────────────────────────────────

    @abstractmethod
    def check(self) -> TrapResult | None:
        """Run the health check.

        Returns ``TrapResult`` if a condition is detected, otherwise ``None``.
        """
        raise NotImplementedError

    def escalate(self, result: TrapResult) -> None:
        """Route *result* to the correct handler based on severity.

        Default behaviour:
        - INFO    → log at INFO level
        - WARNING → log at WARNING level + notify channels
        - CRITICAL → log at CRITICAL level + notify channels
        """
        if result.severity == TrapSeverity.INFO:
            logger.info("[%s] %s — %s", self.name, result.condition, result.message)
        elif result.severity == TrapSeverity.WARNING:
            logger.warning("[%s] %s — %s", self.name, result.condition, result.message)
            self.notify(result)
        elif result.severity == TrapSeverity.CRITICAL:
            logger.critical("[%s] %s — %s", self.name, result.condition, result.message)
            self.notify(result)

    def notify(self, result: TrapResult) -> None:
        """Send the alert through configured channels.

        Channels are strings from ``notify_channels``:
        - ``"log"``     — already handled by ``escalate()``; this is a no-op.
        - ``"callback"`` — calls ``self._callback`` if set.
        - ``"a2a"``     — calls ``self._a2a_callback`` if set.
        """
        for channel in self.notify_channels:
            if channel == "callback" and hasattr(self, "_callback"):
                self._callback(result)
            elif channel == "a2a" and hasattr(self, "_a2a_callback"):
                self._a2a_callback(result)

    def rate_limit(self, key: str, interval: float | None = None) -> bool:
        """Return ``True`` if the alert for *key* should be suppressed.

        Uses a per-trap in-memory store keyed by ``(trap_name, key)``.
        Thread-safe.
        """
        interval = interval if interval is not None else self.rate_limit_interval
        with self._lock:
            now = time.monotonic()
            last = self._rate_limit_store.get(key)
            if last is not None and (now - last) < interval:
                return True  # suppressed
            self._rate_limit_store[key] = now
            return False

    def run(self) -> TrapResult | None:
        """Execute ``check()`` and handle escalation / rate-limiting.

        Returns the ``TrapResult`` if the trap fired (and was not rate-limited),
        otherwise ``None``.
        """
        self._total_checks += 1
        result = self.check()
        if result is None:
            return None

        self._total_fired += 1
        self._last_result = result

        if self.rate_limit(result.condition):
            logger.debug(
                "[%s] %s suppressed by rate limit", self.name, result.condition
            )
            return None

        self.escalate(result)
        return result

    def get_status(self) -> dict[str, Any]:
        """Snapshot of trap state. Thread-safe."""
        with self._lock:
            return {
                "name": self.name,
                "total_checks": self._total_checks,
                "total_fired": self._total_fired,
                "last_result": self._last_result,
                "rate_limit_store": dict(self._rate_limit_store),
            }

    def set_callback(self, callback: Callable[[TrapResult], None]) -> None:
        """Attach a Python callable for ``"callback"`` channel."""
        self._callback = callback

    def set_a2a_callback(self, callback: Callable[[TrapResult], None]) -> None:
        """Attach a callable for ``"a2a"`` channel (e.g. A2A message send)."""
        self._a2a_callback = callback


# ── built-in traps ──────────────────────────────────────


class ThermalTrap(OperationalTrap):
    """Detects thermal overcommit on compute devices.

    Monitors a ``ThermalBudget`` instance.  Fires when any device's
    utilization exceeds ``threshold`` (default 95 %) or when an explicit
    overcommit is detected (current > max).
    """

    def __init__(
        self,
        budget: Any,
        threshold: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="thermal", **kwargs)
        self.budget = budget
        self.threshold = threshold

    def check(self) -> TrapResult | None:
        # Lazy import to avoid circular dependency at module load time
        from swarm.thermal import DeviceType, ThermalBudget

        if not isinstance(self.budget, ThermalBudget):
            return None

        overcommitted: list[str] = []
        max_util = 0.0
        for dt in DeviceType:
            db = self.budget._devices.get(dt)
            if db is None:
                continue
            util = db.utilization
            max_util = max(max_util, util)
            if db.current_agents > db.max_agents or util > self.threshold:
                overcommitted.append(
                    f"{dt.value}: {db.current_agents}/{db.max_agents} ({util:.0%})"
                )

        if not overcommitted:
            return None

        severity = TrapSeverity.CRITICAL if max_util > 1.0 else TrapSeverity.WARNING
        return TrapResult(
            condition="thermal_overcommit",
            severity=severity,
            message=f"Thermal overcommit on {len(overcommitted)} device(s): "
            f"{', '.join(overcommitted)}",
            metadata={
                "devices": overcommitted,
                "max_utilization": max_util,
                "threshold": self.threshold,
            },
        )


class FluxViolationTrap(OperationalTrap):
    """Detects FLUX constraint violations from recent check results.

    Accepts either a ``get_recent_results`` callable that returns a list of
    ``FluxCheckResult``‑like objects, or a *checker* with an optional ``_wal``
    attribute for backward compatibility.
    """

    def __init__(
        self,
        checker: Any | None = None,
        get_recent_results: Callable[[], list[Any]] | None = None,
        score_threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="flux_violation", **kwargs)
        self.checker = checker
        self.get_recent_results = get_recent_results
        self.score_threshold = score_threshold
        self._last_results: list[Any] = []

    def check(self) -> TrapResult | None:
        from swarm.flux_gating import FluxGatingChecker

        results: list[Any] = []

        # Prefer the explicit callable
        if self.get_recent_results is not None:
            try:
                results = self.get_recent_results()
            except Exception:
                logger.exception("get_recent_results failed")
                return None
        # Backward compat: checker with a _wal attribute
        elif (
            self.checker is not None
            and hasattr(self.checker, "_wal")
            and self.checker._wal is not None
        ):
            try:
                recent = (
                    self.checker._wal.all()
                    if hasattr(self.checker._wal, "all")
                    else list(self.checker._wal)
                )
                now = time.time()
                results = [r for r in recent if now - r.get("timestamp", 0) < 60.0]
            except Exception:
                return None

        if not results:
            return None

        # Categorise by severity
        critical = 0
        warning = 0
        for r in results:
            if isinstance(r, dict):
                sev = r.get("severity", "").lower()
                if sev == "critical":
                    critical += 1
                elif sev == "warning":
                    warning += 1
            else:
                # FluxCheckResult-like object
                score = getattr(r, "score", 0.0)
                passed = getattr(r, "passed", True)
                if not passed:
                    if score > 0.7:
                        critical += 1
                    else:
                        warning += 1

        if critical == 0 and warning == 0:
            return None

        severity = TrapSeverity.CRITICAL if critical > 0 else TrapSeverity.WARNING
        return TrapResult(
            condition="flux_constraint_breach",
            severity=severity,
            message=f"FLUX breach: {critical} critical, {warning} warning violations "
            f"in last 60s ({len(results)} total records)",
            metadata={
                "critical_count": critical,
                "warning_count": warning,
                "window_size": len(results),
                "score_threshold": self.score_threshold,
            },
        )


class AgentCrashTrap(OperationalTrap):
    """Monitors agent process health.

    Accepts a ``get_agent_pids`` callable that should return a mapping of
    ``agent_id → pid``.  If any PID is missing (``None`` or ``0``) or the
    agent is present in ``expected_agents`` but absent from the mapping,
    the trap fires.
    """

    def __init__(
        self,
        get_agent_pids: Callable[[], dict[str, int | None]],
        expected_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="agent_crash", **kwargs)
        self.get_agent_pids = get_agent_pids
        self.expected_agents = set(expected_agents or [])

    def check(self) -> TrapResult | None:
        mapping = self.get_agent_pids()
        if not mapping:
            return None

        missing: list[str] = []
        for agent_id in self.expected_agents:
            pid = mapping.get(agent_id)
            if pid is None or pid == 0:
                missing.append(agent_id)

        # Also flag any agent explicitly reported with a dead PID
        for agent_id, pid in mapping.items():
            if pid is None or pid == 0:
                if agent_id not in missing:
                    missing.append(agent_id)

        if not missing:
            return None

        return TrapResult(
            condition="agent_crash",
            severity=TrapSeverity.CRITICAL,
            message=f"{len(missing)} agent(s) missing or crashed: {missing}",
            metadata={
                "missing_agents": missing,
                "expected_count": len(self.expected_agents),
            },
        )


# ── registry ────────────────────────────────────────────


class TrapRegistry:
    """Collects and runs a set of operational traps.

    Intended to be invoked by ``FleetConductor`` on every beat.
    """

    def __init__(self) -> None:
        self._traps: list[OperationalTrap] = []
        self._lock = threading.Lock()
        self._results: list[TrapResult] = []

    def register(self, trap: OperationalTrap) -> None:
        """Add a trap to the registry. Thread-safe."""
        with self._lock:
            self._traps.append(trap)

    def unregister(self, trap: OperationalTrap) -> None:
        """Remove a trap from the registry. Thread-safe."""
        with self._lock:
            try:
                self._traps.remove(trap)
            except ValueError:
                pass

    def run_all(self) -> list[TrapResult]:
        """Execute every registered trap and collect fired results.

        Returns the list of non-suppressed ``TrapResult`` objects.
        """
        fired: list[TrapResult] = []
        with self._lock:
            traps = list(self._traps)
        for trap in traps:
            result = trap.run()
            if result is not None:
                fired.append(result)
        self._results = fired
        return fired

    def get_status(self) -> dict[str, Any]:
        """Return status for every registered trap."""
        with self._lock:
            traps = list(self._traps)
        return {
            "trap_count": len(traps),
            "last_run_results": [r for r in self._results],
            "traps": [t.get_status() for t in traps],
        }


# ── dashboard ───────────────────────────────────────────


class TrapDashboard:
    """Unified view of all trap states.

    Wraps a ``TrapRegistry`` (or any object with ``get_status()``) and
    presents a flattened snapshot suitable for SSE streams or health
    check endpoints.
    """

    def __init__(self, registry: TrapRegistry) -> None:
        self.registry = registry

    def get_status(self) -> dict[str, Any]:
        """Flattened status suitable for dashboards.

        Returns a dict with:
        - ``summary``: counts of traps, checks, fired, and criticals
        - ``traps``: per-trap status list
        - ``alerts``: only the fired results from the last run
        - ``timestamp``: when the snapshot was taken
        """
        raw = self.registry.get_status()
        now = time.time()
        traps = raw.get("traps", [])
        results = raw.get("last_run_results", [])

        total_checks = sum(t.get("total_checks", 0) for t in traps)
        total_fired = sum(t.get("total_fired", 0) for t in traps)
        critical_count = sum(1 for r in results if r.severity == TrapSeverity.CRITICAL)

        return {
            "summary": {
                "trap_count": raw.get("trap_count", 0),
                "total_checks": total_checks,
                "total_fired": total_fired,
                "critical_count": critical_count,
                "warning_count": sum(
                    1 for r in results if r.severity == TrapSeverity.WARNING
                ),
                "info_count": sum(
                    1 for r in results if r.severity == TrapSeverity.INFO
                ),
            },
            "traps": traps,
            "alerts": [
                {
                    "condition": r.condition,
                    "severity": r.severity.name,
                    "message": r.message,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp,
                }
                for r in results
            ],
            "timestamp": now,
        }
