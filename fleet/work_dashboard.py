"""work_dashboard.py — Central fleet work status dashboard.

Aggregates real-time status from all fleet subsystems:
- Subagent conductor (queue depth, in-flight, health)
- Kimicode bridge (cache metrics, latency)
- Breeding (generation, best fitness, species count)
- Consensus (BFT nodes, quorum status)
- Mesh (node count, topology health)
- SSE dashboard (event throughput, subscriber count)

Provides a unified report() method that returns a snapshot of the
entire fleet's operational state. Designed to be polled by external
monitoring and the SSE stream dashboard.
"""
from __future__ import annotations

__all__ = [
    "FleetWorkDashboard",
    "SubsystemStatus",
    "FleetHealthLevel",
]

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class FleetHealthLevel(enum.IntEnum):
    HEALTHY = 4      # All green, operating normally
    DEGRADED = 3     # Minor issues, self-healing
    WARNING = 2      # Significant issues, attention needed
    CRITICAL = 1     # Multiple failures, manual intervention
    DOWN = 0         # Complete outage


@dataclass
class SubsystemStatus:
    """Status of a single fleet subsystem."""
    name: str
    healthy: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    last_update: float = 0.0
    error: str = ""


class FleetWorkDashboard:
    """Central dashboard aggregating all fleet subsystem status.

    Usage:
        dashboard = FleetWorkDashboard()
        dashboard.register("breeding", breeder.report)
        dashboard.register("consensus", consensus.report)
        snapshot = dashboard.snapshot()
        health = dashboard.health_level()
    """

    def __init__(self) -> None:
        self._subsystems: dict[str, callable] = {}  # name -> report_fn
        self._overrides: dict[str, SubsystemStatus] = {}  # manual overrides
        self._history: list[dict[str, Any]] = []  # historical snapshots
        self._max_history: int = 1000

    def register(self, name: str, report_fn: callable) -> None:
        """Register a subsystem report function."""
        self._subsystems[name] = report_fn
        logger.info(f"Registered subsystem: {name}")

    def unregister(self, name: str) -> None:
        self._subsystems.pop(name, None)
        self._overrides.pop(name, None)

    def set_override(self, status: SubsystemStatus) -> None:
        """Manually set a subsystem status (overrides report_fn)."""
        self._overrides[status.name] = status

    def clear_override(self, name: str) -> None:
        self._overrides.pop(name, None)

    # ── snapshot ──────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Collect current status from all registered subsystems."""
        import time
        result: dict[str, Any] = {"timestamp": time.time(), "subsystems": {}}

        for name, report_fn in self._subsystems.items():
            if name in self._overrides:
                status = self._overrides[name]
            else:
                try:
                    metrics = report_fn()
                    status = SubsystemStatus(
                        name=name,
                        healthy=True,
                        metrics=metrics if isinstance(metrics, dict) else {"data": metrics},
                        last_update=time.time(),
                    )
                except Exception as e:
                    status = SubsystemStatus(
                        name=name,
                        healthy=False,
                        last_update=time.time(),
                        error=str(e),
                    )
            result["subsystems"][name] = {
                "healthy": status.healthy,
                "metrics": status.metrics,
                "last_update": status.last_update,
                "error": status.error,
            }

        # Aggregate health
        result["health_level"] = self._compute_health_level(result["subsystems"]).name
        result["overall_healthy"] = all(s["healthy"] for s in result["subsystems"].values())
        result["subsystem_count"] = len(result["subsystems"])
        result["healthy_count"] = sum(1 for s in result["subsystems"].values() if s["healthy"])

        # Store history
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return result

    def _compute_health_level(self, subsystems: dict[str, Any]) -> FleetHealthLevel:
        if not subsystems:
            return FleetHealthLevel.DEGRADED

        total = len(subsystems)
        healthy = sum(1 for s in subsystems.values() if s["healthy"])
        ratio = healthy / total

        if ratio == 1.0:
            return FleetHealthLevel.HEALTHY
        elif ratio >= 0.8:
            return FleetHealthLevel.DEGRADED
        elif ratio >= 0.5:
            return FleetHealthLevel.WARNING
        elif ratio > 0.0:
            return FleetHealthLevel.CRITICAL
        else:
            return FleetHealthLevel.DOWN

    # ── analysis ────────────────────────────────────────

    def trend(self, metric_path: list[str], window: int = 10) -> list[Any]:
        """Extract a metric time series from history.

        metric_path: e.g. ["breeding", "metrics", "best_fitness"]
        """
        values = []
        for snap in self._history[-window:]:
            val = snap
            for key in metric_path:
                if isinstance(val, dict) and key in val:
                    val = val[key]
                else:
                    val = None
                    break
            values.append(val)
        return values

    def delta(self, metric_path: list[str]) -> Any:
        """Change in metric between last two snapshots."""
        series = self.trend(metric_path, window=2)
        if len(series) < 2:
            return None
        a, b = series[-2:]
        if a is None or b is None:
            return None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return b - a
        return {"from": a, "to": b}

    # ── alerts ────────────────────────────────────────────

    def alerts(self) -> list[dict[str, Any]]:
        """Generate alerts for unhealthy subsystems."""
        import time
        snap = self.snapshot()
        alerts = []
        for name, data in snap["subsystems"].items():
            if not data["healthy"]:
                alerts.append({
                    "severity": "error",
                    "subsystem": name,
                    "message": data.get("error", "Subsystem unhealthy"),
                    "timestamp": time.time(),
                })
        return alerts

    # ── history ─────────────────────────────────────────

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        """One-line summary of fleet state."""
        snap = self.snapshot()
        return {
            "health": snap["health_level"],
            "subsystems": snap["subsystem_count"],
            "healthy": snap["healthy_count"],
            "alerts": len(self.alerts()),
            "snapshots_stored": len(self._history),
        }
