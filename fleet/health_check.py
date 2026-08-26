"""fleet/health_check.py — Lightweight fleet service health checker.

Cross-pollinated from cocapn-health. Extended for sunset-ecosystem
service endpoints, thermal-aware health scoring, and FLUX gating
integration.

Usage
-----
    from fleet.health_check import FleetHealthChecker, ServiceDef

    checker = FleetHealthChecker()
    results = checker.check_all()
    print(checker.report(results, format="markdown"))
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fleet.config import get_config


@dataclass
class ServiceDef:
    """Service definition for health checking."""

    name: str
    host: str
    port: int
    path: str = "/"
    method: str = "GET"
    timeout: float = 5.0
    expect_status: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    extract: Optional[Dict[str, str]] = None


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    ok: bool
    latency_ms: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def pressure_penalty(self) -> float:
        """Return 0.0–1.0 penalty based on latency (slow = higher)."""
        if self.latency_ms < 100:
            return 0.0
        if self.latency_ms < 500:
            return 0.1
        if self.latency_ms < 1000:
            return 0.3
        return 0.5


class FleetHealthChecker:
    """Check fleet services and produce reports."""

    def __init__(self, services: Optional[List[ServiceDef]] = None):
        self.services = services or self._default_services()

    @staticmethod
    def _default_services() -> List[ServiceDef]:
        """Build default service list from config."""
        cfg = get_config()
        host = os.environ.get("SUNSET_HOST", None)
        services = []
        for s in cfg.health_services():
            svc = ServiceDef(
                name=s["name"],
                host=host or s["host"],
                port=s["port"],
                path=s.get("path", "/status"),
                extract=s.get("extract"),
            )
            services.append(svc)
        return services

    def check_one(self, svc: ServiceDef) -> CheckResult:
        """Check a single service."""
        url = f"http://{svc.host}:{svc.port}{svc.path}"
        start = time.time()

        try:
            req = urllib.request.Request(url, method=svc.method, headers=svc.headers)
            with urllib.request.urlopen(req, timeout=svc.timeout) as resp:
                latency = (time.time() - start) * 1000
                status_code = resp.status
                body = resp.read(2048).decode("utf-8", errors="replace")

                details = {"status_code": status_code, "latency_ms": round(latency, 1)}
                try:
                    data = json.loads(body)
                    if svc.extract:
                        for key, path in svc.extract.items():
                            val = data
                            for part in path.split("."):
                                val = (
                                    val.get(part, {}) if isinstance(val, dict) else None
                                )
                            details[key] = val
                    else:
                        for k in [
                            "rooms",
                            "tiles",
                            "total_rules",
                            "total_matches",
                            "total_players",
                            "uptime_seconds",
                            "total_drills",
                            "streams",
                            "agents",
                            "vectors",
                            "proofs",
                        ]:
                            if k in data:
                                details[k] = data[k]
                except json.JSONDecodeError:
                    details["body_preview"] = body[:100]

                if svc.expect_status and status_code != svc.expect_status:
                    return CheckResult(
                        name=svc.name,
                        ok=False,
                        latency_ms=round(latency, 1),
                        status=f"HTTP {status_code} (expected {svc.expect_status})",
                        details=details,
                    )

                return CheckResult(
                    name=svc.name,
                    ok=True,
                    latency_ms=round(latency, 1),
                    status=f"UP | HTTP {status_code}",
                    details=details,
                )

        except urllib.error.HTTPError as e:
            latency = (time.time() - start) * 1000
            if e.code in (404, 400, 401):
                return CheckResult(
                    name=svc.name,
                    ok=True,
                    latency_ms=round(latency, 1),
                    status=f"UP | HTTP {e.code}",
                    details={"status_code": e.code},
                )
            return CheckResult(
                name=svc.name,
                ok=False,
                latency_ms=round(latency, 1),
                status=f"DOWN | HTTP {e.code}",
                details={"status_code": e.code, "error": str(e)},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return CheckResult(
                name=svc.name,
                ok=False,
                latency_ms=round(latency, 1),
                status=f"DOWN | {type(e).__name__}",
                details={"error": str(e)},
            )

    def check_all(self) -> List[CheckResult]:
        """Check all services."""
        return [self.check_one(svc) for svc in self.services]

    @staticmethod
    def report(results: List[CheckResult], format: str = "markdown") -> str:
        """Generate a report string."""
        up = sum(1 for r in results if r.ok)
        down = len(results) - up

        if format == "json":
            return json.dumps(
                {
                    "summary": {"total": len(results), "up": up, "down": down},
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "services": [
                        {
                            "name": r.name,
                            "ok": r.ok,
                            "status": r.status,
                            "latency_ms": r.latency_ms,
                            "details": r.details,
                        }
                        for r in results
                    ],
                },
                indent=2,
                default=str,
            )

        elif format == "markdown":
            lines = [
                "# Fleet Health Report",
                "",
                f"**{up}/{len(results)} services UP** — {down} down",
                "",
                "| Service | Status | Latency | Details |",
                "|---------|--------|---------|---------|",
            ]
            for r in results:
                emoji = "🟢" if r.ok else "🔴"
                details = " | ".join(f"{k}={v}" for k, v in list(r.details.items())[:3])
                lines.append(
                    f"| {emoji} {r.name} | {r.status} | {r.latency_ms:.0f}ms | {details} |"
                )
            return "\n".join(lines)

        elif format == "oneline":
            status = "✅" if down == 0 else f"⚠️ {down} down"
            slow = [r for r in results if r.latency_ms > 1000]
            slow_str = f", {len(slow)} slow" if slow else ""
            return f"Fleet: {up}/{len(results)} up{slow_str} {status}"

        return ""

    def thermal_score(self, results: List[CheckResult]) -> float:
        """Compute fleet thermal pressure from check latencies."""
        if not results:
            return 0.0
        penalties = [r.pressure_penalty() for r in results]
        return sum(penalties) / len(penalties)


# ---------------------------------------------------------------------------
# Health Check System (dependency-chain aware) — added 2026-05-30
# ---------------------------------------------------------------------------

from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """A health check result."""

    name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class HealthCheckSystem:
    """
    Health check system with dependency chains.

    Checks services and their dependencies, reports overall health.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._checks: Dict[str, callable] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._results: Dict[str, HealthCheck] = {}

    def register(
        self, name: str, check_func: callable, dependencies: Optional[List[str]] = None
    ):
        """Register a health check."""
        self._checks[name] = check_func
        self._dependencies[name] = dependencies or []

    def check(self, name: str) -> HealthCheck:
        """Run a single health check."""
        start = time.time()
        try:
            func = self._checks[name]
            result = func()
            if isinstance(result, tuple):
                status, message = result
            else:
                status, message = result, "OK"
            elapsed = (time.time() - start) * 1000
            check = HealthCheck(
                name=name,
                status=HealthStatus(status),
                response_time_ms=elapsed,
                message=message,
                timestamp=time.time(),
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=elapsed,
                message=str(e),
                timestamp=time.time(),
            )
        self._results[name] = check
        return check

    def check_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks in dependency order."""
        visited = set()
        results = {}

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self._dependencies.get(name, []):
                visit(dep)
            results[name] = self.check(name)

        for name in self._checks:
            visit(name)

        self._results = results
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall health status."""
        if not self._results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in self._results.values()]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_dependents(self, name: str) -> List[str]:
        """Get services that depend on a given service."""
        dependents = []
        for service, deps in self._dependencies.items():
            if name in deps:
                dependents.append(service)
        return dependents

    def get_stats(self) -> Dict[str, Any]:
        """Get health check statistics."""
        if not self._results:
            return {"status": "unknown", "checks": 0}

        statuses = [r.status for r in self._results.values()]
        return {
            "overall": self.get_overall_status().value,
            "checks": len(self._results),
            "healthy": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
            "degraded": sum(1 for s in statuses if s == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for s in statuses if s == HealthStatus.UNHEALTHY),
            "avg_response_time_ms": sum(
                r.response_time_ms for r in self._results.values()
            )
            / len(self._results.values())
            if self._results
            else 0.0,
        }

    def export_json(self) -> str:
        """Export health status as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "overall": self.get_overall_status().value,
                "checks": {k: v.to_dict() for k, v in self._results.items()},
                "stats": self.get_stats(),
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
            "stats": self.get_stats(),
        }
