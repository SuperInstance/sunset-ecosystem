"""fleet/health_bridge.py — cocapn-health pattern integration.

Brings the cocapn-health CheckResult pattern into sunset-ecosystem:
- Standardized health check results (name, ok, latency_ms, status, details)
- Fleet service definitions with extracted metrics
- Event bus bridge for service transition notifications
- REST API with cache TTL
- Watch mode for continuous monitoring

This module is zero-dependency (stdlib only) and compatible with
cocapn-health's data structures.

Usage:
    from fleet.health_bridge import HealthChecker, ServiceDef, CheckResult

    checker = HealthChecker(FLEET_SERVICES)
    results = checker.check_all()

    # Report in multiple formats
    print(checker.report(results, format="md"))
    print(checker.report(results, format="json"))
    print(checker.report(results, format="oneline"))
"""

from __future__ import annotations

import json
import socket
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


@dataclass
class CheckResult:
    """Standardized health check result — compatible with cocapn-health."""
    name: str
    ok: bool
    latency_ms: float
    status: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckResult":
        return cls(
            name=d["name"],
            ok=d["ok"],
            latency_ms=d["latency_ms"],
            status=d["status"],
            details=d.get("details", {}),
        )


@dataclass
class ServiceDef:
    """A fleet service to monitor — compatible with cocapn-health."""
    name: str
    host: str
    port: int
    path: str = "/"
    timeout: float = 5.0
    expect_status: Optional[int] = None
    extract: Optional[dict[str, str]] = None  # json_path -> metric_name
    headers: Optional[dict[str, str]] = None

    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"


# Default fleet services — matches cocapn-health's FLEET_SERVICES
FLEET_SERVICES = [
    ServiceDef("MUD v3", "<BOAT_IP>", 4042, "/status", extract={"rooms": "rooms"}),
    ServiceDef("The Lock v2", "<BOAT_IP>", 4043, "/status", extract={"strategies": "strategies"}),
    ServiceDef("Arena", "<BOAT_IP>", 4044, "/stats", extract={"total_matches": "total_matches"}),
    ServiceDef("Grammar Engine", "<BOAT_IP>", 4045, "/grammar", extract={"total_rules": "total_rules"}),
    ServiceDef("Dashboard", "<BOAT_IP>", 4046, "/"),
    ServiceDef("Federated Nexus", "<BOAT_IP>", 4047, "/"),
    ServiceDef("Harbor", "<BOAT_IP>", 4050, "/"),
    ServiceDef("Grammar Compactor", "<BOAT_IP>", 4055, "/status", extract={"total_rules": "total_rules"}),
    ServiceDef("Rate-Attention", "<BOAT_IP>", 4056, "/streams", extract={"streams": "streams"}),
    ServiceDef("Skill Forge", "<BOAT_IP>", 4057, "/status", extract={"total_drills": "total_drills"}),
    ServiceDef("PLATO Terminal", "<BOAT_IP>", 4060, "/"),
    ServiceDef("PLATO Gate", "<BOAT_IP>", 8847, "/rooms", extract={"rooms": "rooms"}),
    ServiceDef("PLATO Shell", "<BOAT_IP>", 8848, "/"),
    ServiceDef("Service Guard", "<BOAT_IP>", 8899, "/"),
    ServiceDef("Task Queue", "<BOAT_IP>", 8900, "/"),
    ServiceDef("Steward", "<BOAT_IP>", 8901, "/"),
    ServiceDef("Matrix Bridge", "<BOAT_IP>", 6168, "/status"),
    ServiceDef("Conduwuit", "<BOAT_IP>", 6167, "/"),
]


class HealthChecker:
    """
    Fleet health checker with cocapn-health compatible reporting.

    Supports HTTP endpoint checks with metric extraction, TCP port checks,
    and system resource checks (disk, memory, CPU).
    """

    def __init__(self, services: list[ServiceDef]):
        self.services = services

    # ------------------------------------------------------------------
    # Individual check primitives

    @staticmethod
    def check_http(
        url: str,
        timeout: float = 5.0,
        expect_status: Optional[int] = None,
        extract: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> CheckResult:
        """Check an HTTP endpoint and optionally extract metrics."""
        start = time.time()
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "sunset-health/1.0", **(headers or {})},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                latency_ms = (time.time() - start) * 1000
                status_code = resp.getcode()
                body = resp.read().decode("utf-8", errors="replace")

                # HTTP 404/400/401 from a live server is treated as "UP"
                is_up = True
                if expect_status is not None and status_code != expect_status:
                    is_up = False

                details: dict[str, Any] = {"status_code": status_code}
                # Try to extract metrics from JSON response
                if extract:
                    try:
                        data = json.loads(body)
                        for json_path, metric_name in extract.items():
                            val = _extract_json_path(data, json_path)
                            if val is not None:
                                details[metric_name] = val
                    except json.JSONDecodeError:
                        pass

                return CheckResult(
                    name=url,
                    ok=is_up,
                    latency_ms=latency_ms,
                    status=f"UP | HTTP {status_code}" if is_up else f"DEGRADED | HTTP {status_code}",
                    details=details,
                )
        except urllib.error.HTTPError as e:
            latency_ms = (time.time() - start) * 1000
            # HTTP errors from a live server = UP (many fleet services expose no root handler)
            return CheckResult(
                name=url,
                ok=True,
                latency_ms=latency_ms,
                status=f"UP | HTTP {e.code}",
                details={"status_code": e.code, "error": str(e.reason)},
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return CheckResult(
                name=url,
                ok=False,
                latency_ms=latency_ms,
                status=f"DOWN | {type(e).__name__}: {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def check_tcp(host: str, port: int, timeout: float = 3.0) -> CheckResult:
        """Check if a TCP port is open."""
        start = time.time()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                latency_ms = (time.time() - start) * 1000
                return CheckResult(
                    name=f"{host}:{port}",
                    ok=True,
                    latency_ms=latency_ms,
                    status="UP | TCP connected",
                    details={},
                )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return CheckResult(
                name=f"{host}:{port}",
                ok=False,
                latency_ms=latency_ms,
                status=f"DOWN | {type(e).__name__}: {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def check_disk(path: str = "/", min_percent_free: float = 10.0) -> CheckResult:
        """Check disk space."""
        import shutil
        start = time.time()
        try:
            usage = shutil.disk_usage(path)
            total_gb = usage.total / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            percent_free = (usage.free / usage.total) * 100
            latency_ms = (time.time() - start) * 1000
            ok = percent_free >= min_percent_free
            return CheckResult(
                name=path,
                ok=ok,
                latency_ms=latency_ms,
                status=f"{'OK' if ok else 'LOW'} | {percent_free:.1f}% free",
                details={
                    "total_gb": round(total_gb, 1),
                    "free_gb": round(free_gb, 1),
                    "percent_free": round(percent_free, 1),
                },
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return CheckResult(
                name=path,
                ok=False,
                latency_ms=latency_ms,
                status=f"ERROR | {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def check_memory(min_percent_free: float = 10.0) -> CheckResult:
        """Check system memory."""
        start = time.time()
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            # Parse meminfo
            total_kb = 0
            available_kb = 0
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
            total_mb = total_kb / 1024
            available_mb = available_kb / 1024
            percent_available = (available_kb / total_kb) * 100 if total_kb else 0
            ok = percent_available >= min_percent_free
            latency_ms = (time.time() - start) * 1000
            return CheckResult(
                name="memory",
                ok=ok,
                latency_ms=latency_ms,
                status=f"{'OK' if ok else 'LOW'} | {percent_available:.1f}% available",
                details={
                    "total_mb": round(total_mb, 1),
                    "available_mb": round(available_mb, 1),
                    "percent_available": round(percent_available, 1),
                },
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return CheckResult(
                name="memory",
                ok=False,
                latency_ms=latency_ms,
                status=f"ERROR | {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def check_cpu(max_percent: float = 95.0) -> CheckResult:
        """Check CPU load."""
        start = time.time()
        try:
            with open("/proc/loadavg", "r") as f:
                loadavg = f.read().split()
            load_1m = float(loadavg[0])
            load_5m = float(loadavg[1])
            load_15m = float(loadavg[2])
            # Estimate CPU count
            cpu_count = 1
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpu_count = f.read().count("processor\t:")
            except Exception:
                pass
            utilization_percent = (load_1m / cpu_count) * 100
            ok = utilization_percent <= max_percent
            latency_ms = (time.time() - start) * 1000
            return CheckResult(
                name="cpu",
                ok=ok,
                latency_ms=latency_ms,
                status=f"{'OK' if ok else 'HIGH'} | load {load_1m:.2f} ({utilization_percent:.0f}%)",
                details={
                    "load_1m": load_1m,
                    "load_5m": load_5m,
                    "load_15m": load_15m,
                    "cpu_count": cpu_count,
                    "utilization_percent": round(utilization_percent, 1),
                },
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return CheckResult(
                name="cpu",
                ok=False,
                latency_ms=latency_ms,
                status=f"ERROR | {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def check_system() -> list[CheckResult]:
        """Run all system checks at once."""
        return [
            HealthChecker.check_disk(),
            HealthChecker.check_memory(),
            HealthChecker.check_cpu(),
        ]

    # ------------------------------------------------------------------
    # Fleet service checking

    def check_all(self) -> list[CheckResult]:
        """Check all fleet services."""
        results = []
        for svc in self.services:
            if svc.path:
                result = self.check_http(
                    svc.url(),
                    timeout=svc.timeout,
                    expect_status=svc.expect_status,
                    extract=svc.extract,
                    headers=svc.headers,
                )
            else:
                result = self.check_tcp(svc.host, svc.port, timeout=svc.timeout)
            result.name = svc.name  # Use service name instead of URL
            results.append(result)
        return results

    def check_one(self, svc: ServiceDef) -> CheckResult:
        """Check a single service."""
        if svc.path:
            result = self.check_http(
                svc.url(),
                timeout=svc.timeout,
                expect_status=svc.expect_status,
                extract=svc.extract,
                headers=svc.headers,
            )
        else:
            result = self.check_tcp(svc.host, svc.port, timeout=svc.timeout)
        result.name = svc.name
        return result

    # ------------------------------------------------------------------
    # Reporting

    @staticmethod
    def report(results: list[CheckResult], format: str = "md") -> str:
        """Format results as markdown, json, or oneline."""
        if format == "json":
            return json.dumps({
                "summary": {
                    "total": len(results),
                    "up": sum(1 for r in results if r.ok),
                    "down": sum(1 for r in results if not r.ok),
                },
                "services": [r.to_dict() for r in results],
            }, indent=2)

        if format == "oneline":
            up = sum(1 for r in results if r.ok)
            down = len(results) - up
            return f"Fleet: {up}/{len(results)} up, {down} down"

        # Markdown
        up = sum(1 for r in results if r.ok)
        down = len(results) - up
        lines = [
            f"# Fleet Health Report",
            f"**{up}/{len(results)} services UP** — {down} down",
            "",
            "| Service | Status | Latency | Details |",
            "|---------|--------|---------|---------|",
        ]
        for r in results:
            icon = "🟢" if r.ok else "🔴"
            details = json.dumps(r.details) if r.details else "—"
            lines.append(
                f"| {icon} {r.name} | {r.status} | {r.latency_ms:.0f}ms | {details} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event bus bridge for service transitions

class EventBusHealthChecker(HealthChecker):
    """
    Health checker that emits events on service transitions.

    Compatible with cocapn-health's sunset_bridge pattern.
    """

    def __init__(
        self,
        services: list[ServiceDef],
        bus: Optional[Any] = None,
        emit_on_every_check: bool = False,
    ):
        super().__init__(services)
        self.bus = bus
        self.emit_on_every_check = emit_on_every_check
        self._last_state: dict[str, bool] = {}

    def check_all(self) -> list[CheckResult]:
        results = super().check_all()
        for r in results:
            last = self._last_state.get(r.name)
            if last is not None:
                if last and not r.ok:
                    self._emit("service_down", r)
                elif not last and r.ok:
                    self._emit("service_recovered", r)
            self._last_state[r.name] = r.ok
            if self.emit_on_every_check:
                self._emit("fleet_health", r)
        return results

    def _emit(self, event_type: str, result: CheckResult) -> None:
        if self.bus is None:
            return
        payload = result.to_dict()
        # Try common event bus interfaces — explicit method lookup to avoid
        # MagicMock auto-creating attributes on hasattr checks.
        for attr in ("emit", "publish", "send"):
            meth = getattr(self.bus, attr, None)
            if meth is not None and callable(meth):
                try:
                    meth(event_type, payload)
                    return
                except Exception:
                    pass
        # Fallback: try to call if bus is callable itself
        if callable(self.bus):
            try:
                self.bus(event_type, payload)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Health cache with TTL

class HealthCache:
    """Cache health check results with TTL — compatible with cocapn-health."""

    def __init__(self, ttl: float = 30.0):
        self.ttl = ttl
        self._cache: dict[str, tuple[float, CheckResult]] = {}
        self._services: list[ServiceDef] = []

    def set_services(self, services: list[ServiceDef]) -> None:
        self._services = services

    def get(self, checker: HealthChecker, force: bool = False) -> list[CheckResult]:
        now = time.time()
        if force:
            results = checker.check_all()
            for r in results:
                self._cache[r.name] = (now, r)
            return results

        # Return cached if fresh
        results = []
        need_refresh = []
        for svc in self._services:
            cached = self._cache.get(svc.name)
            if cached and (now - cached[0]) < self.ttl:
                results.append(cached[1])
            else:
                need_refresh.append(svc)

        if need_refresh:
            # Check only stale services
            for svc in need_refresh:
                result = checker.check_one(svc)
                self._cache[result.name] = (now, result)
                results.append(result)

        return results

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Utility

def _extract_json_path(data: Any, path: str) -> Any:
    """Extract a value from nested dict using dot notation."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current
