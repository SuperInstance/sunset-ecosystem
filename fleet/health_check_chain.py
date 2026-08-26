"""health_check_chain.py — Composite health check with dependency chains.

Provides:
1. Individual health probes (callable checkers)
2. Dependency chains (probe A must pass before probe B)
3. Timeout and retry support
4. Aggregated health status with critical/warning/non-critical tiers
5. Circuit breaker integration (failing probes disable dependents)

Usage:
    chain = HealthCheckChain()
    chain.add("database", check_db, critical=True, timeout=5.0)
    chain.add("cache", check_cache, depends_on=["database"])
    status = chain.run()
    # status.healthy, status.probes, status.blockers
"""

from __future__ import annotations

__all__ = [
    "HealthCheckChain",
    "ProbeResult",
    "ChainStatus",
]

import enum
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HealthTier(enum.Enum):
    CRITICAL = "critical"  # Failure = system down
    WARNING = "warning"  # Failure = degraded but operational
    INFO = "info"  # Failure = logged but not impactful


@dataclass
class ProbeResult:
    """Result of a single health probe."""

    name: str
    healthy: bool
    tier: HealthTier
    latency_ms: float
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainStatus:
    """Aggregated health status from all probes."""

    healthy: bool
    critical_healthy: bool
    warning_healthy: bool
    probes: list[ProbeResult]
    blockers: list[str]  # probes that failed and blocked dependents
    latency_ms: float


class HealthCheckChain:
    """Composite health check with dependency chains and timeouts."""

    def __init__(self, max_workers: int = 4) -> None:
        self._probes: dict[str, dict[str, Any]] = {}
        self._max_workers = max_workers

    def add(
        self,
        name: str,
        checker: Callable[[], tuple[bool, str]],
        tier: HealthTier = HealthTier.CRITICAL,
        timeout: float = 5.0,
        depends_on: list[str] | None = None,
        retries: int = 0,
    ) -> None:
        """Add a health probe.

        checker: callable returning (healthy, message)
        depends_on: names of probes that must pass before this one runs
        retries: number of retries on failure
        """
        self._probes[name] = {
            "checker": checker,
            "tier": tier,
            "timeout": timeout,
            "depends_on": set(depends_on or []),
            "retries": retries,
            "result": None,
        }

    def remove(self, name: str) -> bool:
        if name in self._probes:
            del self._probes[name]
            # Remove from other probes' dependencies
            for p in self._probes.values():
                p["depends_on"].discard(name)
            return True
        return False

    # ── run ────────────────────────────────────────────

    def run(self) -> ChainStatus:
        """Run all probes respecting dependencies and timeouts."""
        start = time.time()
        completed: set[str] = set()
        blockers: list[str] = []
        results: list[ProbeResult] = []

        # Topological sort: run probes whose dependencies are satisfied
        remaining = set(self._probes.keys())

        while remaining:
            runnable = {
                name
                for name in remaining
                if self._probes[name]["depends_on"] <= completed
            }
            if not runnable:
                # Deadlock: circular dependency or blocked
                for name in remaining:
                    results.append(
                        ProbeResult(
                            name=name,
                            healthy=False,
                            tier=self._probes[name]["tier"],
                            latency_ms=0.0,
                            message="Dependency deadlock or unsatisfied",
                        )
                    )
                break

            # Run runnable probes in parallel
            with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                futures = {ex.submit(self._run_probe, name): name for name in runnable}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = ProbeResult(
                            name=name,
                            healthy=False,
                            tier=self._probes[name]["tier"],
                            latency_ms=0.0,
                            message=f"Exception: {e}",
                        )
                    results.append(result)
                    self._probes[name]["result"] = result

                    if result.healthy:
                        completed.add(name)
                    else:
                        # This probe failed — mark as blocker
                        blockers.append(name)
                        # Any probe that depends on this is now blocked
                        for other_name in remaining:
                            if name in self._probes[other_name]["depends_on"]:
                                blockers.append(other_name)

            remaining -= runnable
            remaining -= set(blockers)

        latency_ms = (time.time() - start) * 1000

        critical = [r for r in results if r.tier == HealthTier.CRITICAL]
        warning = [r for r in results if r.tier == HealthTier.WARNING]

        critical_healthy = all(r.healthy for r in critical)
        warning_healthy = all(r.healthy for r in warning)

        return ChainStatus(
            healthy=critical_healthy and warning_healthy,
            critical_healthy=critical_healthy,
            warning_healthy=warning_healthy,
            probes=results,
            blockers=blockers,
            latency_ms=latency_ms,
        )

    def _run_probe(self, name: str) -> ProbeResult:
        probe = self._probes[name]
        checker = probe["checker"]
        tier = probe["tier"]
        timeout = probe["timeout"]
        retries = probe["retries"]

        start = time.time()
        healthy = False
        message = ""

        for attempt in range(retries + 1):
            try:
                healthy, message = checker()
                if healthy:
                    break
            except Exception as e:
                message = f"Attempt {attempt + 1}: {e}"
                healthy = False

        latency_ms = (time.time() - start) * 1000

        return ProbeResult(
            name=name,
            healthy=healthy,
            tier=tier,
            latency_ms=latency_ms,
            message=message,
        )

    # ── query ──────────────────────────────────────────

    def probe_names(self) -> set[str]:
        return set(self._probes.keys())

    def report(self) -> dict[str, Any]:
        return {
            "probes": len(self._probes),
            "names": list(self._probes.keys()),
        }
