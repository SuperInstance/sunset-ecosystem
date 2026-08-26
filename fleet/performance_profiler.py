"""Simple performance profiler and timing utilities.

Provides timing, call counting, and profiling for fleet operations.
Used for identifying bottlenecks, measuring latency, and tracking
performance regressions.

Usage:
    prof = PerformanceProfiler()
    with prof.time("db_query"):
        run_query()
    stats = prof.stats()
    assert stats["db_query"]["count"] == 1
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


class PerformanceProfiler:
    """
    Simple profiler with timing and counters.
    """

    def __init__(self):
        self._timings: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        """Context manager to time a block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._record(name, elapsed)

    def _record(self, name: str, elapsed: float) -> None:
        """Record a timing."""
        if name not in self._timings:
            self._timings[name] = {
                "count": 0,
                "total": 0.0,
                "min": elapsed,
                "max": elapsed,
            }
        t = self._timings[name]
        t["count"] += 1
        t["total"] += elapsed
        t["min"] = min(t["min"], elapsed)
        t["max"] = max(t["max"], elapsed)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get stats for all or a specific measurement."""
        if name:
            return self._timings.get(name, {})
        return dict(self._timings)

    def avg(self, name: str) -> float:
        """Get average time for a measurement."""
        t = self._timings.get(name)
        if not t or t["count"] == 0:
            return 0.0
        return t["total"] / t["count"]

    def count(self, name: str) -> int:
        """Get call count for a measurement."""
        return self._timings.get(name, {}).get("count", 0)

    def reset(self, name: Optional[str] = None) -> None:
        """Reset stats for a measurement or all."""
        if name:
            self._timings.pop(name, None)
        else:
            self._timings.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Get summary of all measurements."""
        return {
            "measurements": len(self._timings),
            "total_calls": sum(t["count"] for t in self._timings.values()),
        }

    def __repr__(self) -> str:
        return f"<PerformanceProfiler measurements={len(self._timings)}>"
