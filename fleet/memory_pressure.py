"""memory_pressure.py — Memory monitoring and pressure handling.

Provides:
1. RSS / VMS memory tracking
2. Configurable thresholds (warn, critical)
3. Automatic GC trigger
4. Cache eviction on pressure
5. Alert callbacks

Usage:
    mp = MemoryPressure(warn_threshold_mb=512, critical_threshold_mb=1024)
    mp.register_alert("critical", lambda level: print(f"CRITICAL: {level}"))
    mp.check()  # Runs GC, checks thresholds, fires alerts
"""

from __future__ import annotations

__all__ = [
    "MemoryPressure",
    "MemorySnapshot",
]

import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """A memory usage snapshot."""

    rss_mb: float
    vms_mb: float
    percent: float  # Of total system memory
    timestamp: float


class MemoryPressure:
    """Monitor memory and react to pressure thresholds."""

    def __init__(
        self,
        warn_threshold_mb: float = 512.0,
        critical_threshold_mb: float = 1024.0,
        gc_threshold_mb: float = 768.0,
        check_interval: float = 30.0,
    ) -> None:
        self._warn_mb = warn_threshold_mb
        self._critical_mb = critical_threshold_mb
        self._gc_mb = gc_threshold_mb
        self._check_interval = check_interval
        self._last_check = 0.0
        self._alerts: dict[str, list[Callable[[str, MemorySnapshot], None]]] = {
            "warn": [],
            "critical": [],
        }
        self._evictors: list[Callable[[], int]] = []  # Return bytes freed
        self._history: list[MemorySnapshot] = []
        self._max_history = 100
        self._peak_mb = 0.0

    def register_alert(
        self, level: str, fn: Callable[[str, MemorySnapshot], None]
    ) -> None:
        """Register an alert callback (warn or critical)."""
        if level in self._alerts:
            self._alerts[level].append(fn)

    def register_evictor(self, fn: Callable[[], int]) -> None:
        """Register a cache evictor that returns bytes freed."""
        self._evictors.append(fn)

    def snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
            total = psutil.virtual_memory().total / (1024 * 1024)
            percent = (rss_mb / total) * 100 if total > 0 else 0.0
        except ImportError:
            # Fallback without psutil
            rss_mb = 0.0
            vms_mb = 0.0
            percent = 0.0
        snap = MemorySnapshot(
            rss_mb=rss_mb, vms_mb=vms_mb, percent=percent, timestamp=time.time()
        )
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._peak_mb = max(self._peak_mb, rss_mb)
        return snap

    def check(self) -> MemorySnapshot:
        """Check memory pressure and react."""
        now = time.time()
        if now - self._last_check < self._check_interval:
            # Return latest without rechecking
            return self._history[-1] if self._history else self.snapshot()
        self._last_check = now

        snap = self.snapshot()

        # Trigger GC if over threshold
        if snap.rss_mb > self._gc_mb:
            logger.info(
                f"Memory {snap.rss_mb:.1f}MB > GC threshold {self._gc_mb:.1f}MB, running GC"
            )
            gc.collect()
            # Re-measure after GC
            snap = self.snapshot()

        # Evict caches if still over
        if snap.rss_mb > self._gc_mb:
            for evictor in self._evictors:
                freed = evictor()
                logger.info(f"Evictor freed {freed} bytes")

        # Check thresholds and fire alerts
        if snap.rss_mb > self._critical_mb:
            self._fire_alert("critical", snap)
        elif snap.rss_mb > self._warn_mb:
            self._fire_alert("warn", snap)

        return snap

    def _fire_alert(self, level: str, snap: MemorySnapshot) -> None:
        logger.warning(f"Memory {level}: {snap.rss_mb:.1f}MB")
        for fn in self._alerts.get(level, []):
            try:
                fn(level, snap)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def peak(self) -> float:
        """Peak RSS memory observed (MB)."""
        return self._peak_mb

    def average(self, n: int = 10) -> float:
        """Average RSS over last N snapshots."""
        if not self._history:
            return 0.0
        samples = self._history[-n:]
        return sum(s.rss_mb for s in samples) / len(samples)

    def trend(self, n: int = 10) -> float:
        """Memory trend: positive = growing, negative = shrinking (MB/min)."""
        if len(self._history) < 2:
            return 0.0
        samples = self._history[-n:]
        if len(samples) < 2:
            return 0.0
        dt = samples[-1].timestamp - samples[0].timestamp
        if dt <= 0:
            return 0.0
        drss = samples[-1].rss_mb - samples[0].rss_mb
        return (drss / dt) * 60.0  # MB per minute

    def stats(self) -> dict[str, Any]:
        snap = self.snapshot()
        return {
            "rss_mb": snap.rss_mb,
            "vms_mb": snap.vms_mb,
            "percent": snap.percent,
            "peak_mb": self._peak_mb,
            "average_mb": self.average(),
            "trend_mb_per_min": self.trend(),
            "history_samples": len(self._history),
        }

    def __repr__(self) -> str:
        return f"MemoryPressure(warn={self._warn_mb}MB, critical={self._critical_mb}MB)"
