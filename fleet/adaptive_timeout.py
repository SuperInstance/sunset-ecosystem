"""Adaptive timeout based on historical latency.

Adjusts operation timeouts dynamically based on observed latency
percentiles. Supports exponential moving average, percentile-based
thresholds, and bounded adjustments. Used for fleet RPC timeouts,
health check intervals, and breeding operation deadlines.

Usage:
    timer = AdaptiveTimeout(initial_sec=1.0, min_sec=0.5, max_sec=10.0)
    timer.record_latency(0.8)  # 800ms
    timer.record_latency(1.2)  # 1.2s
    timeout = timer.current_timeout()  # Adjusts based on history
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AdaptiveTimeout:
    """
    Adaptive timeout based on latency history.

    :param initial_sec: Starting timeout value.
    :param min_sec: Minimum allowed timeout.
    :param max_sec: Maximum allowed timeout.
    :param percentile: Percentile to use (0.0-1.0, default 0.95).
    :param alpha: EMA smoothing factor (0.0-1.0, default 0.3).
    :param sample_size: Maximum history size (default 100).
    """

    def __init__(
        self,
        initial_sec: float = 1.0,
        min_sec: float = 0.1,
        max_sec: float = 60.0,
        percentile: float = 0.95,
        alpha: float = 0.3,
        sample_size: int = 100,
    ):
        self._initial = initial_sec
        self._min = min_sec
        self._max = max_sec
        self._percentile = max(0.0, min(1.0, percentile))
        self._alpha = max(0.0, min(1.0, alpha))
        self._sample_size = sample_size
        self._latencies: List[float] = []
        self._ema = initial_sec
        self._count = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_latency(self, latency_sec: float) -> None:
        """
        Record an observed latency.

        :param latency_sec: Observed latency in seconds.
        """
        self._latencies.append(latency_sec)
        if len(self._latencies) > self._sample_size:
            self._latencies.pop(0)
        self._count += 1
        # Update EMA
        self._ema = (self._alpha * latency_sec) + ((1 - self._alpha) * self._ema)

    def record_success(self, latency_sec: float) -> None:
        """Alias for record_latency."""
        self.record_latency(latency_sec)

    def record_timeout(self) -> None:
        """Record a timeout event (adds max latency)."""
        self.record_latency(self._max)

    # ------------------------------------------------------------------
    # Timeout calculation
    # ------------------------------------------------------------------

    def current_timeout(self) -> float:
        """
        Calculate current adaptive timeout.

        :returns: Timeout value clamped to [min, max].
        """
        if not self._latencies:
            return self._initial

        # Calculate percentile-based timeout
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * self._percentile)
        idx = min(idx, len(sorted_latencies) - 1)
        percentile_value = sorted_latencies[idx]

        # Blend EMA and percentile
        blended = (self._ema + percentile_value) / 2.0

        # Add small headroom (10%)
        timeout = blended * 1.1

        return max(self._min, min(self._max, timeout))

    def reset(self) -> None:
        """Reset all history."""
        self._latencies.clear()
        self._ema = self._initial
        self._count = 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def latency_history(self) -> List[float]:
        """Get recorded latency history."""
        return list(self._latencies)

    def average_latency(self) -> Optional[float]:
        """Calculate average latency."""
        if not self._latencies:
            return None
        return sum(self._latencies) / len(self._latencies)

    def percentile_latency(self, p: float) -> Optional[float]:
        """Calculate latency at a given percentile."""
        if not self._latencies:
            return None
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * max(0.0, min(1.0, p)))
        idx = min(idx, len(sorted_latencies) - 1)
        return sorted_latencies[idx]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "initial": self._initial,
            "min": self._min,
            "max": self._max,
            "percentile": self._percentile,
            "alpha": self._alpha,
            "sample_size": self._sample_size,
            "count": self._count,
            "ema": round(self._ema, 4),
            "current_timeout": round(self.current_timeout(), 4),
            "average_latency": round(self.average_latency(), 4)
            if self._latencies
            else None,
        }

    def __repr__(self) -> str:
        return f"<AdaptiveTimeout timeout={self.current_timeout():.3f} samples={self._count}>"
