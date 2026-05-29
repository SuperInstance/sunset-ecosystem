"""Health status aggregation from multiple sources.

Collects health signals from multiple services or nodes, computes
aggregate health states, and supports customizable aggregation strategies
(healthiest, unhealthiest, threshold-based, quorum). Used for fleet-wide
health dashboards, load balancer health decisions, and alerting.

Usage:
    ha = HealthAggregator(strategy="quorum", threshold=0.5)
    ha.report("svc-a", "healthy")
    ha.report("svc-b", "degraded")
    ha.report("svc-c", "healthy")
    status = ha.status()  # "healthy" (2/3 meet threshold)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class HealthAggregator:
    """
    Aggregate health from multiple sources.

    :param strategy: Aggregation strategy name.
    :param threshold: Threshold for threshold/quorum strategies.
    :param health_levels: Ordered health levels (worst to best).
    """

    HEALTH_LEVELS = ["critical", "unhealthy", "degraded", "healthy", "excellent"]

    def __init__(
        self,
        strategy: str = "unhealthiest",
        threshold: float = 0.5,
        health_levels: Optional[List[str]] = None,
        custom_strategy: Optional[Callable[[List[str]], str]] = None,
    ):
        self._strategy = strategy
        self._threshold = threshold
        self._health_levels = health_levels or list(self.HEALTH_LEVELS)
        self._custom_strategy = custom_strategy
        self._reports: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, source: str, health: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Report health status from a source.

        :param source: Source identifier.
        :param health: Health level string.
        :param metadata: Additional context.
        """
        self._reports[source] = health
        if metadata:
            self._metadata[source] = metadata

    def remove(self, source: str) -> bool:
        """Remove a source's report."""
        if source in self._reports:
            del self._reports[source]
            self._metadata.pop(source, None)
            return True
        return False

    def clear(self) -> None:
        """Clear all reports."""
        self._reports.clear()
        self._metadata.clear()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def status(self) -> str:
        """
        Compute aggregate health status.

        :returns: Aggregate health string.
        """
        if not self._reports:
            return "unknown"

        if self._custom_strategy:
            return self._custom_strategy(list(self._reports.values()))

        levels = self._health_levels
        values = [self._level_index(h) for h in self._reports.values()]

        if self._strategy == "healthiest":
            return levels[max(values)]
        if self._strategy == "unhealthiest":
            return levels[min(values)]
        if self._strategy == "average":
            avg = sum(values) / len(values)
            return levels[int(avg)]
        if self._strategy == "threshold":
            healthy_count = sum(1 for v in values if v >= self._threshold * len(levels))
            ratio = healthy_count / len(values)
            return "healthy" if ratio >= self._threshold else "unhealthy"
        if self._strategy == "quorum":
            healthy_count = sum(1 for v in values if v >= levels.index("healthy"))
            return "healthy" if healthy_count / len(values) >= self._threshold else "unhealthy"

        return "unknown"

    def _level_index(self, health: str) -> int:
        try:
            return self._health_levels.index(health)
        except ValueError:
            return 0  # Unknown = worst

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def sources(self) -> List[str]:
        return list(self._reports.keys())

    def get(self, source: str) -> Optional[str]:
        return self._reports.get(source)

    def counts(self) -> Dict[str, int]:
        """Count of each health level."""
        counts: Dict[str, int] = {}
        for h in self._reports.values():
            counts[h] = counts.get(h, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "sources": len(self._reports),
            "strategy": self._strategy,
            "threshold": self._threshold,
            "aggregate_status": self.status(),
            "counts": self.counts(),
        }

    def __repr__(self) -> str:
        return f"<HealthAggregator sources={len(self._reports)} status={self.status()}>"
