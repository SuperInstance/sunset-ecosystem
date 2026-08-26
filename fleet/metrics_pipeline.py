"""Metrics collection and transformation pipeline.

Collects raw metrics, applies transformations (filtering, aggregation,
derivatives), and outputs processed metrics. Used for fleet telemetry
processing and dashboard data preparation.

Usage:
    pipe = MetricsPipeline()
    pipe.add_transform(lambda m: {**m, "value": m["value"] * 2})
    result = pipe.process({"name": "cpu", "value": 50})
    # result["value"] == 100
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class MetricsPipeline:
    """
    Chainable metrics transformation pipeline.

    Transforms flow: input -> [filter] -> [transform] -> [aggregate] -> output
    """

    def __init__(self):
        self._filters: List[Callable[[Dict[str, Any]], bool]] = []
        self._transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []
        self._aggregators: List[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = []
        self._processed = 0
        self._dropped = 0

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def add_filter(self, fn: Callable[[Dict[str, Any]], bool]) -> "MetricsPipeline":
        """Add a filter stage (returns True to keep)."""
        self._filters.append(fn)
        return self

    def add_transform(
        self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> "MetricsPipeline":
        """Add a transform stage."""
        self._transforms.append(fn)
        return self

    def add_aggregator(
        self, fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]]
    ) -> "MetricsPipeline":
        """Add an aggregator stage."""
        self._aggregators.append(fn)
        return self

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, metric: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single metric through the pipeline.

        :returns: Transformed metric, or None if filtered out.
        """
        # Filters
        for f in self._filters:
            if not f(metric):
                self._dropped += 1
                return None
        # Transforms
        for t in self._transforms:
            metric = t(metric)
        self._processed += 1
        return metric

    def process_batch(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of metrics."""
        results: List[Dict[str, Any]] = []
        for m in metrics:
            out = self.process(m)
            if out:
                results.append(out)
        # Aggregators
        for agg in self._aggregators:
            if results:
                agg_result = agg(results)
                if agg_result:
                    results.append(agg_result)
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "processed": self._processed,
            "dropped": self._dropped,
            "filters": len(self._filters),
            "transforms": len(self._transforms),
            "aggregators": len(self._aggregators),
        }

    def reset_stats(self) -> None:
        self._processed = 0
        self._dropped = 0

    def __repr__(self) -> str:
        return (
            f"<MetricsPipeline filters={len(self._filters)} "
            f"transforms={len(self._transforms)} "
            f"aggregators={len(self._aggregators)}>"
        )
