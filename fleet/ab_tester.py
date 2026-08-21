from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ABTestVariant:
    """A variant in an A/B test."""

    name: str
    config: Dict[str, Any]
    traffic_percentage: float
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "traffic_percentage": self.traffic_percentage,
            "metrics": self.metrics,
        }


@dataclass
class ABTest:
    """An A/B test."""

    test_id: str
    name: str
    variants: List[ABTestVariant]
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
        }


class ABTester:
    """
    A/B testing framework for breeding configurations.

    Runs multiple variants and compares metrics.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._tests: Dict[str, ABTest] = {}

    def create(self, name: str, variants: List[Dict[str, Any]]) -> ABTest:
        """Create a new A/B test."""
        test_id = f"ab_{int(time.time() * 1000000)}"
        variant_objs = []
        total_pct = 0.0
        for i, v in enumerate(variants):
            pct = v.get("traffic_percentage", 100.0 / len(variants))
            total_pct += pct
            variant_objs.append(
                ABTestVariant(
                    name=v["name"],
                    config=v.get("config", {}),
                    traffic_percentage=pct,
                )
            )
        test = ABTest(
            test_id=test_id,
            name=name,
            variants=variant_objs,
        )
        self._tests[test_id] = test
        return test

    def record_metric(
        self, test_id: str, variant_name: str, metric_name: str, value: float
    ) -> bool:
        """Record a metric for a variant."""
        test = self._tests.get(test_id)
        if not test:
            return False
        for variant in test.variants:
            if variant.name == variant_name:
                if metric_name not in variant.metrics:
                    variant.metrics[metric_name] = []
                variant.metrics[metric_name].append(value)
                return True
        return False

    def get_winner(self, test_id: str, metric_name: str) -> Optional[str]:
        """Get the winning variant for a metric."""
        test = self._tests.get(test_id)
        if not test:
            return None
        best_variant = None
        best_mean = -float("inf")
        for variant in test.variants:
            values = variant.metrics.get(metric_name, [])
            if values:
                mean = np.mean(values)
                if mean > best_mean:
                    best_mean = mean
                    best_variant = variant.name
        return best_variant

    def end(self, test_id: str) -> bool:
        """End an A/B test."""
        test = self._tests.get(test_id)
        if not test:
            return False
        test.status = "ended"
        test.ended_at = time.time()
        return True

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get an A/B test."""
        return self._tests.get(test_id)

    def get_all_tests(self) -> List[ABTest]:
        """Get all tests."""
        return list(self._tests.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get A/B test statistics."""
        return {
            "total_tests": len(self._tests),
            "running": sum(1 for t in self._tests.values() if t.status == "running"),
            "ended": sum(1 for t in self._tests.values() if t.status == "ended"),
        }

    def export_json(self) -> str:
        """Export tests as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "tests": [t.to_dict() for t in self._tests.values()],
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
