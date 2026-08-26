"""Fleet capacity planner with forecasting and scaling recommendations.

Analyzes resource usage trends and recommends fleet scaling actions.
Used for proactive capacity management and cost optimization.

Usage:
    planner = CapacityPlanner()
    planner.add_usage("cpu", [10, 20, 30, 40, 50])
    rec = planner.recommend()
    # rec["action"] == "scale_up"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class CapacityPlanner:
    """
    Capacity planner with trend analysis.

    :param scale_up_threshold: Utilization threshold to trigger scale-up.
    :param scale_down_threshold: Utilization threshold to trigger scale-down.
    """

    def __init__(
        self,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
    ):
        self._scale_up = scale_up_threshold
        self._scale_down = scale_down_threshold
        self._usage: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_usage(self, resource: str, values: List[float]) -> None:
        """Add usage history for a resource."""
        self._usage[resource] = list(values)

    def add_point(self, resource: str, value: float) -> None:
        """Add a single usage point."""
        if resource not in self._usage:
            self._usage[resource] = []
        self._usage[resource].append(value)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def current_utilization(self) -> Dict[str, float]:
        """Get latest utilization per resource."""
        return {r: values[-1] if values else 0.0 for r, values in self._usage.items()}

    def avg_utilization(self) -> Dict[str, float]:
        """Get average utilization per resource."""
        return {
            r: sum(values) / len(values) if values else 0.0
            for r, values in self._usage.items()
        }

    def trend(self, resource: str) -> float:
        """
        Calculate linear trend slope.

        Returns slope (change per sample). Positive = increasing.
        """
        values = self._usage.get(resource, [])
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator else 0.0

    def forecast(self, resource: str, steps: int = 5) -> List[float]:
        """Forecast future usage based on trend."""
        values = self._usage.get(resource, [])
        if not values:
            return [0.0] * steps
        slope = self.trend(resource)
        last = values[-1]
        return [last + slope * (i + 1) for i in range(steps)]

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend(self) -> Dict[str, Any]:
        """
        Generate scaling recommendation.

        Returns: {"action": str, "reasons": list, "resources": dict}
        """
        current = self.current_utilization()
        reasons: List[str] = []
        actions: set = set()
        resource_actions: Dict[str, str] = {}

        for r, util in current.items():
            if util >= self._scale_up:
                actions.add("scale_up")
                reasons.append(f"{r} at {util:.1%} >= {self._scale_up:.1%}")
                resource_actions[r] = "scale_up"
            elif util <= self._scale_down:
                actions.add("scale_down")
                reasons.append(f"{r} at {util:.1%} <= {self._scale_down:.1%}")
                resource_actions[r] = "scale_down"

        if not actions:
            action = "stable"
        elif "scale_up" in actions:
            action = "scale_up"
        else:
            action = "scale_down"

        return {
            "action": action,
            "reasons": reasons,
            "resources": resource_actions,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "resources": len(self._usage),
            "samples": {r: len(v) for r, v in self._usage.items()},
        }

    def __repr__(self) -> str:
        return f"<CapacityPlanner resources={len(self._usage)}>"
