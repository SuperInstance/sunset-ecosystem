"""auto_scaler.py — Auto-scale fleet nodes based on load metrics.

Provides:
1. Scale-up decisions based on CPU/memory thresholds
2. Scale-down with cooldown to prevent flapping
3. Target utilization-based scaling
4. Predictive scaling (simple trend extrapolation)
5. Scaling recommendations with confidence

Usage:
    scaler = AutoScaler(target_cpu=0.7, target_memory=0.8)
    decision = scaler.evaluate(metrics={"cpu": 0.85, "memory": 0.9})
    if decision.action == "scale_up":
        provision_nodes(decision.count)
"""

from __future__ import annotations

__all__ = [
    "AutoScaler",
    "ScaleDecision",
    "ScalingPolicy",
]

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScaleDecision:
    """A scaling decision."""

    action: str  # "scale_up", "scale_down", "hold"
    count: int = 0
    reason: str = ""
    confidence: float = 0.0  # 0.0-1.0
    current_nodes: int = 0
    target_nodes: int = 0


@dataclass
class ScalingPolicy:
    """Scaling policy parameters."""

    target_cpu: float = 0.7
    target_memory: float = 0.8
    scale_up_threshold: float = 0.85
    scale_down_threshold: float = 0.3
    max_nodes: int = 100
    min_nodes: int = 1
    scale_up_cooldown: float = 300.0
    scale_down_cooldown: float = 600.0
    scale_up_step: int = 2
    scale_down_step: int = 1


class AutoScaler:
    """Auto-scale fleet based on utilization metrics."""

    def __init__(self, policy: ScalingPolicy | None = None) -> None:
        self._policy = policy or ScalingPolicy()
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0
        self._history: list[tuple[float, dict[str, float]]] = []
        self._max_history = 60

    def evaluate(
        self,
        metrics: dict[str, float],
        current_nodes: int,
    ) -> ScaleDecision:
        """Evaluate metrics and return a scaling decision."""
        now = time.time()
        self._history.append((now, dict(metrics)))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        cpu = metrics.get("cpu", 0.0)
        memory = metrics.get("memory", 0.0)
        p = self._policy

        # Check scale-up
        if cpu > p.scale_up_threshold or memory > p.scale_up_threshold:
            if now - self._last_scale_up >= p.scale_up_cooldown:
                if current_nodes < p.max_nodes:
                    target = min(current_nodes + p.scale_up_step, p.max_nodes)
                    self._last_scale_up = now
                    return ScaleDecision(
                        action="scale_up",
                        count=target - current_nodes,
                        reason=f"cpu={cpu:.2f}, memory={memory:.2f} above threshold",
                        confidence=min(1.0, max(cpu, memory)),
                        current_nodes=current_nodes,
                        target_nodes=target,
                    )

        # Check scale-down
        if cpu < p.scale_down_threshold and memory < p.scale_down_threshold:
            if now - self._last_scale_down >= p.scale_down_cooldown:
                if current_nodes > p.min_nodes:
                    target = max(current_nodes - p.scale_down_step, p.min_nodes)
                    self._last_scale_down = now
                    return ScaleDecision(
                        action="scale_down",
                        count=current_nodes - target,
                        reason=f"cpu={cpu:.2f}, memory={memory:.2f} below threshold",
                        confidence=min(1.0, 1.0 - max(cpu, memory)),
                        current_nodes=current_nodes,
                        target_nodes=target,
                    )

        # Predictive: if trend shows rapid increase, suggest early scale
        trend_confidence = self._predict_trend()
        if trend_confidence > 0.7 and current_nodes < p.max_nodes:
            if now - self._last_scale_up >= p.scale_up_cooldown:
                target = min(current_nodes + p.scale_up_step, p.max_nodes)
                self._last_scale_up = now
                return ScaleDecision(
                    action="scale_up",
                    count=target - current_nodes,
                    reason="predictive: trend shows rapid increase",
                    confidence=trend_confidence,
                    current_nodes=current_nodes,
                    target_nodes=target,
                )

        return ScaleDecision(
            action="hold",
            reason=f"cpu={cpu:.2f}, memory={memory:.2f} within bounds",
            current_nodes=current_nodes,
            target_nodes=current_nodes,
        )

    def _predict_trend(self) -> float:
        """Simple linear trend prediction from history."""
        if len(self._history) < 5:
            return 0.0
        recent = self._history[-10:]
        cpus = [m.get("cpu", 0.0) for _, m in recent]
        # Simple slope: last - first
        if len(cpus) >= 2:
            slope = cpus[-1] - cpus[0]
            if slope > 0.2:
                return min(1.0, slope)
        return 0.0

    def cooldown_remaining(self) -> dict[str, float]:
        """Get remaining cooldown time for scale up/down."""
        now = time.time()
        return {
            "scale_up": max(
                0.0, self._policy.scale_up_cooldown - (now - self._last_scale_up)
            ),
            "scale_down": max(
                0.0, self._policy.scale_down_cooldown - (now - self._last_scale_down)
            ),
        }

    def stats(self) -> dict[str, Any]:
        """Scaler statistics."""
        return {
            "history_points": len(self._history),
            "last_scale_up": self._last_scale_up,
            "last_scale_down": self._last_scale_down,
            "cooldown": self.cooldown_remaining(),
        }

    def __repr__(self) -> str:
        return f"AutoScaler(nodes={self._policy.min_nodes}-{self._policy.max_nodes})"
