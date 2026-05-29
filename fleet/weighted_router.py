"""Weighted routing with health-adjusted weights.

Routes requests to backends using weighted selection, with automatic
weight adjustment based on health status. Supports sticky sessions,
health-based weight reduction, and weight normalization. Used for
fleet load balancing, A/B test routing, and canary deployments.

Usage:
    router = WeightedRouter()
    router.add_backend("svc-a", weight=5, health="healthy")
    router.add_backend("svc-b", weight=3, health="degraded")
    target = router.select()  # Probabilistic weighted selection
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class WeightedRouter:
    """
    Weighted router with health-adjusted weights.

    :param health_weight_map: Dict mapping health status to weight multiplier.
    """

    def __init__(
        self,
        health_weight_map: Optional[Dict[str, float]] = None,
    ):
        self._health_map = health_weight_map or {
            "excellent": 1.0,
            "healthy": 1.0,
            "degraded": 0.5,
            "unhealthy": 0.1,
            "critical": 0.0,
        }
        self._backends: Dict[str, Dict[str, Any]] = {}
        self._sticky: Dict[str, str] = {}  # session_id -> backend

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def add_backend(
        self,
        name: str,
        weight: float,
        health: str = "healthy",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a backend with weight and health.

        :param name: Backend identifier.
        :param weight: Base routing weight.
        :param health: Health status string.
        :param metadata: Optional backend metadata.
        :returns: True if added, False if already exists.
        """
        if name in self._backends:
            return False
        self._backends[name] = {
            "weight": weight,
            "health": health,
            "metadata": metadata or {},
        }
        return True

    def remove_backend(self, name: str) -> bool:
        """Remove a backend."""
        if name in self._backends:
            del self._backends[name]
            # Remove sticky sessions pointing to this backend
            sessions_to_remove = [sid for sid, b in self._sticky.items() if b == name]
            for sid in sessions_to_remove:
                del self._sticky[sid]
            return True
        return False

    def update_health(self, name: str, health: str) -> bool:
        """Update backend health status."""
        backend = self._backends.get(name)
        if not backend:
            return False
        backend["health"] = health
        return True

    def update_weight(self, name: str, weight: float) -> bool:
        """Update backend weight."""
        backend = self._backends.get(name)
        if not backend:
            return False
        backend["weight"] = weight
        return True

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, session_id: Optional[str] = None) -> Optional[str]:
        """
        Select a backend using weighted random selection.

        :param session_id: Optional session ID for sticky routing.
        :returns: Selected backend name or None.
        """
        if session_id and session_id in self._sticky:
            sticky_backend = self._sticky[session_id]
            if sticky_backend in self._backends:
                return sticky_backend

        # Calculate effective weights
        effective_weights = {}
        total_weight = 0.0
        for name, backend in self._backends.items():
            multiplier = self._health_map.get(backend["health"], 0.0)
            effective = backend["weight"] * multiplier
            if effective > 0:
                effective_weights[name] = effective
                total_weight += effective

        if total_weight <= 0:
            return None

        # Weighted random selection
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for name, weight in effective_weights.items():
            cumulative += weight
            if r <= cumulative:
                if session_id:
                    self._sticky[session_id] = name
                return name

        # Fallback to last backend
        last = list(effective_weights.keys())[-1]
        if session_id:
            self._sticky[session_id] = last
        return last

    def sticky_clear(self, session_id: str) -> bool:
        """Clear sticky session mapping."""
        if session_id in self._sticky:
            del self._sticky[session_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def backends(self) -> List[str]:
        return list(self._backends.keys())

    def get_backend(self, name: str) -> Optional[Dict[str, Any]]:
        return self._backends.get(name)

    def effective_weights(self) -> Dict[str, float]:
        """Get effective weights after health adjustment."""
        weights = {}
        for name, backend in self._backends.items():
            multiplier = self._health_map.get(backend["health"], 0.0)
            weights[name] = backend["weight"] * multiplier
        return weights

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total_weight = sum(self.effective_weights().values())
        return {
            "backends": len(self._backends),
            "sticky_sessions": len(self._sticky),
            "total_effective_weight": round(total_weight, 4),
            "health_distribution": self._health_distribution(),
        }

    def _health_distribution(self) -> Dict[str, int]:
        distribution = {}
        for backend in self._backends.values():
            h = backend["health"]
            distribution[h] = distribution.get(h, 0) + 1
        return distribution

    def __repr__(self) -> str:
        return f"<WeightedRouter backends={len(self._backends)} sticky={len(self._sticky)}>"
