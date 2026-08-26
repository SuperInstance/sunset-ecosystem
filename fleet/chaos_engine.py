"""Chaos engineering fault injection engine.

Injects controlled faults (latency, errors, crashes) into fleet
operations for resilience testing. Supports probability-based fault
triggers and per-service targeting.

Usage:
    chaos = ChaosEngine()
    chaos.add_fault("latency", target="users-service", probability=0.1, delay_sec=2)
    if chaos.should_trigger("users-service"):
        chaos.apply_fault("users-service")
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional


class ChaosEngine:
    """
    Fault injection engine for resilience testing.
    """

    def __init__(self, enabled: bool = True, seed: Optional[int] = None):
        self.enabled = enabled
        self._faults: Dict[str, List[Dict[str, Any]]] = {}
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Fault definition
    # ------------------------------------------------------------------

    def add_fault(
        self,
        name: str,
        target: str,
        probability: float,
        fault_type: str = "latency",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a fault rule.

        :param name: Fault rule name.
        :param target: Target service identifier.
        :param probability: Trigger probability (0.0-1.0).
        :param fault_type: Type of fault (latency, error, crash).
        :param params: Fault-specific parameters.
        """
        if target not in self._faults:
            self._faults[target] = []
        self._faults[target].append(
            {
                "name": name,
                "probability": probability,
                "type": fault_type,
                "params": params or {},
            }
        )

    def remove_fault(self, target: str, name: str) -> bool:
        """Remove a fault rule."""
        if target not in self._faults:
            return False
        before = len(self._faults[target])
        self._faults[target] = [f for f in self._faults[target] if f["name"] != name]
        return len(self._faults[target]) < before

    # ------------------------------------------------------------------
    # Triggering
    # ------------------------------------------------------------------

    def should_trigger(self, target: str) -> bool:
        """Check if any fault should trigger for a target."""
        if not self.enabled:
            return False
        faults = self._faults.get(target, [])
        for fault in faults:
            if random.random() < fault["probability"]:
                return True
        return False

    def get_triggered_fault(self, target: str) -> Optional[Dict[str, Any]]:
        """Get a randomly selected triggered fault."""
        if not self.enabled:
            return None
        faults = self._faults.get(target, [])
        triggered = [f for f in faults if random.random() < f["probability"]]
        if triggered:
            return random.choice(triggered)
        return None

    def apply_fault(self, target: str) -> Optional[Dict[str, Any]]:
        """
        Apply a fault to a target.

        :returns: Applied fault dict or None.
        """
        fault = self.get_triggered_fault(target)
        if fault and fault["type"] == "latency":
            delay = fault["params"].get("delay_sec", 1.0)
            time.sleep(delay)
        return fault

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def targets(self) -> List[str]:
        return list(self._faults.keys())

    def faults_for_target(self, target: str) -> List[Dict[str, Any]]:
        return list(self._faults.get(target, []))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = sum(len(v) for v in self._faults.values())
        return {
            "enabled": self.enabled,
            "targets": len(self._faults),
            "faults": total,
        }

    def __repr__(self) -> str:
        return f"<ChaosEngine enabled={self.enabled} targets={len(self._faults)}>"
