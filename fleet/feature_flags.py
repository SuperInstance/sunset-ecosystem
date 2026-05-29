from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FeatureFlag:
    """A feature flag with rollout rules."""
    name: str
    enabled: bool
    rollout_percentage: float = 100.0
    targeting: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_enabled_for(self, context: Dict[str, Any]) -> bool:
        """Check if flag is enabled for a specific context."""
        if not self.enabled:
            return False

        # Check targeting rules
        for key, value in self.targeting.items():
            if key not in context or context[key] != value:
                return False

        # Percentage rollout
        if self.rollout_percentage < 100.0:
            # Deterministic hash-based rollout
            import hashlib
            context_str = json.dumps(context, sort_keys=True)
            hash_val = int(hashlib.sha256(f"{self.name}:{context_str}".encode()).hexdigest(), 16)
            return (hash_val % 100) < self.rollout_percentage

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "rollout_percentage": self.rollout_percentage,
            "targeting": self.targeting,
            "metadata": self.metadata,
        }


class FeatureFlagManager:
    """
    Feature flag system for gradual rollouts.

    Supports percentage rollouts, targeting, and A/B testing.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.flags: Dict[str, FeatureFlag] = {}
        self._evaluations: Dict[str, int] = {"enabled": 0, "disabled": 0}

    def create(self, name: str, enabled: bool = False,
               rollout_percentage: float = 100.0,
               targeting: Optional[Dict[str, Any]] = None) -> FeatureFlag:
        """Create a feature flag."""
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            rollout_percentage=rollout_percentage,
            targeting=targeting or {},
        )
        self.flags[name] = flag
        return flag

    def enable(self, name: str) -> bool:
        """Enable a feature flag."""
        if name not in self.flags:
            return False
        self.flags[name].enabled = True
        return True

    def disable(self, name: str) -> bool:
        """Disable a feature flag."""
        if name not in self.flags:
            return False
        self.flags[name].enabled = False
        return True

    def set_rollout(self, name: str, percentage: float) -> bool:
        """Set rollout percentage."""
        if name not in self.flags:
            return False
        self.flags[name].rollout_percentage = max(0.0, min(100.0, percentage))
        return True

    def check(self, name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature is enabled for a context."""
        if name not in self.flags:
            return False
        context = context or {}
        result = self.flags[name].is_enabled_for(context)
        if result:
            self._evaluations["enabled"] += 1
        else:
            self._evaluations["disabled"] += 1
        return result

    def get(self, name: str) -> Optional[FeatureFlag]:
        """Get a feature flag."""
        return self.flags.get(name)

    def list_flags(self) -> List[FeatureFlag]:
        """List all feature flags."""
        return list(self.flags.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get feature flag statistics."""
        return {
            "total_flags": len(self.flags),
            "enabled_flags": sum(1 for f in self.flags.values() if f.enabled),
            "evaluations": self._evaluations.copy(),
        }

    def export_json(self) -> str:
        """Export all flags as JSON."""
        return json.dumps({
            "node": self.fleet_node_id,
            "flags": [f.to_dict() for f in self.flags.values()],
            "stats": self.get_stats(),
        }, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
