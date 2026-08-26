"""Feature flag / toggle system with rollout percentages.

Manages feature flags with boolean, percentage, and user-targeted
rollouts. Supports flag groups, default values, and override checks.
Used for fleet feature gating, A/B testing, and gradual rollouts.

Usage:
    flags = FeatureFlags()
    flags.set("dark_mode", True)
    flags.set("beta_feature", 10)  # 10% rollout
    assert flags.is_enabled("dark_mode") is True
    assert flags.is_enabled("beta_feature", user_id="user-1") in [True, False]
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Union


class FeatureFlags:
    """
    Feature flag system with percentage rollouts.
    """

    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        self._flags: Dict[str, Any] = defaults or {}
        self._overrides: Dict[str, List[str]] = {}  # user_id -> [flag names]

    # ------------------------------------------------------------------
    # Flag management
    # ------------------------------------------------------------------

    def set(self, name: str, value: Union[bool, int, float]) -> None:
        """
        Set a feature flag.

        :param value: True/False for boolean, 0-100 for percentage rollout.
        """
        self._flags[name] = value

    def get(self, name: str) -> Any:
        """Get flag raw value."""
        return self._flags.get(name)

    def remove(self, name: str) -> bool:
        """Remove a flag."""
        if name in self._flags:
            del self._flags[name]
            return True
        return False

    def list_flags(self) -> List[str]:
        """List all flag names."""
        return list(self._flags.keys())

    # ------------------------------------------------------------------
    # Override management
    # ------------------------------------------------------------------

    def add_override(self, user_id: str, flag_name: str) -> None:
        """Add a user override for a flag."""
        if user_id not in self._overrides:
            self._overrides[user_id] = []
        if flag_name not in self._overrides[user_id]:
            self._overrides[user_id].append(flag_name)

    def remove_override(self, user_id: str, flag_name: str) -> bool:
        """Remove a user override."""
        if user_id in self._overrides and flag_name in self._overrides[user_id]:
            self._overrides[user_id].remove(flag_name)
            return True
        return False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def is_enabled(self, name: str, user_id: Optional[str] = None) -> bool:
        """
        Check if a feature is enabled.

        :param name: Flag name.
        :param user_id: Optional user ID for percentage/override checks.
        :returns: True if enabled.
        """
        value = self._flags.get(name)
        if value is None:
            return False

        # Boolean
        if isinstance(value, bool):
            return value

        # Percentage rollout
        if isinstance(value, (int, float)):
            # Check override first
            if user_id and user_id in self._overrides:
                if name in self._overrides[user_id]:
                    return True

            # Hash user_id to determine if in rollout bucket
            if user_id:
                hash_val = int(
                    hashlib.md5(f"{name}:{user_id}".encode()).hexdigest(), 16
                )
                bucket = hash_val % 100
                return bucket < value
            return False

        return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "flags": len(self._flags),
            "overrides": sum(len(v) for v in self._overrides.values()),
        }

    def __repr__(self) -> str:
        return f"<FeatureFlags flags={len(self._flags)}>"
