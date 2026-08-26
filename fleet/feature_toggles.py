"""Feature toggles with rollout percentages and user targeting.

Manages feature flags with enable/disable, percentage-based rollout,
and user-targeted activation. Used for fleet canary releases, A/B
testing, and gradual feature rollouts.

Usage:
    toggles = FeatureToggles()
    toggles.register("new-ui", default=False, rollout=10)
    assert toggles.is_enabled("new-ui", user_id="user-123")  # 10% chance
    toggles.set_enabled("new-ui", True)
    assert toggles.is_enabled("new-ui")  # Now enabled for all
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


class FeatureToggles:
    """
    Feature toggles with rollout and targeting.
    """

    def __init__(self):
        self._features: Dict[str, Dict[str, Any]] = {}
        self._user_overrides: Dict[
            str, Dict[str, bool]
        ] = {}  # feature -> {user_id: enabled}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        default: bool = False,
        rollout: int = 0,
        description: str = "",
    ) -> bool:
        """
        Register a feature toggle.

        :param name: Feature name.
        :param default: Default state (enabled/disabled).
        :param rollout: Rollout percentage (0-100).
        :param description: Feature description.
        :returns: True if registered, False if already exists.
        """
        if name in self._features:
            return False
        self._features[name] = {
            "enabled": default,
            "rollout": max(0, min(100, rollout)),
            "description": description,
        }
        self._user_overrides[name] = {}
        return True

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def is_enabled(self, name: str, user_id: Optional[str] = None) -> bool:
        """
        Check if a feature is enabled.

        :param name: Feature name.
        :param user_id: Optional user ID for targeted rollout.
        :returns: True if feature is enabled for this user.
        """
        feature = self._features.get(name)
        if not feature:
            return False

        # Check user override
        if user_id and user_id in self._user_overrides.get(name, {}):
            return self._user_overrides[name][user_id]

        # If fully enabled
        if feature["enabled"]:
            return True

        # Rollout percentage check
        if user_id and feature["rollout"] > 0:
            hash_value = int(hashlib.md5(f"{name}:{user_id}".encode()).hexdigest(), 16)
            user_percentile = hash_value % 100
            return user_percentile < feature["rollout"]

        return False

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """
        Set feature enabled state.

        :param name: Feature name.
        :param enabled: New state.
        :returns: True if updated.
        """
        feature = self._features.get(name)
        if not feature:
            return False
        feature["enabled"] = enabled
        return True

    def set_rollout(self, name: str, rollout: int) -> bool:
        """
        Set rollout percentage.

        :param name: Feature name.
        :param rollout: Rollout percentage (0-100).
        :returns: True if updated.
        """
        feature = self._features.get(name)
        if not feature:
            return False
        feature["rollout"] = max(0, min(100, rollout))
        return True

    def set_user_override(self, name: str, user_id: str, enabled: bool) -> bool:
        """
        Set user-specific override.

        :param name: Feature name.
        :param user_id: User ID.
        :param enabled: Override state.
        :returns: True if updated.
        """
        if name not in self._features:
            return False
        self._user_overrides[name][user_id] = enabled
        return True

    def remove_user_override(self, name: str, user_id: str) -> bool:
        """Remove user override."""
        if name in self._user_overrides and user_id in self._user_overrides[name]:
            del self._user_overrides[name][user_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._features.keys())

    def get_feature(self, name: str) -> Optional[Dict[str, Any]]:
        """Get feature configuration."""
        return self._features.get(name)

    def get_user_override(self, name: str, user_id: str) -> Optional[bool]:
        """Get user override state."""
        return self._user_overrides.get(name, {}).get(user_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        enabled = sum(1 for f in self._features.values() if f["enabled"])
        rollout = sum(
            1 for f in self._features.values() if f["rollout"] > 0 and not f["enabled"]
        )
        return {
            "features": len(self._features),
            "enabled": enabled,
            "rollout": rollout,
            "disabled": len(self._features) - enabled - rollout,
            "user_overrides": sum(len(v) for v in self._user_overrides.values()),
        }

    def __repr__(self) -> str:
        return f"<FeatureToggles features={len(self._features)}>"
