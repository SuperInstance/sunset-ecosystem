"""feature_flags.py — Dynamic feature toggle system.

Provides:
1. Boolean feature flags with default values
2. Percentage-based rollouts (e.g., 10% of agents)
3. Agent-specific overrides
4. Flag change notifications
5. Persist/restore flag state

Usage:
    flags = FeatureFlags()
    flags.define("new_breeder", default=False)
    if flags.is_enabled("new_breeder", agent_id="agent-1"):
        run_new_breeder()
"""
from __future__ import annotations

__all__ = [
    "FeatureFlags",
    "FlagDefinition",
]

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class FlagDefinition:
    """Definition of a feature flag."""
    name: str
    default: bool = False
    rollout_percent: float = 0.0  # 0-100
    overrides: dict[str, bool] = field(default_factory=dict)
    description: str = ""


class FeatureFlags:
    """Dynamic feature toggle system."""

    def __init__(self) -> None:
        self._flags: dict[str, FlagDefinition] = {}
        self._callbacks: list[Callable[[str, bool], None]] = []

    def define(
        self,
        name: str,
        default: bool = False,
        rollout_percent: float = 0.0,
        description: str = "",
    ) -> FlagDefinition:
        """Define a new feature flag."""
        flag = FlagDefinition(
            name=name,
            default=default,
            rollout_percent=rollout_percent,
            description=description,
        )
        self._flags[name] = flag
        logger.info(f"Defined flag '{name}' (default={default})")
        return flag

    def is_enabled(self, name: str, agent_id: str = "") -> bool:
        """Check if a feature is enabled for an agent."""
        if name not in self._flags:
            return False

        flag = self._flags[name]

        # Check agent-specific override
        if agent_id in flag.overrides:
            return flag.overrides[agent_id]

        # Check rollout percentage
        if flag.rollout_percent > 0.0 and agent_id:
            hash_val = int(hashlib.md5(f"{name}:{agent_id}".encode()).hexdigest(), 16)
            percent = (hash_val % 10000) / 100.0
            return percent < flag.rollout_percent

        return flag.default

    def set_override(self, name: str, agent_id: str, value: bool) -> None:
        """Set an agent-specific override."""
        if name not in self._flags:
            raise ValueError(f"Flag '{name}' not defined")
        self._flags[name].overrides[agent_id] = value

    def remove_override(self, name: str, agent_id: str) -> None:
        if name in self._flags and agent_id in self._flags[name].overrides:
            del self._flags[name].overrides[agent_id]

    def set_default(self, name: str, value: bool) -> None:
        """Change the default value of a flag."""
        if name not in self._flags:
            raise ValueError(f"Flag '{name}' not defined")
        old = self._flags[name].default
        self._flags[name].default = value
        if old != value:
            for cb in self._callbacks:
                try:
                    cb(name, value)
                except Exception as e:
                    logger.warning(f"Flag callback error: {e}")

    def set_rollout(self, name: str, percent: float) -> None:
        """Set rollout percentage."""
        if name not in self._flags:
            raise ValueError(f"Flag '{name}' not defined")
        self._flags[name].rollout_percent = max(0.0, min(100.0, percent))

    def on_change(self, callback: Callable[[str, bool], None]) -> None:
        """Register a callback for flag changes."""
        self._callbacks.append(callback)

    # ── query ──────────────────────────────────────────

    def all_flags(self) -> dict[str, FlagDefinition]:
        return dict(self._flags)

    def enabled_flags(self, agent_id: str = "") -> list[str]:
        return [name for name in self._flags if self.is_enabled(name, agent_id)]

    def report(self) -> dict[str, Any]:
        return {
            name: {
                "default": flag.default,
                "rollout": flag.rollout_percent,
                "overrides": len(flag.overrides),
            }
            for name, flag in self._flags.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "default": flag.default,
                "rollout_percent": flag.rollout_percent,
                "overrides": dict(flag.overrides),
                "description": flag.description,
            }
            for name, flag in self._flags.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureFlags":
        ff = cls()
        for name, flag_data in data.items():
            ff.define(
                name=name,
                default=flag_data.get("default", False),
                rollout_percent=flag_data.get("rollout_percent", 0.0),
                description=flag_data.get("description", ""),
            )
            ff._flags[name].overrides = dict(flag_data.get("overrides", {}))
        return ff

    def __repr__(self) -> str:
        return f"FeatureFlags(flags={len(self._flags)})"
