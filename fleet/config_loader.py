"""Configuration loader with environment override and validation.

Loads configuration from nested dicts with environment variable
overrides, type validation, and default values. Used for fleet service
configuration, deployment settings, and runtime parameters.

Usage:
    loader = ConfigLoader(defaults={"db": {"host": "localhost"}})
    config = loader.load()
    assert config["db.host"] == "localhost"
    loader.set("db.host", "prod-db")
    assert loader.get("db.host") == "prod-db"
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class ConfigLoader:
    """
    Configuration loader with environment override.

    :param defaults: Default configuration dict.
    """

    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {}
        if defaults:
            self._deep_update(self._config, defaults)
        self._validators: Dict[str, callable] = {}

    # ------------------------------------------------------------------
    # Load / Get / Set
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """Return full configuration dict."""
        return dict(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated key.

        :param key: Dot-separated key (e.g., "db.host").
        :param default: Default value if key not found.
        :returns: Configuration value or default.
        """
        parts = key.split(".")
        current = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by dot-separated key.

        :param key: Dot-separated key.
        :param value: Value to set.
        """
        parts = key.split(".")
        current = self._config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    # ------------------------------------------------------------------
    # Environment override
    # ------------------------------------------------------------------

    def override_from_env(self, prefix: str = "FLEET_") -> int:
        """
        Override config from environment variables.

        :param prefix: Environment variable prefix.
        :returns: Number of overrides applied.
        """
        count = 0
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("__", ".")
                # Try to parse as int, float, bool, or string
                parsed = self._parse_value(value)
                self.set(config_key, parsed)
                count += 1
        return count

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to int, float, bool, or string."""
        lower = value.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def add_validator(self, key: str, fn: callable) -> None:
        """
        Add a validator for a key.

        :param key: Configuration key.
        :param fn: Validator function(value) -> bool.
        """
        self._validators[key] = fn

    def validate(self) -> List[str]:
        """
        Validate all configuration values.

        :returns: List of invalid keys.
        """
        invalid = []
        for key, validator in self._validators.items():
            value = self.get(key)
            if value is not None and not validator(value):
                invalid.append(key)
        return invalid

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Deep update target dict with source dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "keys": self._count_keys(self._config),
            "validators": len(self._validators),
        }

    def _count_keys(self, d: Dict) -> int:
        count = 0
        for v in d.values():
            count += 1
            if isinstance(v, dict):
                count += self._count_keys(v)
        return count

    def __repr__(self) -> str:
        return f"<ConfigLoader keys={self.stats()['keys']}>"
