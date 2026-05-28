"""config_reloader.py — Hot-reload configuration without restart.

Provides:
1. Watch config files for changes (mtime polling)
2. Reload callbacks for individual sections
3. Validation of new config before applying
4. Rollback on validation failure
5. Change notifications

Usage:
    reloader = ConfigReloader()
    reloader.register("breeding", path="configs/breeding.yaml", validator=validate_breeding)
    reloader.register("gateway", path="configs/gateway.yaml", validator=validate_gateway)
    reloader.check()  # Returns list of changed sections
"""
from __future__ import annotations

__all__ = [
    "ConfigReloader",
    "ConfigSection",
    "ReloadResult",
]

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """A watched configuration section."""
    name: str
    path: str
    validator: Callable[[Any], bool] | None = None
    current_config: Any = None
    last_mtime: float = 0.0
    last_loaded: float = 0.0


@dataclass
class ReloadResult:
    """Result of a reload check."""
    name: str
    changed: bool
    success: bool
    message: str = ""
    old_config: Any = None
    new_config: Any = None


class ConfigReloader:
    """Hot-reload configuration from files."""

    def __init__(self, load_fn: Callable[[str], Any] | None = None) -> None:
        self._load_fn = load_fn or self._default_load
        self._sections: dict[str, ConfigSection] = {}
        self._callbacks: list[Callable[[str, Any, Any], None]] = []

    @staticmethod
    def _default_load(path: str) -> Any:
        """Default loader: reads as text. Override for YAML/JSON/TOML."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def register(
        self,
        name: str,
        path: str,
        validator: Callable[[Any], bool] | None = None,
    ) -> None:
        """Register a config file to watch."""
        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}")
        self._sections[name] = ConfigSection(
            name=name,
            path=path,
            validator=validator,
        )

    def check(self) -> list[ReloadResult]:
        """Check all sections for changes. Returns list of results."""
        results = []
        for name, section in self._sections.items():
            result = self._check_section(section)
            if result.changed:
                results.append(result)
        return results

    def _check_section(self, section: ConfigSection) -> ReloadResult:
        if not os.path.exists(section.path):
            return ReloadResult(
                name=section.name,
                changed=False,
                success=False,
                message="file not found",
            )

        mtime = os.path.getmtime(section.path)
        if mtime == section.last_mtime:
            return ReloadResult(
                name=section.name,
                changed=False,
                success=True,
            )

        # File changed — attempt reload
        try:
            new_config = self._load_fn(section.path)
        except Exception as e:
            return ReloadResult(
                name=section.name,
                changed=True,
                success=False,
                message=f"load failed: {e}",
            )

        # Validate if validator provided
        if section.validator is not None:
            if not section.validator(new_config):
                return ReloadResult(
                    name=section.name,
                    changed=True,
                    success=False,
                    message="validation failed",
                    old_config=section.current_config,
                    new_config=new_config,
                )

        # Apply change
        old_config = section.current_config
        section.current_config = new_config
        section.last_mtime = mtime
        section.last_loaded = time.time()

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(section.name, old_config, new_config)
            except Exception as e:
                logger.warning(f"Config reload callback error: {e}")

        return ReloadResult(
            name=section.name,
            changed=True,
            success=True,
            old_config=old_config,
            new_config=new_config,
        )

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Register a callback for config changes."""
        self._callbacks.append(callback)

    def get(self, name: str) -> Any | None:
        """Get current config for a section."""
        section = self._sections.get(name)
        return section.current_config if section else None

    def sections(self) -> list[str]:
        """List registered section names."""
        return list(self._sections.keys())

    def force_reload(self, name: str) -> ReloadResult:
        """Force reload of a section."""
        section = self._sections.get(name)
        if section is None:
            return ReloadResult(
                name=name,
                changed=False,
                success=False,
                message="section not registered",
            )
        section.last_mtime = 0.0  # Force mtime mismatch
        return self._check_section(section)

    def __repr__(self) -> str:
        return f"ConfigReloader(sections={len(self._sections)})"
