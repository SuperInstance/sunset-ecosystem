"""plugin_registry.py — Plugin management and discovery.

Provides:
1. Plugin registration with metadata
2. Plugin dependency resolution
3. Plugin lifecycle (load, unload, reload)
4. Capability-based discovery
5. Version compatibility checking

Usage:
    registry = PluginRegistry()
    registry.register("breeder", BreederPlugin(), version="1.0.0", capabilities=["breed"])
    plugin = registry.get("breeder")
    compatible = registry.resolve(["breeder", "evaluator"])
"""

from __future__ import annotations

__all__ = [
    "PluginRegistry",
    "PluginInfo",
    "PluginError",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Raised for plugin-related errors."""


@dataclass
class PluginInfo:
    """Metadata for a registered plugin."""

    name: str
    version: str
    capabilities: list[str]
    instance: Any
    dependencies: list[str]
    loaded_at: float
    active: bool = True


class PluginRegistry:
    """Plugin management and discovery."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInfo] = {}

    def register(
        self,
        name: str,
        instance: Any,
        version: str = "0.0.1",
        capabilities: list[str] | None = None,
        dependencies: list[str] | None = None,
    ) -> PluginInfo:
        """Register a plugin."""
        info = PluginInfo(
            name=name,
            version=version,
            capabilities=capabilities or [],
            instance=instance,
            dependencies=dependencies or [],
            loaded_at=time.time(),
        )
        self._plugins[name] = info
        logger.info(f"Registered plugin '{name}' v{version}")
        return info

    def get(self, name: str) -> Any:
        """Get plugin instance by name."""
        info = self._plugins.get(name)
        if info is None:
            raise PluginError(f"Plugin '{name}' not found")
        return info.instance

    def get_info(self, name: str) -> PluginInfo | None:
        """Get plugin metadata."""
        return self._plugins.get(name)

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        if name in self._plugins:
            del self._plugins[name]
            return True
        return False

    def list_plugins(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def find_by_capability(self, capability: str) -> list[str]:
        """Find plugins that provide a capability."""
        return [
            name
            for name, info in self._plugins.items()
            if capability in info.capabilities
        ]

    def resolve(self, required: list[str]) -> list[str] | None:
        """Resolve dependencies: return load order or None if unsatisfied."""
        # Simple topological sort (preserves requested order when no deps)
        loaded: set[str] = set()
        order: list[str] = []
        unresolved = list(dict.fromkeys(required))  # preserve order, dedupe

        while unresolved:
            progress = False
            for i, name in enumerate(list(unresolved)):
                info = self._plugins.get(name)
                if info is None:
                    logger.error(f"Required plugin '{name}' not registered")
                    return None
                # Check if all dependencies are satisfied
                deps_satisfied = all(d in loaded for d in info.dependencies)
                if deps_satisfied:
                    loaded.add(name)
                    order.append(name)
                    unresolved.pop(i)
                    progress = True
                    break
            if not progress:
                logger.error(f"Dependency cycle or missing plugin for: {unresolved}")
                return None

        return order

    def check_compatibility(self, name: str, min_version: str) -> bool:
        """Check if a plugin meets minimum version (simple string compare)."""
        info = self._plugins.get(name)
        if info is None:
            return False

        # Simple version comparison: split by dots and compare numerically
        def parse(v: str) -> list[int]:
            return [int(x) for x in v.split(".")]

        return parse(info.version) >= parse(min_version)

    def reload(self, name: str, new_instance: Any) -> bool:
        """Replace a plugin instance while keeping metadata."""
        info = self._plugins.get(name)
        if info is None:
            return False
        info.instance = new_instance
        info.loaded_at = time.time()
        return True

    def disable(self, name: str) -> bool:
        """Temporarily disable a plugin."""
        info = self._plugins.get(name)
        if info:
            info.active = False
            return True
        return False

    def enable(self, name: str) -> bool:
        """Re-enable a plugin."""
        info = self._plugins.get(name)
        if info:
            info.active = True
            return True
        return False

    def active_plugins(self) -> list[str]:
        """List active plugin names."""
        return [name for name, info in self._plugins.items() if info.active]

    def stats(self) -> dict[str, Any]:
        return {
            "total": len(self._plugins),
            "active": len(self.active_plugins()),
            "capabilities": sorted(
                {cap for info in self._plugins.values() for cap in info.capabilities}
            ),
        }

    def __repr__(self) -> str:
        return f"PluginRegistry(plugins={len(self._plugins)}, active={len(self.active_plugins())})"
