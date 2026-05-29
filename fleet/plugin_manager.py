"""Plugin discovery and lifecycle management.

Discovers plugins by name or path, manages their lifecycle (load,
unload, reload), and provides dependency ordering. Used for fleet
extension points, service plugins, and modular capabilities.

Usage:
    mgr = PluginManager()
    mgr.register("logger", LoggerPlugin)
    mgr.load("logger")
    plugin = mgr.get("logger")
    mgr.unload("logger")
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type


class Plugin:
    """Base plugin interface."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class PluginManager:
    """
    Plugin lifecycle manager.
    """

    def __init__(self):
        self._registry: Dict[str, Type[Plugin]] = {}
        self._instances: Dict[str, Plugin] = {}
        self._dependencies: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        plugin_class: Type[Plugin],
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a plugin class."""
        self._registry[name] = plugin_class
        self._dependencies[name] = dependencies or []

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        if name in self._registry:
            self.unload(name)
            del self._registry[name]
            del self._dependencies[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin with dependencies."""
        if name in self._instances:
            return True
        if name not in self._registry:
            return False

        # Load dependencies first
        for dep in self._dependencies[name]:
            if dep not in self._instances:
                self.load(dep)

        plugin_class = self._registry[name]
        instance = plugin_class(name, config)
        instance.load()
        self._instances[name] = instance
        return True

    def unload(self, name: str) -> bool:
        """Unload a plugin."""
        instance = self._instances.get(name)
        if not instance:
            return False
        instance.unload()
        del self._instances[name]
        return True

    def reload(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Reload a plugin."""
        self.unload(name)
        return self.load(name, config)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Plugin]:
        """Get loaded plugin instance."""
        return self._instances.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if plugin is loaded."""
        return name in self._instances

    def loaded_plugins(self) -> List[str]:
        """List loaded plugin names."""
        return list(self._instances.keys())

    def available_plugins(self) -> List[str]:
        """List registered plugin names."""
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "registered": len(self._registry),
            "loaded": len(self._instances),
        }

    def __repr__(self) -> str:
        return f"<PluginManager registered={len(self._registry)} loaded={len(self._instances)}>"
