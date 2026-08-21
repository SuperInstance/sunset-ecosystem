from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import numpy as np


@dataclass
class A2APlugin:
    """A registered A2A plugin."""

    name: str
    version: str
    handler: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata,
            "enabled": self.enabled,
            "registered_at": self.registered_at,
        }


class A2APluginManager:
    """
    Plugin manager for A2A agents.

    Allows dynamic registration, discovery, and invocation of
    agent capabilities via plugins.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.plugins: Dict[str, A2APlugin] = {}
        self._hooks: Dict[str, List[str]] = {}  # hook_name -> [plugin_name]

    def register(
        self,
        name: str,
        version: str,
        handler: Callable,
        metadata: Optional[Dict[str, Any]] = None,
        hooks: Optional[List[str]] = None,
    ) -> A2APlugin:
        """Register a plugin."""
        plugin = A2APlugin(
            name=name,
            version=version,
            handler=handler,
            metadata=metadata or {},
        )
        self.plugins[name] = plugin

        for hook in hooks or []:
            self._hooks.setdefault(hook, []).append(name)

        return plugin

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        if name not in self.plugins:
            return False
        del self.plugins[name]

        # Remove from hooks
        for hook_names in self._hooks.values():
            if name in hook_names:
                hook_names.remove(name)

        return True

    def get(self, name: str) -> Optional[A2APlugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)

    def list_plugins(self, enabled_only: bool = False) -> List[A2APlugin]:
        """List all registered plugins."""
        plugins = list(self.plugins.values())
        if enabled_only:
            plugins = [p for p in plugins if p.enabled]
        return plugins

    def invoke(self, name: str, *args, **kwargs) -> Any:
        """Invoke a plugin's handler."""
        plugin = self.plugins.get(name)
        if not plugin:
            raise ValueError(f"Plugin not found: {name}")
        if not plugin.enabled:
            raise ValueError(f"Plugin disabled: {name}")
        return plugin.handler(*args, **kwargs)

    def invoke_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Invoke all plugins registered for a hook."""
        results = []
        for plugin_name in self._hooks.get(hook_name, []):
            plugin = self.plugins.get(plugin_name)
            if plugin and plugin.enabled:
                try:
                    result = plugin.handler(*args, **kwargs)
                    results.append(
                        {
                            "plugin": plugin_name,
                            "result": result,
                            "success": True,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "plugin": plugin_name,
                            "error": str(e),
                            "success": False,
                        }
                    )
        return results

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        if name not in self.plugins:
            return False
        self.plugins[name].enabled = True
        return True

    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        if name not in self.plugins:
            return False
        self.plugins[name].enabled = False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            "total": len(self.plugins),
            "enabled": sum(1 for p in self.plugins.values() if p.enabled),
            "hooks": len(self._hooks),
            "plugins": list(self.plugins.keys()),
        }

    def export_manifest(self) -> str:
        """Export plugin manifest as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "plugins": [p.to_dict() for p in self.plugins.values()],
                "hooks": {k: v for k, v in self._hooks.items()},
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
