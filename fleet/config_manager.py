from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ConfigEntry:
    """A single configuration entry."""
    key: str
    value: Any
    source: str
    timestamp: float
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "version": self.version,
            "metadata": self.metadata,
        }


class ConfigManager:
    """
    Fleet configuration management.

    Hierarchical config with defaults, overrides, and versioning.
    Supports per-node, per-service, and per-breeder configuration.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._configs: Dict[str, ConfigEntry] = {}
        self._defaults: Dict[str, Any] = {
            "breeding.population_size": 50,
            "breeding.dimensions": 10,
            "breeding.max_generations": 100,
            "flux.hard_constraints": True,
            "metronome.interval_ms": 1000,
            "mesh.gossip_interval_ms": 5000,
        }

    def set(self, key: str, value: Any, source: str = "manual",
            metadata: Optional[Dict[str, Any]] = None):
        """Set a configuration value."""
        entry = ConfigEntry(
            key=key,
            value=value,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._configs[key] = entry

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if key in self._configs:
            return self._configs[key].value
        if key in self._defaults:
            return self._defaults[key]
        return default

    def get_with_metadata(self, key: str) -> Optional[ConfigEntry]:
        """Get full config entry with metadata."""
        return self._configs.get(key)

    def has(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._configs or key in self._defaults

    def delete(self, key: str) -> bool:
        """Delete a config key."""
        if key in self._configs:
            del self._configs[key]
            return True
        return False

    def get_all(self) -> Dict[str, Any]:
        """Get all config values (merged with defaults)."""
        result = dict(self._defaults)
        for key, entry in self._configs.items():
            result[key] = entry.value
        return result

    def get_by_source(self, source: str) -> List[ConfigEntry]:
        """Get all configs from a specific source."""
        return [e for e in self._configs.values() if e.source == source]

    def get_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get all configs with a given prefix."""
        return {
            k: e.value
            for k, e in self._configs.items()
            if k.startswith(prefix)
        }

    def export_json(self) -> str:
        """Export all configs as JSON."""
        return json.dumps({
            "node": self.fleet_node_id,
            "configs": {k: e.to_dict() for k, e in self._configs.items()},
            "defaults": self._defaults,
        }, indent=2)

    def load_json(self, json_str: str):
        """Load configs from JSON."""
        data = json.loads(json_str)
        for key, entry_dict in data.get("configs", {}).items():
            self.set(key, entry_dict["value"], entry_dict.get("source", "loaded"))

    def get_stats(self) -> Dict[str, Any]:
        """Get config manager statistics."""
        sources = {}
        for entry in self._configs.values():
            sources[entry.source] = sources.get(entry.source, 0) + 1
        return {
            "total_keys": len(self._configs),
            "default_keys": len(self._defaults),
            "sources": sources,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
            "stats": self.get_stats(),
            "configs": len(self._configs),
        }
