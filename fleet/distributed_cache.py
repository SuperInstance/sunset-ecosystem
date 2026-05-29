from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CacheEntry:
    """A cache entry with TTL."""
    key: str
    value: Any
    timestamp: float
    ttl: float

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }


class DistributedCache:
    """
    Distributed cache for fleet state.

    In-memory cache with TTL, supports get/set/delete/clear.
    Can be synchronized across nodes.
    """

    def __init__(self, fleet_node_id: str = "default", default_ttl: float = 300.0):
        self.fleet_node_id = fleet_node_id
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a cache entry."""
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
        )
        self._cache[key] = entry

    def get(self, key: str) -> Optional[Any]:
        """Get a cache entry, returns None if expired or missing."""
        entry = self._cache.get(key)
        if not entry:
            self._misses += 1
            return None
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.value

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache = {}

    def keys(self) -> List[str]:
        """Get all non-expired keys."""
        self._cleanup()
        return list(self._cache.keys())

    def _cleanup(self) -> int:
        """Remove expired entries."""
        expired = [k for k, e in self._cache.items() if e.is_expired()]
        for k in expired:
            del self._cache[k]
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def export_json(self) -> str:
        """Export cache as JSON."""
        return json.dumps({
            "node": self.fleet_node_id,
            "keys": list(self._cache.keys()),
            "stats": self.get_stats(),
        }, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
