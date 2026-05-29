"""Simple distributed cache with TTL and invalidation.

Provides a lightweight in-memory cache with TTL (time-to-live), explicit
invalidation, and hit/miss tracking. Supports namespacing for multi-tenant
usage. Used for fleet data caching, temporary vector store, and
intermediate computation results.

Usage:
    cache = DistributedCache(default_ttl_sec=300)
    cache.set("key-1", "value-1", ttl_sec=60)
    assert cache.get("key-1") == "value-1"
    cache.invalidate("key-1")
    assert cache.get("key-1") is None
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class DistributedCache:
    """
    Lightweight distributed cache with TTL.

    :param default_ttl_sec: Default TTL in seconds.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        default_ttl_sec: float = 300.0,
        clock: Optional[callable] = None,
    ):
        self._default_ttl = default_ttl_sec
        self._clock = clock or time.time
        self._data: Dict[str, Any] = {}
        self._expires: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0
        self._sets = 0

    # ------------------------------------------------------------------
    # Get / Set
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        :param key: Cache key.
        :returns: Value or None if expired/missing.
        """
        if key not in self._data:
            self._misses += 1
            return None
        if self._expires.get(key, float("inf")) <= self._clock():
            self.invalidate(key)
            self._misses += 1
            return None
        self._hits += 1
        return self._data[key]

    def set(self, key: str, value: Any, ttl_sec: Optional[float] = None) -> None:
        """
        Set a cache entry.

        :param key: Cache key.
        :param value: Value to store.
        :param ttl_sec: TTL (uses default if None).
        """
        self._data[key] = value
        self._expires[key] = self._clock() + (ttl_sec or self._default_ttl)
        self._sets += 1

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        if key in self._data:
            del self._data[key]
            self._expires.pop(key, None)
            return True
        return False

    def invalidate(self, key: str) -> bool:
        """Alias for delete."""
        return self.delete(key)

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate keys matching a prefix pattern.

        :param pattern: Key prefix to match.
        :returns: Number of keys removed.
        """
        to_remove = [k for k in self._data.keys() if k.startswith(pattern)]
        for k in to_remove:
            self.delete(k)
        return len(to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._data.clear()
        self._expires.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        """List all cache keys."""
        return list(self._data.keys())

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL for a key."""
        if key not in self._expires:
            return None
        remaining = self._expires[key] - self._clock()
        return max(0.0, remaining) if remaining > 0 else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        # Clean expired keys first for accurate size
        expired = [k for k, exp in self._expires.items() if exp <= self._clock()]
        for k in expired:
            self.invalidate(k)
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        return {
            "size": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "hit_rate": round(hit_rate, 4),
        }

    def __repr__(self) -> str:
        return f"<DistributedCache size={len(self._data)} hit_rate={self.stats()['hit_rate']}>"
