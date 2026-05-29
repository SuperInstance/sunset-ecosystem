"""Distributed cache with TTL and invalidation protocol.

In-memory cache shard with TTL eviction and explicit invalidation.
Used for fleet node state caching and breeding result memoization.

Usage:
    cache = DistributedCache(ttl_sec=60.0, max_size=1000)
    cache.set("room-alpha:trap", trap_state)
    state = cache.get("room-alpha:trap")
    cache.invalidate("room-alpha:*")
"""
from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CacheEntry:
    """A cached value with metadata."""

    def __init__(self, key: str, value: Any, ttl: float, created: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created = created
        self.access_count = 0
        self.last_accessed = created

    def is_expired(self, now: float) -> bool:
        return self.ttl > 0 and now - self.created > self.ttl

    def touch(self, now: float) -> None:
        self.access_count += 1
        self.last_accessed = now


class DistributedCache:
    """
    TTL-backed cache with pattern invalidation.

    :param ttl_sec: Default TTL for entries (0 = no expiry).
    :param max_size: Maximum number of entries (LRU eviction).
    :param cleanup_interval: How often to run background eviction (seconds).
    """

    def __init__(
        self,
        ttl_sec: float = 60.0,
        max_size: int = 1000,
        cleanup_interval: float = 30.0,
    ):
        self._ttl = ttl_sec
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._store: Dict[str, CacheEntry] = {}
        self._last_cleanup = time.time()
        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
        }

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Store a value."""
        now = time.time()
        self._maybe_cleanup(now)

        # Evict oldest if at capacity
        if len(self._store) >= self._max_size and key not in self._store:
            self._evict_oldest(1)

        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl if ttl is not None else self._ttl,
            created=now,
        )
        self._store[key] = entry

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value. Returns *default* on miss or expiry."""
        now = time.time()
        self._maybe_cleanup(now)

        entry = self._store.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return default
        if entry.is_expired(now):
            del self._store[key]
            self._stats["misses"] += 1
            return default
        entry.touch(now)
        self._stats["hits"] += 1
        return entry.value

    def delete(self, key: str) -> bool:
        """Delete a single key. Returns True if it existed."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def invalidate(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern. Returns count deleted."""
        now = time.time()
        to_delete = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
        for k in to_delete:
            del self._store[k]
        self._stats["invalidations"] += len(to_delete)
        return len(to_delete)

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key, _SENTINEL) is not _SENTINEL

    def keys(self) -> List[str]:
        """Return all non-expired keys."""
        now = time.time()
        return [k for k, e in self._store.items() if not e.is_expired(now)]

    def clear(self) -> None:
        """Clear all entries."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _maybe_cleanup(self, now: float) -> None:
        if now - self._last_cleanup < self._cleanup_interval:
            return
        expired = [k for k, e in self._store.items() if e.is_expired(now)]
        for k in expired:
            del self._store[k]
            self._stats["evictions"] += 1
        self._last_cleanup = now

    def _evict_oldest(self, n: int) -> None:
        items = sorted(self._store.values(), key=lambda e: e.last_accessed)
        for entry in items[:n]:
            del self._store[entry.key]
            self._stats["evictions"] += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def size(self) -> int:
        return len(self._store)

    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total

    def __repr__(self) -> str:
        return f"<DistributedCache size={len(self._store)} max={self._max_size}>"


_SENTINEL = object()
