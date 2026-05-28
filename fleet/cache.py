"""cache.py — LRU + TTL caching layer for fleet-wide shared state.

Provides:
1. LRU eviction with configurable max size
2. TTL-based expiration per entry
3. Thread-safe operations
4. Hit/miss statistics
5. Bulk operations (get_many, set_many, delete_many)

Usage:
    cache = FleetCache(max_size=1000, default_ttl=60.0)
    cache.set("key", value, ttl=30.0)
    value = cache.get("key")
    stats = cache.stats()
"""
from __future__ import annotations

__all__ = [
    "FleetCache",
    "CacheEntry",
    "CacheStats",
]

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class CacheEntry:
    """A cached value with metadata."""
    value: Any
    expires_at: float
    created_at: float


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int
    misses: int
    evictions: int
    expired: int
    size: int
    max_size: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class FleetCache:
    """Thread-safe LRU + TTL cache."""

    def __init__(self, max_size: int = 1000, default_ttl: float | None = None) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    # ── core operations ─────────────────────────────────

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get a value from cache. Returns default if missing or expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return default

            if entry.expires_at is not None and entry.expires_at < time.time():
                del self._cache[key]
                self._expired += 1
                self._misses += 1
                return default

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set a value in cache."""
        with self._lock:
            expires = None
            if ttl is not None:
                expires = time.time() + ttl
            elif self._default_ttl is not None:
                expires = time.time() + self._default_ttl

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires,
                created_at=time.time(),
            )
            self._cache.move_to_end(key)
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    # ── bulk operations ────────────────────────────────

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values."""
        return {k: v for k, v in ((k, self.get(k)) for k in keys) if v is not None}

    def set_many(self, items: dict[str, Any], ttl: float | None = None) -> None:
        """Set multiple values."""
        for key, value in items.items():
            self.set(key, value, ttl=ttl)

    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys. Returns count deleted."""
        return sum(1 for k in keys if self.delete(k))

    # ── eviction ─────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over capacity."""
        while len(self._cache) > self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1

    def expire_all(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, e in self._cache.items()
                if e.expires_at is not None and e.expires_at < now
            ]
            for k in expired_keys:
                del self._cache[k]
                self._expired += 1
            return len(expired_keys)

    # ── query ─────────────────────────────────────────

    def keys(self) -> list[str]:
        """Get all valid (non-expired) keys."""
        # Expire on read to keep clean
        return [k for k in list(self._cache.keys()) if self.has(k)]

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def max_size(self) -> int:
        return self._max_size

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            expired=self._expired,
            size=self.size(),
            max_size=self._max_size,
        )

    def __repr__(self) -> str:
        return f"FleetCache(size={self.size()}/{self._max_size}, hits={self._hits}, misses={self._misses})"
