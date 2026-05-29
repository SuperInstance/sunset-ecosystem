"""Request deduplication by key with TTL.

Prevents duplicate request processing by tracking recently seen request
keys with time-to-live expiration. Supports configurable TTL and
automatic cleanup. Used for fleet idempotency, deduplicated event
handling, and safe retry semantics.

Usage:
    dedup = RequestDeduplicator(ttl_sec=60)
    assert dedup.is_duplicate("req-1") is False  # First time
    assert dedup.is_duplicate("req-1") is True   # Duplicate
    dedup.forget("req-1")
    assert dedup.is_duplicate("req-1") is False
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class RequestDeduplicator:
    """
    Request deduplicator with TTL-based expiration.

    :param ttl_sec: Time-to-live for deduplication entries.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        ttl_sec: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._ttl = ttl_sec
        self._clock = clock or time.time
        self._seen: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def is_duplicate(self, key: str) -> bool:
        """
        Check if a request key is a duplicate.

        :param key: Request identifier.
        :returns: True if seen before and not expired.
        """
        now = self._clock()
        # Clean expired entries
        expired = [k for k, ts in self._seen.items() if ts + self._ttl <= now]
        for k in expired:
            del self._seen[k]

        if key in self._seen:
            self._hits += 1
            return True
        self._seen[key] = now
        self._misses += 1
        return False

    def mark_seen(self, key: str) -> None:
        """Explicitly mark a key as seen."""
        self._seen[key] = self._clock()

    def forget(self, key: str) -> bool:
        """Remove a key from deduplication tracking."""
        if key in self._seen:
            del self._seen[key]
            return True
        return False

    def forget_pattern(self, prefix: str) -> int:
        """
        Forget all keys matching a prefix.

        :param prefix: Key prefix to match.
        :returns: Number of keys removed.
        """
        to_remove = [k for k in self._seen.keys() if k.startswith(prefix)]
        for k in to_remove:
            del self._seen[k]
        return len(to_remove)

    def clear(self) -> None:
        """Clear all tracked keys."""
        self._seen.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        """List all currently tracked keys."""
        now = self._clock()
        return [k for k, ts in self._seen.items() if ts + self._ttl > now]

    def ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL for a key."""
        if key not in self._seen:
            return None
        remaining = self._seen[key] + self._ttl - self._clock()
        return max(0.0, remaining) if remaining > 0 else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.keys()),
            "ttl": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / (self._hits + self._misses), 4) if (self._hits + self._misses) > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return f"<RequestDeduplicator size={len(self.keys())} ttl={self._ttl}>"
