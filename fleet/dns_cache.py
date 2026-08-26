"""DNS cache with TTL and negative caching.

Caches DNS resolution results with configurable TTL, supports negative
caching (NXDOMAIN) and automatic expiration. Used for fleet service
discovery, reducing DNS lookup latency, and resilience against DNS
failures.

Usage:
    cache = DNSCache(default_ttl_sec=300)
    cache.put("svc-a.fleet.local", "192.168.1.1", ttl_sec=60)
    assert cache.resolve("svc-a.fleet.local") == ["192.168.1.1"]
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class DNSCache:
    """
    DNS cache with TTL and negative caching.

    :param default_ttl_sec: Default TTL for cached entries.
    :param negative_ttl_sec: TTL for negative cache entries (NXDOMAIN).
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        default_ttl_sec: float = 300.0,
        negative_ttl_sec: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._default_ttl = default_ttl_sec
        self._negative_ttl = negative_ttl_sec
        self._clock = clock or time.time
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._negative_hits = 0

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def put(
        self,
        hostname: str,
        addresses: List[str],
        ttl_sec: Optional[float] = None,
    ) -> None:
        """
        Cache a DNS resolution result.

        :param hostname: Hostname to cache.
        :param addresses: List of IP addresses.
        :param ttl_sec: TTL (uses default if None).
        """
        self._cache[hostname] = {
            "addresses": list(addresses),
            "expires": self._clock() + (ttl_sec or self._default_ttl),
            "negative": False,
        }

    def put_negative(self, hostname: str) -> None:
        """Cache a negative result (NXDOMAIN)."""
        self._cache[hostname] = {
            "addresses": [],
            "expires": self._clock() + self._negative_ttl,
            "negative": True,
        }

    def resolve(self, hostname: str) -> Optional[List[str]]:
        """
        Resolve a hostname from cache.

        :param hostname: Hostname to resolve.
        :returns: List of addresses, empty list for negative cache, or None if expired/missing.
        """
        entry = self._cache.get(hostname)
        if not entry:
            self._misses += 1
            return None

        if entry["expires"] <= self._clock():
            del self._cache[hostname]
            self._misses += 1
            return None

        if entry["negative"]:
            self._negative_hits += 1
            return []

        self._hits += 1
        return list(entry["addresses"])

    def get_one(self, hostname: str) -> Optional[str]:
        """Get a single address (first in list)."""
        addresses = self.resolve(hostname)
        if addresses:
            return addresses[0]
        return None

    def invalidate(self, hostname: str) -> bool:
        """Remove a cache entry."""
        if hostname in self._cache:
            del self._cache[hostname]
            return True
        return False

    def invalidate_pattern(self, suffix: str) -> int:
        """
        Invalidate entries matching a suffix.

        :param suffix: Hostname suffix to match.
        :returns: Number of entries removed.
        """
        to_remove = [h for h in self._cache.keys() if h.endswith(suffix)]
        for h in to_remove:
            del self._cache[h]
        return len(to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def hostnames(self) -> List[str]:
        """List all cached hostnames."""
        return list(self._cache.keys())

    def ttl(self, hostname: str) -> Optional[float]:
        """Get remaining TTL for a hostname."""
        entry = self._cache.get(hostname)
        if not entry:
            return None
        remaining = entry["expires"] - self._clock()
        return max(0.0, remaining) if remaining > 0 else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        # Clean expired entries
        expired = [h for h, e in self._cache.items() if e["expires"] <= self._clock()]
        for h in expired:
            del self._cache[h]

        total = self._hits + self._misses + self._negative_hits
        hit_rate = (
            self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0
            else 0.0
        )

        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "negative_hits": self._negative_hits,
            "hit_rate": round(hit_rate, 4),
            "total_lookups": total,
        }

    def __repr__(self) -> str:
        return f"<DNSCache size={len(self._cache)} hits={self._hits} misses={self._misses}>"
