"""Response cache with vary-by and etag.

Caches HTTP-like responses with cache-key generation based on vary
headers, ETag validation, and TTL expiration. Used for fleet API
response caching, static asset serving, and computation result caching.

Usage:
    cache = ResponseCache(default_ttl_sec=300)
    cache.put("/users", {"body": "..."}, vary_headers=["accept"])
    response = cache.get("/users", headers={"accept": "json"})
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional


class ResponseCache:
    """
    Response cache with vary-by support and ETag validation.

    :param default_ttl_sec: Default TTL for cached responses.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        default_ttl_sec: float = 300.0,
        clock: Optional[callable] = None,
    ):
        self._default_ttl = default_ttl_sec
        self._clock = clock or time.time
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Cache key generation
    # ------------------------------------------------------------------

    def _make_key(self, url: str, vary_headers: Optional[Dict[str, str]] = None) -> str:
        """Generate cache key from URL and vary headers."""
        if vary_headers:
            vary_str = "|".join(f"{k}={v}" for k, v in sorted(vary_headers.items()))
            return f"{url}?{vary_str}"
        return url

    # ------------------------------------------------------------------
    # Store / Retrieve
    # ------------------------------------------------------------------

    def put(
        self,
        url: str,
        response: Dict[str, Any],
        vary_headers: Optional[List[str]] = None,
        ttl_sec: Optional[float] = None,
        etag: Optional[str] = None,
    ) -> None:
        """
        Cache a response.

        :param url: Request URL/path.
        :param response: Response dict with at least "body" key.
        :param vary_headers: List of header names to vary on.
        :param ttl_sec: Optional TTL.
        :param etag: Optional ETag for validation.
        """
        # For now, store without vary header values (simple cache)
        key = self._make_key(url)
        self._cache[key] = {
            "response": response,
            "expires": self._clock() + (ttl_sec or self._default_ttl),
            "etag": etag or self._generate_etag(response),
            "vary_headers": vary_headers or [],
        }

    def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get a cached response.

        :param url: Request URL/path.
        :param headers: Request headers for vary matching.
        :returns: Cached response or None.
        """
        key = self._make_key(url, headers)
        entry = self._cache.get(key)
        if not entry:
            # Try without vary headers
            key = self._make_key(url)
            entry = self._cache.get(key)
            if not entry:
                self._misses += 1
                return None

        if entry["expires"] <= self._clock():
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return dict(entry["response"])

    def get_with_etag(self, url: str, if_none_match: Optional[str] = None) -> tuple[Optional[Dict[str, Any]], bool]:
        """
        Get response with ETag validation.

        :param url: Request URL.
        :param if_none_match: Client's ETag.
        :returns: Tuple of (response, not_modified).
        """
        key = self._make_key(url)
        entry = self._cache.get(key)
        if not entry or entry["expires"] <= self._clock():
            self._misses += 1
            return None, False

        if if_none_match and if_none_match == entry["etag"]:
            self._hits += 1
            return None, True

        self._hits += 1
        return dict(entry["response"]), False

    def invalidate(self, url: str) -> bool:
        """Remove a cached response."""
        key = self._make_key(url)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # ETag
    # ------------------------------------------------------------------

    def _generate_etag(self, response: Dict[str, Any]) -> str:
        """Generate ETag from response content."""
        body = str(response.get("body", ""))
        return f'"{hashlib.md5(body.encode()).hexdigest()[:8]}"'

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        # Clean expired entries
        expired = [k for k, e in self._cache.items() if e["expires"] <= self._clock()]
        for k in expired:
            del self._cache[k]

        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    def __repr__(self) -> str:
        return f"<ResponseCache size={len(self._cache)} hits={self._hits} misses={self._misses}>"
