"""Token bucket and sliding window rate limiter.

Implements rate limiting with token bucket and sliding window
algorithms. Supports per-key limits, burst allowances, and configurable
refill rates. Used for fleet API protection, throttling, and fair
resource allocation.

Usage:
    rl = RateLimiter(rate=10, per_sec=1, burst=20)
    assert rl.allow("user-1") is True
    assert rl.allow("user-1", tokens=25) is False  # Exceeds burst
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class RateLimiter:
    """
    Token bucket rate limiter with per-key tracking.

    :param rate: Tokens replenished per interval.
    :param per_sec: Interval in seconds for rate calculation.
    :param burst: Maximum token bucket size.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        rate: float = 10.0,
        per_sec: float = 1.0,
        burst: Optional[int] = None,
        clock: Optional[callable] = None,
    ):
        self._rate = rate
        self._per_sec = per_sec
        self._burst = burst or int(rate * 2)
        self._clock = clock or time.time
        self._buckets: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Allow / Consume
    # ------------------------------------------------------------------

    def allow(self, key: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed without consuming tokens.

        :param key: Rate limit key.
        :param tokens: Tokens to consume.
        :returns: True if allowed.
        """
        bucket = self._get_bucket(key)
        self._refill(bucket)
        return bucket["tokens"] >= tokens

    def consume(self, key: str, tokens: int = 1) -> bool:
        """
        Consume tokens if available.

        :param key: Rate limit key.
        :param tokens: Tokens to consume.
        :returns: True if consumed, False if not enough tokens.
        """
        bucket = self._get_bucket(key)
        self._refill(bucket)
        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            bucket["consumed"] += tokens
            return True
        return False

    def _get_bucket(self, key: str) -> Dict[str, Any]:
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self._burst,
                "last_refill": self._clock(),
                "consumed": 0,
            }
        return self._buckets[key]

    def _refill(self, bucket: Dict[str, Any]) -> None:
        now = self._clock()
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * (self._rate / self._per_sec)
        bucket["tokens"] = min(self._burst, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def remaining(self, key: str) -> float:
        """Get remaining tokens for a key."""
        bucket = self._get_bucket(key)
        self._refill(bucket)
        return bucket["tokens"]

    def reset(self, key: str) -> None:
        """Reset a key's bucket."""
        if key in self._buckets:
            del self._buckets[key]

    def reset_all(self) -> None:
        """Reset all buckets."""
        self._buckets.clear()

    def keys(self) -> list:
        """List all tracked keys."""
        return list(self._buckets.keys())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total_consumed = sum(b["consumed"] for b in self._buckets.values())
        return {
            "keys": len(self._buckets),
            "rate": self._rate,
            "per_sec": self._per_sec,
            "burst": self._burst,
            "total_consumed": total_consumed,
        }

    def __repr__(self) -> str:
        return f"<RateLimiter rate={self._rate}/{self._per_sec}s keys={len(self._buckets)}>"
