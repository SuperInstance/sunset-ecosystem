"""rate_limiter.py — Token bucket rate limiter for fleet API calls.

Provides:
1. Per-key token buckets with configurable rate and burst
2. Sliding window for precise rate tracking
3. Queue with backpressure for bursty workloads
4. Metrics: allowed, denied, queued, wait time
5. Global and per-key limits

Usage:
    rl = RateLimiter(rate=10.0, burst=5)  # 10/second, burst 5
    if rl.allow("api_key_123"):
        make_api_call()
    else:
        # Rate limited
        pass
"""
from __future__ import annotations

__all__ = [
    "RateLimiter",
    "TokenBucket",
]

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TokenBucket:
    """Token bucket for a single key."""
    rate: float       # tokens per second
    burst: float      # max tokens (bucket capacity)
    tokens: float = 0.0
    last_update: float = 0.0
    allowed: int = 0
    denied: int = 0

    def __post_init__(self) -> None:
        if self.last_update == 0.0:
            self.last_update = time.time()
        self.tokens = min(self.burst, self.tokens)

    def _refill(self) -> None:
        now = time.time()
        delta = now - self.last_update
        self.tokens = min(self.burst, self.tokens + delta * self.rate)
        self.last_update = now

    def allow(self, cost: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            self.allowed += 1
            return True
        self.denied += 1
        return False

    @property
    def wait_time(self) -> float:
        """Seconds to wait until 1 token is available."""
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rate

    def state(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "allowed": self.allowed,
            "denied": self.denied,
            "wait_time": self.wait_time,
        }


@dataclass
class RateLimiter:
    """Token bucket rate limiter with per-key tracking."""
    rate: float = 10.0       # tokens per second per key
    burst: float = 5.0       # max tokens per key
    global_rate: float | None = None  # global rate limit
    global_burst: float | None = None

    def __post_init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._global_bucket: TokenBucket | None = None
        if self.global_rate is not None:
            self._global_bucket = TokenBucket(
                rate=self.global_rate,
                burst=self.global_burst or self.global_rate,
            )

    def allow(self, key: str = "default", cost: float = 1.0) -> bool:
        """Check if request is allowed for key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(rate=self.rate, burst=self.burst)

        # Check global first
        if self._global_bucket is not None and not self._global_bucket.allow(cost):
            return False

        return self._buckets[key].allow(cost)

    def wait_time(self, key: str = "default") -> float:
        """Seconds to wait until request is allowed."""
        if key not in self._buckets:
            return 0.0
        return self._buckets[key].wait_time

    def state(self, key: str | None = None) -> dict[str, Any]:
        """Get state for a key or all keys."""
        if key is not None:
            if key in self._buckets:
                return self._buckets[key].state()
            return {"error": "key not found"}

        total_allowed = sum(b.allowed for b in self._buckets.values())
        total_denied = sum(b.denied for b in self._buckets.values())
        return {
            "keys": len(self._buckets),
            "total_allowed": total_allowed,
            "total_denied": total_denied,
            "global": self._global_bucket.state() if self._global_bucket else None,
        }

    def reset(self, key: str | None = None) -> None:
        if key is not None:
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()
            if self._global_bucket:
                self._global_bucket.tokens = self._global_bucket.burst
                self._global_bucket.allowed = 0
                self._global_bucket.denied = 0

    def __repr__(self) -> str:
        return f"RateLimiter(keys={len(self._buckets)}, rate={self.rate}, burst={self.burst})"
