from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class RateLimiter:
    """
    Token bucket rate limiter.

    Enforces rate limits with burst capacity.
    """

    def __init__(self, name: str, rate: float, burst: float):
        self.name = name
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()

    def allow(self) -> bool:
        """Check if a request is allowed."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        return {
            "name": self.name,
            "rate": self.rate,
            "burst": self.burst,
            "tokens": self.tokens,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_status()


class RateLimiterPanel:
    """Panel of rate limiters for fleet-wide throttling."""

    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}

    def get(self, name: str, rate: float = 10.0, burst: float = 20.0) -> RateLimiter:
        """Get or create a rate limiter."""
        if name not in self.limiters:
            self.limiters[name] = RateLimiter(name, rate, burst)
        return self.limiters[name]

    def allow(self, name: str, rate: float = 10.0, burst: float = 20.0) -> bool:
        """Check if a request is allowed through a named limiter."""
        limiter = self.get(name, rate, burst)
        return limiter.allow()

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all limiters."""
        return {name: limiter.get_status() for name, limiter in self.limiters.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limiters": len(self.limiters),
            "status": self.get_all_status(),
        }
