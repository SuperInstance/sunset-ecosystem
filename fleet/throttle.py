"""Rate throttler with burst and sliding window.

Controls request rate with burst allowance and sliding window
 tracking. Used for fleet API rate limiting and backpressure.

Usage:
    throttle = Throttle(rate=10, burst=5)
    assert throttle.acquire()  # allowed
    assert throttle.acquire()  # allowed
    time.sleep(0.1)
    assert not throttle.acquire()  # rate limited
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class Throttle:
    """
    Token-bucket-style throttler with burst.

    :param rate: Requests per second.
    :param burst: Maximum burst allowance.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Optional[callable] = None,
    ):
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._clock = clock or time.time
        self._allowed = 0
        self._denied = 0

    # ------------------------------------------------------------------
    # Rate control
    # ------------------------------------------------------------------

    def acquire(self, tokens: float = 1.0) -> bool:
        """
        Attempt to acquire tokens.

        :param tokens: Tokens to acquire (default 1).
        :returns: True if allowed.
        """
        now = self._clock()
        elapsed = now - self._last_update
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_update = now
        if self._tokens >= tokens:
            self._tokens -= tokens
            self._allowed += 1
            return True
        self._denied += 1
        return False

    def wait_time(self, tokens: float = 1.0) -> float:
        """Return estimated seconds to wait before tokens are available."""
        now = self._clock()
        elapsed = now - self._last_update
        tokens_available = min(self._burst, self._tokens + elapsed * self._rate)
        if tokens_available >= tokens:
            return 0.0
        deficit = tokens - tokens_available
        return deficit / self._rate

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "rate": self._rate,
            "burst": self._burst,
            "tokens": self._tokens,
            "allowed": self._allowed,
            "denied": self._denied,
        }

    def __repr__(self) -> str:
        return f"<Throttle rate={self._rate} burst={self._burst} tokens={self._tokens:.1f}>"
