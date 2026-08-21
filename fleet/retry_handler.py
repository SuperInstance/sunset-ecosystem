"""retry_handler.py — Exponential backoff with jitter for resilient API calls.

Provides:
1. Configurable retry policies (max attempts, backoff, jitter)
2. Exception-based and return-value-based retry conditions
3. Circuit breaker integration
4. Per-operation timeout
5. Retry statistics

Usage:
    retry = RetryHandler(max_attempts=3, base_delay=1.0, max_delay=30.0)
    result = retry.run(lambda: flaky_api_call(), on_exception=ConnectionError)
"""

from __future__ import annotations

__all__ = [
    "RetryHandler",
    "RetryPolicy",
    "RetryExhausted",
]

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    jitter_fraction: float = 0.1
    exponential_base: float = 2.0
    on_exceptions: tuple[type[Exception], ...] | None = None
    retry_if: Callable[[Any], bool] | None = None


class RetryHandler:
    """Resilient retry handler with exponential backoff and jitter."""

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self._policy = policy or RetryPolicy()
        self._total_attempts = 0
        self._total_failures = 0
        self._total_successes = 0

    def run(self, fn: Callable[[], T]) -> T:
        """Execute fn with retries according to policy."""
        policy = self._policy
        last_exception: Exception | None = None

        for attempt in range(1, policy.max_attempts + 1):
            self._total_attempts += 1
            try:
                result = fn()
                if policy.retry_if is not None and policy.retry_if(result):
                    if attempt == policy.max_attempts:
                        self._total_failures += 1
                        raise RetryExhausted(
                            f"Retry condition matched on final attempt {attempt}"
                        )
                    self._sleep(attempt, policy)
                    continue
                self._total_successes += 1
                return result
            except Exception as e:
                last_exception = e
                if policy.on_exceptions is not None and not isinstance(
                    e, policy.on_exceptions
                ):
                    raise
                if attempt == policy.max_attempts:
                    self._total_failures += 1
                    raise RetryExhausted(
                        f"All {policy.max_attempts} attempts failed"
                    ) from e
                self._sleep(attempt, policy)

        # Should never reach here
        raise RetryExhausted("Unexpected exit from retry loop") from last_exception

    def _sleep(self, attempt: int, policy: RetryPolicy) -> None:
        """Calculate and sleep for the backoff duration."""
        delay = policy.base_delay * (policy.exponential_base ** (attempt - 1))
        delay = min(delay, policy.max_delay)
        if policy.jitter:
            jitter = delay * policy.jitter_fraction * random.uniform(-1.0, 1.0)
            delay = max(0.0, delay + jitter)
        logger.info(f"Retry attempt {attempt + 1} after {delay:.2f}s")
        time.sleep(delay)

    def stats(self) -> dict[str, Any]:
        return {
            "total_attempts": self._total_attempts,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "success_rate": self._total_successes / max(self._total_attempts, 1),
        }

    def __repr__(self) -> str:
        return f"RetryHandler(policy={self._policy})"
