"""Retry policy with backoff strategies.

Implements configurable retry policies with multiple backoff strategies:
fixed, exponential, jittered exponential, and custom. Used for fleet
RPC resilience, message delivery guarantees, and transient failure handling.

Usage:
    policy = RetryPolicy(max_retries=3, backoff="exponential", base_delay_sec=1.0)
    delay = policy.next_delay(attempt=1)  # 2.0 seconds
    delay = policy.next_delay(attempt=2)  # 4.0 seconds
    can_retry = policy.should_retry(attempt=3)  # False (max reached)
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, Optional


class RetryPolicy:
    """
    Retry policy with configurable backoff strategies.

    :param max_retries: Maximum number of retry attempts.
    :param backoff: Backoff strategy ("fixed", "exponential", "jitter").
    :param base_delay_sec: Base delay between retries.
    :param max_delay_sec: Maximum delay cap.
    :param jitter_factor: Jitter factor (0.0-1.0).
    :param retryable_errors: Optional list of error types to retry.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff: str = "exponential",
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 60.0,
        jitter_factor: float = 0.1,
        retryable_errors: Optional[list] = None,
    ):
        self._max_retries = max_retries
        self._backoff = backoff
        self._base_delay = base_delay_sec
        self._max_delay = max_delay_sec
        self._jitter_factor = max(0.0, min(1.0, jitter_factor))
        self._retryable_errors = set(retryable_errors) if retryable_errors else None
        self._attempts = 0
        self._total_delays = 0.0

    # ------------------------------------------------------------------
    # Delay calculation
    # ------------------------------------------------------------------

    def next_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt.

        :param attempt: Attempt number (0-indexed).
        :returns: Delay in seconds.
        """
        if self._backoff == "fixed":
            delay = self._base_delay
        elif self._backoff == "exponential":
            delay = self._base_delay * (2**attempt)
        elif self._backoff == "jitter":
            base = self._base_delay * (2**attempt)
            jitter = base * self._jitter_factor * random.uniform(-1.0, 1.0)
            delay = base + jitter
        else:
            delay = self._base_delay

        delay = min(delay, self._max_delay)
        if delay < 0:
            delay = 0
        self._total_delays += delay
        return delay

    # ------------------------------------------------------------------
    # Retry decisions
    # ------------------------------------------------------------------

    def should_retry(self, attempt: int, error: Optional[Exception] = None) -> bool:
        """
        Check if a retry should be attempted.

        :param attempt: Current attempt number (0-indexed).
        :param error: Optional exception to check against retryable list.
        :returns: True if retry should be attempted.
        """
        if attempt >= self._max_retries:
            return False
        if self._retryable_errors and error:
            error_type = type(error).__name__
            if error_type not in self._retryable_errors:
                return False
        return True

    def execute(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.

        :param fn: Function to execute.
        :param args: Function arguments.
        :param kwargs: Function keyword arguments.
        :returns: Function result.
        :raises: Last exception if all retries exhausted.
        """
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                if not self.should_retry(attempt, e):
                    raise
                if attempt < self._max_retries:
                    delay = self.next_delay(attempt)
                    time.sleep(delay)
        raise last_error

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "max_retries": self._max_retries,
            "backoff": self._backoff,
            "base_delay": self._base_delay,
            "max_delay": self._max_delay,
            "jitter_factor": self._jitter_factor,
            "total_delays": self._total_delays,
        }

    def __repr__(self) -> str:
        return f"<RetryPolicy max_retries={self._max_retries} backoff={self._backoff}>"
