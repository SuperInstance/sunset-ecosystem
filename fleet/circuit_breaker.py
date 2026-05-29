"""Circuit breaker pattern for fault tolerance.

Implements the circuit breaker pattern with closed, open, and half-open
states. Prevents cascade failures by failing fast when a service is
degraded. Used for fleet service resilience, API client protection, and
dependency isolation.

Usage:
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout_sec=30)
    try:
        result = cb.call(lambda: requests.get("http://api"))
    except CircuitBreakerOpen:
        # Fail fast
        ...
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Dict, Optional


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker with three states.

    :param failure_threshold: Number of failures before opening.
    :param recovery_timeout_sec: Seconds before attempting half-open.
    :param half_open_max_calls: Max calls allowed in half-open state.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        half_open_max_calls: int = 3,
        clock: Optional[callable] = None,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_sec
        self._half_open_max = half_open_max_calls
        self._clock = clock or time.time
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._total_calls = 0
        self._total_failures = 0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def call(self, fn: Callable[[], Any]) -> Any:
        """
        Execute a function through the circuit breaker.

        :param fn: Function to call.
        :returns: Function result.
        :raises CircuitBreakerOpen: If circuit is open.
        :raises Exception: If function raises in half-open state.
        """
        self._check_state()

        if self._state == CircuitState.OPEN:
            raise CircuitBreakerOpen("Circuit breaker is open")

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self._half_open_max:
                raise CircuitBreakerOpen("Circuit breaker half-open limit reached")
            self._half_open_calls += 1

        self._total_calls += 1

        try:
            result = fn()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _check_state(self) -> None:
        """Transition from open to half-open if timeout elapsed."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = self._clock() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._failures = 0
                    self._successes = 0

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self._half_open_max:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._half_open_calls = 0
        else:
            self._failures = max(0, self._failures - 1)

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._total_failures += 1
        self._failures += 1
        self._last_failure_time = self._clock()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failures >= self._failure_threshold:
            self._state = CircuitState.OPEN

    # ------------------------------------------------------------------
    # Manual control
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Manually open the circuit."""
        self._state = CircuitState.OPEN
        self._last_failure_time = self._clock()

    def close(self) -> None:
        """Manually close the circuit."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._half_open_calls = 0
        self._last_failure_time = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def state(self) -> str:
        """Get current circuit state."""
        return self._state.value

    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "failures": self._failures,
            "successes": self._successes,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
        }

    def __repr__(self) -> str:
        return f"<CircuitBreaker state={self._state.value} failures={self._failures}>"
