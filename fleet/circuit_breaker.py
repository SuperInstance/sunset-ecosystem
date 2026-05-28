"""circuit_breaker.py — Circuit breaker pattern for external service resilience.

Provides:
1. Three states: Closed (normal), Open (failing fast), Half-Open (testing recovery)
2. Failure threshold and timeout-based recovery
3. Success/failure half-open probes
4. Fallback execution
5. Metrics: trips, successes, failures

Usage:
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
    result = cb.call(lambda: external_api_request())
    if result is None:
        # Circuit is open, handle fallback
"""
from __future__ import annotations

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitStats",
]

import enum
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    state: CircuitState
    failures: int = 0
    successes: int = 0
    trips: int = 0
    last_failure_time: float | None = None
    consecutive_failures: int = 0


class CircuitBreaker:
    """Circuit breaker for resilient external calls."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        exception_types: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exception_types = exception_types or (Exception,)
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats(state=CircuitState.CLOSED)
        self._half_open_calls = 0

    # ── call ───────────────────────────────────────────

    def call(self, fn: Callable[[], T], fallback: Callable[[], T] | None = None) -> T | None:
        """Execute fn with circuit breaker protection."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
            else:
                logger.warning(f"Circuit '{self.name}' is OPEN, fast-failing")
                if fallback:
                    return fallback()
                return None

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                logger.warning(f"Circuit '{self.name}' half-open limit reached")
                if fallback:
                    return fallback()
                return None
            self._half_open_calls += 1

        try:
            result = fn()
            self._on_success()
            return result
        except self.exception_types as e:
            self._on_failure()
            logger.warning(f"Circuit '{self.name}' call failed: {e}")
            if fallback:
                return fallback()
            raise

    # ── state machine ────────────────────────────────

    def _on_success(self) -> None:
        self._stats.consecutive_failures = 0
        self._stats.successes += 1
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._transition_to(CircuitState.CLOSED)

    def _on_failure(self) -> None:
        self._stats.consecutive_failures += 1
        self._stats.failures += 1
        self._stats.last_failure_time = time.time()
        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)

    def _should_attempt_reset(self) -> bool:
        if self._stats.last_failure_time is None:
            return True
        elapsed = time.time() - self._stats.last_failure_time
        return elapsed >= self.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        self._stats.state = new_state
        if new_state == CircuitState.OPEN and old_state != CircuitState.OPEN:
            self._stats.trips += 1
            logger.warning(f"Circuit '{self.name}' TRIPPED: {old_state.value} -> OPEN")
        elif new_state == CircuitState.CLOSED and old_state == CircuitState.OPEN:
            logger.info(f"Circuit '{self.name}' RESET: OPEN -> CLOSED")
        logger.debug(f"Circuit '{self.name}' state: {old_state.value} -> {new_state.value}")

    # ── query ─────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            # Lazy transition to half-open on query
            pass
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def stats(self) -> CircuitStats:
        return CircuitStats(
            state=self._state,
            failures=self._stats.failures,
            successes=self._stats.successes,
            trips=self._stats.trips,
            last_failure_time=self._stats.last_failure_time,
            consecutive_failures=self._stats.consecutive_failures,
        )

    def reset(self) -> None:
        """Manual reset to closed."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats(state=CircuitState.CLOSED)
        self._half_open_calls = 0
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED")

    def __repr__(self) -> str:
        return f"CircuitBreaker({self.name}, state={self._state.value})"
