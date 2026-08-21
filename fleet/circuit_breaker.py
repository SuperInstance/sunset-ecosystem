from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """
    Circuit breaker for fleet resilience.

    Prevents cascade failures by rejecting requests to failing services.
    Auto-recovers when service health improves.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._call_history: List[Dict[str, Any]] = []

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Returns result if successful, raises if circuit is open
        or function fails.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.successes = 0
            else:
                raise CircuitBreakerOpen(
                    f"Circuit '{self.name}' is OPEN. Service unavailable."
                )

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpen(
                    f"Circuit '{self.name}' HALF_OPEN limit reached."
                )
            self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failures = 0
                self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            self.failures = max(0, self.failures - 1)

        self._call_history.append({"time": time.time(), "success": True})

    def _on_failure(self):
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED:
            if self.failures >= self.config.failure_threshold:
                self.state = CircuitState.OPEN

        self._call_history.append({"time": time.time(), "success": False})

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        recent = [c for c in self._call_history if time.time() - c["time"] < 60]
        successes = sum(1 for c in recent if c["success"])
        failures = sum(1 for c in recent if not c["success"])

        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "successes": self.successes,
            "half_open_calls": self.half_open_calls,
            "recent_success_rate": successes / len(recent) if recent else 0.0,
            "recent_calls": len(recent),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": self.get_stats(),
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreakerPanel:
    """
    Panel of circuit breakers for fleet-wide resilience.

    Manages multiple circuit breakers for different services.
    """

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name)
        return self.breakers[name]

    def call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Call through a named circuit breaker."""
        breaker = self.get(name)
        return breaker.call(func, *args, **kwargs)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breakers": len(self.breakers),
            "stats": self.get_all_stats(),
        }
