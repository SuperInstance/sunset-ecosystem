"""Gateway Pacing — circuit breaker for subagent dispatch.

Prevents cascade failures by tracking consecutive timeouts and enforcing
backoff before the gateway chokes. Integrated into the main session
dispatcher before spawning subagents.

Usage::

    from fleet.gateway_pacing import GatewayPacing

    gp = GatewayPacing()
    ok, reason = gp.can_dispatch()
    if not ok:
        log.warning("Dispatch blocked: %s", reason)
        return

    try:
        subagent.run(...)
        gp.record_success()
    except TimeoutError:
        gp.record_timeout()

Design: OPEN → HALF_OPEN → CLOSED → OPEN
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class State(Enum):
    OPEN = auto()       # Normal operation — dispatches allowed
    HALF_OPEN = auto()  # Probing — 1 dispatch every 30s
    CLOSED = auto()     # Circuit broken — dispatches blocked


@dataclass
class _HistoryEntry:
    timestamp: float
    event: str  # "success", "failure", "timeout", "state_change", "probe"
    detail: str = ""


class GatewayPacing:
    """Circuit breaker for subagent dispatch with state-machine backoff.

    Parameters
    ----------
    max_consecutive_timeouts : int
        Consecutive timeouts before circuit trips to CLOSED (default 2).
    linear_backoff_max : float
        Upper bound of the linear backoff phase in seconds (default 300 = 5min).
    exponential_backoff_max : float
        Hard cap on total backoff in seconds (default 1200 = 20min).
    half_open_probe_interval : float
        Seconds between probe attempts in HALF_OPEN (default 30).
    successes_to_reopen : int
        Successful probes required to transition OPEN (default 3).
    history_limit : int
        Max number of history entries retained (default 50).
    """

    def __init__(
        self,
        max_consecutive_timeouts: int = 2,
        linear_backoff_max: float = 300.0,        # 5min
        exponential_backoff_max: float = 1200.0,  # 20min
        half_open_probe_interval: float = 30.0,
        successes_to_reopen: int = 3,
        history_limit: int = 50,
    ) -> None:
        self._max_consecutive_timeouts = max_consecutive_timeouts
        self._linear_backoff_max = linear_backoff_max
        self._exponential_backoff_max = exponential_backoff_max
        self._half_open_probe_interval = half_open_probe_interval
        self._successes_to_reopen = successes_to_reopen
        self._history_limit = history_limit

        self._lock = threading.Lock()
        self._state = State.OPEN
        self._success_count = 0
        self._failure_count = 0
        self._consecutive_timeouts = 0
        self._last_dispatch_time: float | None = None
        self._circuit_closed_at: float | None = None
        self._half_open_probes_sent = 0
        self._half_open_successes = 0
        self._last_probe_time: float | None = None
        self._history: list[_HistoryEntry] = []

    # ── Public API ────────────────────────────────────────────────────

    def can_dispatch(self) -> tuple[bool, str]:
        """Return (allowed, reason). Thread-safe."""
        with self._lock:
            now = time.monotonic()
            self._maybe_transition(now)

            if self._state == State.OPEN:
                self._last_dispatch_time = now
                self._log(now, "probe", "dispatch allowed (OPEN)")
                return True, "OPEN — dispatch allowed"

            if self._state == State.HALF_OPEN:
                # Allow one probe every interval
                if self._last_probe_time is not None:
                    elapsed = now - self._last_probe_time
                    if elapsed < self._half_open_probe_interval:
                        wait = self._half_open_probe_interval - elapsed
                        return False, (
                            f"HALF_OPEN — probe throttled; "
                            f"wait {wait:.1f}s"
                        )
                self._last_probe_time = now
                self._half_open_probes_sent += 1
                self._log(
                    now,
                    "probe",
                    f"probe {self._half_open_probes_sent} allowed (HALF_OPEN)",
                )
                return True, "HALF_OPEN — probe allowed"

            # CLOSED
            backoff_remaining = self._backoff_remaining(now)
            return False, (
                f"CLOSED — circuit open; "
                f"backoff {backoff_remaining:.1f}s remaining"
            )

    def record_success(self) -> None:
        """Register a successful dispatch outcome. Thread-safe."""
        with self._lock:
            now = time.monotonic()
            self._success_count += 1
            self._consecutive_timeouts = 0
            self._log(now, "success", f"total_success={self._success_count}")

            if self._state == State.HALF_OPEN:
                self._half_open_successes += 1
                self._log(
                    now,
                    "success",
                    f"probe_success={self._half_open_successes}",
                )
                if self._half_open_successes >= self._successes_to_reopen:
                    self._transition(State.OPEN, now)

    def record_failure(self) -> None:
        """Register a non-timeout failure. Thread-safe."""
        with self._lock:
            now = time.monotonic()
            self._failure_count += 1
            self._consecutive_timeouts = 0
            self._log(now, "failure", f"total_failure={self._failure_count}")

    def record_timeout(self) -> None:
        """Register a timeout (the signal that trips the breaker). Thread-safe."""
        with self._lock:
            now = time.monotonic()
            self._consecutive_timeouts += 1
            self._log(
                now,
                "timeout",
                f"consecutive_timeouts={self._consecutive_timeouts}",
            )

            if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                self._transition(State.CLOSED, now)

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of internal state. Thread-safe."""
        with self._lock:
            now = time.monotonic()
            self._maybe_transition(now)
            return {
                "state": self._state.name,
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "consecutive_timeouts": self._consecutive_timeouts,
                "backoff_remaining": self._backoff_remaining(now),
                "last_dispatch_time": self._last_dispatch_time,
                "circuit_closed_at": self._circuit_closed_at,
                "half_open_probes_sent": self._half_open_probes_sent,
                "half_open_successes": self._half_open_successes,
                "successes_to_reopen": self._successes_to_reopen,
                "history": [
                    {
                        "timestamp": e.timestamp,
                        "event": e.event,
                        "detail": e.detail,
                    }
                    for e in self._history
                ],
            }

    # ── Internal mechanics ────────────────────────────────────────────

    def _transition(self, new_state: State, now: float) -> None:
        """Move to *new_state* and log the transition."""
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        detail = f"{old.name} → {new_state.name}"
        if new_state == State.CLOSED:
            self._circuit_closed_at = now
            self._half_open_probes_sent = 0
            self._half_open_successes = 0
        if new_state == State.OPEN:
            self._half_open_probes_sent = 0
            self._half_open_successes = 0
            self._consecutive_timeouts = 0
        self._log(now, "state_change", detail)

    def _maybe_transition(self, now: float) -> None:
        """Check if backoff has elapsed and we should enter HALF_OPEN."""
        if self._state != State.CLOSED:
            return
        elapsed = now - (self._circuit_closed_at or now)
        if elapsed >= self._backoff_duration():
            self._transition(State.HALF_OPEN, now)

    def _backoff_duration(self) -> float:
        """Compute the total backoff for the current CLOSED state.

        Linear up to linear_backoff_max, then exponential up to the hard cap.
        """
        ct = self._consecutive_timeouts
        # Start with 30s per consecutive timeout, capped at linear max
        linear = min(ct * 30.0, self._linear_backoff_max)
        # Exponential kicks in after the linear cap is saturated
        if ct * 30.0 > self._linear_backoff_max:
            # Each additional timeout doubles beyond the linear cap
            extra_timeouts = ct - int(self._linear_backoff_max / 30.0)
            exponential = min(
                (2 ** extra_timeouts) * 30.0,
                self._exponential_backoff_max - self._linear_backoff_max,
            )
            return self._linear_backoff_max + exponential
        return linear

    def _backoff_remaining(self, now: float) -> float:
        """Seconds left in the current backoff, or 0.0 if not CLOSED."""
        if self._state != State.CLOSED or self._circuit_closed_at is None:
            return 0.0
        elapsed = now - self._circuit_closed_at
        total = self._backoff_duration()
        return max(0.0, total - elapsed)

    def _log(self, timestamp: float, event: str, detail: str = "") -> None:
        self._history.append(_HistoryEntry(timestamp, event, detail))
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
