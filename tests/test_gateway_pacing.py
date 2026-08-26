"""Tests for GatewayPacing circuit breaker.

Covers:
  • OPEN state allows dispatch
  • 2 consecutive timeouts → CLOSED
  • CLOSED rejects dispatch with reason
  • Backoff durations correct (linear then exponential)
  • HALF_OPEN allows 1 probe per 30s
  • 3 probes succeed → reopens to OPEN
  • Status dict is accurate
  • Thread-safe (multiple concurrent calls)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Generator

import pytest

from fleet.gateway_pacing import GatewayPacing, State


# ── Time-mocking fixture ──────────────────────────────────────────


class _MockClock:
    """Deterministic monotonic clock for tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


@pytest.fixture
def mock_time(monkeypatch: Any) -> Generator[_MockClock, None, None]:
    """Patch time.monotonic in fleet.gateway_pacing with a deterministic clock."""
    clock = _MockClock(start=1000.0)
    monkeypatch.setattr("fleet.gateway_pacing.time.monotonic", clock.now)
    yield clock


# ── 1. OPEN state allows dispatch ─────────────────────────────────


def test_open_allows_dispatch(mock_time: _MockClock) -> None:
    gp = GatewayPacing()
    ok, reason = gp.can_dispatch()
    assert ok is True
    assert "OPEN" in reason


def test_open_multiple_dispatches_ok(mock_time: _MockClock) -> None:
    gp = GatewayPacing()
    for _ in range(5):
        ok, reason = gp.can_dispatch()
        assert ok is True
        gp.record_success()
    assert gp.get_status()["success_count"] == 5


# ── 2. 2 consecutive timeouts → CLOSED ────────────────────────────


def test_two_timeouts_closes_circuit(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_timeout()
    assert gp.get_status()["state"] == "OPEN"
    gp.record_timeout()
    assert gp.get_status()["state"] == "CLOSED"


def test_single_timeout_stays_open(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_timeout()
    ok, _ = gp.can_dispatch()
    assert ok is True


# ── 3. CLOSED rejects dispatch with reason ─────────────────────────


def test_closed_rejects_dispatch(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_timeout()
    gp.record_timeout()
    ok, reason = gp.can_dispatch()
    assert ok is False
    assert "CLOSED" in reason
    assert "backoff" in reason.lower()


# ── 4. Backoff durations correct (linear then exponential) ───────


def test_backoff_linear_phase(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_timeout()
    gp.record_timeout()  # → CLOSED with 2 timeouts
    status = gp.get_status()
    # 2 * 30s = 60s
    assert status["backoff_remaining"] == pytest.approx(60.0, abs=0.1)


def test_backoff_linear_cap_at_5min(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=10)
    for _ in range(10):
        gp.record_timeout()
    # 10 timeouts * 30s = 300s, capped at 300s (5min)
    status = gp.get_status()
    assert status["backoff_remaining"] == pytest.approx(300.0, abs=0.1)


def test_backoff_exponential_beyond_linear_cap(mock_time: _MockClock) -> None:
    gp = GatewayPacing(
        max_consecutive_timeouts=12,
        linear_backoff_max=300.0,
        exponential_backoff_max=1200.0,
    )
    for _ in range(12):
        gp.record_timeout()
    # 12 timeouts: linear capped at 300s (10 * 30).
    # extra_timeouts = 12 - 10 = 2
    # exponential = 2^2 * 30 = 120
    # total = 300 + 120 = 420
    status = gp.get_status()
    assert status["backoff_remaining"] == pytest.approx(420.0, abs=0.1)


def test_backoff_exponential_hard_cap_at_20min(mock_time: _MockClock) -> None:
    gp = GatewayPacing(
        max_consecutive_timeouts=50,
        linear_backoff_max=300.0,
        exponential_backoff_max=1200.0,
    )
    for _ in range(50):
        gp.record_timeout()
    status = gp.get_status()
    assert status["backoff_remaining"] == pytest.approx(1200.0, abs=0.1)


# ── 5. HALF_OPEN allows 1 probe per 30s ──────────────────────────


def test_half_open_after_backoff_elapsed(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_timeout()
    gp.record_timeout()

    mock_time.advance(61.0)
    ok, reason = gp.can_dispatch()
    assert ok is True
    assert "HALF_OPEN" in reason


def test_half_open_throttles_second_probe(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_timeout()
    gp.record_timeout()

    mock_time.advance(61.0)
    ok1, _ = gp.can_dispatch()
    assert ok1 is True  # first probe

    ok2, reason2 = gp.can_dispatch()
    assert ok2 is False
    assert "throttled" in reason2.lower()


def test_half_open_allows_probe_after_interval(mock_time: _MockClock) -> None:
    gp = GatewayPacing(
        max_consecutive_timeouts=2,
        half_open_probe_interval=30.0,
    )
    gp.record_timeout()
    gp.record_timeout()

    mock_time.advance(61.0)
    ok1, _ = gp.can_dispatch()
    assert ok1 is True

    # Not enough time elapsed
    mock_time.advance(15.0)
    ok2, _ = gp.can_dispatch()
    assert ok2 is False

    # Now enough
    mock_time.advance(20.0)  # total 96s since CLOSED
    ok3, reason3 = gp.can_dispatch()
    assert ok3 is True
    assert "HALF_OPEN" in reason3


# ── 6. 3 probes succeed → reopens to OPEN ─────────────────────────


def test_three_probe_successes_reopen(mock_time: _MockClock) -> None:
    gp = GatewayPacing(
        max_consecutive_timeouts=2,
        successes_to_reopen=3,
    )
    gp.record_timeout()
    gp.record_timeout()

    mock_time.advance(61.0)
    for i in range(3):
        ok, _ = gp.can_dispatch()
        assert ok is True, f"probe {i + 1} should be allowed"
        gp.record_success()
        if i < 2:
            mock_time.advance(31.0)  # advance past probe interval

    assert gp.get_status()["state"] == "OPEN"


def test_two_probe_successes_stay_half_open(mock_time: _MockClock) -> None:
    gp = GatewayPacing(
        max_consecutive_timeouts=2,
        successes_to_reopen=3,
    )
    gp.record_timeout()
    gp.record_timeout()

    mock_time.advance(61.0)
    for i in range(2):
        ok, _ = gp.can_dispatch()
        assert ok is True
        gp.record_success()
        if i < 1:
            mock_time.advance(31.0)  # advance past probe interval

    assert gp.get_status()["state"] == "HALF_OPEN"


def test_probe_failure_in_half_open_resets_successes(mock_time: _MockClock) -> None:
    gp = GatewayPacing(
        max_consecutive_timeouts=2,
        successes_to_reopen=3,
    )
    gp.record_timeout()
    gp.record_timeout()

    mock_time.advance(61.0)
    ok, _ = gp.can_dispatch()
    assert ok is True
    gp.record_success()

    mock_time.advance(31.0)
    ok, _ = gp.can_dispatch()
    assert ok is True
    gp.record_failure()

    # Still HALF_OPEN, successes not yet enough
    status = gp.get_status()
    assert status["state"] == "HALF_OPEN"
    assert status["half_open_successes"] == 1


# ── 7. Status dict is accurate ────────────────────────────────────


def test_status_dict_fields(mock_time: _MockClock) -> None:
    gp = GatewayPacing()
    status = gp.get_status()
    assert status["state"] == "OPEN"
    assert status["success_count"] == 0
    assert status["failure_count"] == 0
    assert status["consecutive_timeouts"] == 0
    assert status["backoff_remaining"] == 0.0
    assert status["history"] == []


def test_status_after_timeout(mock_time: _MockClock) -> None:
    gp = GatewayPacing()
    gp.record_timeout()
    status = gp.get_status()
    assert status["consecutive_timeouts"] == 1
    assert len(status["history"]) == 1
    assert status["history"][0]["event"] == "timeout"


def test_status_history_limit(mock_time: _MockClock) -> None:
    gp = GatewayPacing(history_limit=5)
    for _ in range(10):
        gp.record_success()
    status = gp.get_status()
    assert len(status["history"]) == 5


# ── 8. Thread-safe (multiple concurrent calls) ────────────────────


def test_thread_safe_concurrent_can_dispatch() -> None:
    gp = GatewayPacing()
    results: list[tuple[bool, str]] = []
    errors: list[Exception] = []

    def caller() -> None:
        try:
            results.append(gp.can_dispatch())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=caller) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 20
    assert all(ok for ok, _ in results)


def test_thread_safe_concurrent_record_timeout() -> None:
    gp = GatewayPacing(max_consecutive_timeouts=20)
    errors: list[Exception] = []

    def recorder() -> None:
        try:
            gp.record_timeout()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=recorder) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    status = gp.get_status()
    assert status["consecutive_timeouts"] == 20


def test_thread_safe_mixed_operations() -> None:
    gp = GatewayPacing()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(50):
                gp.can_dispatch()
                gp.record_success()
                gp.record_failure()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    status = gp.get_status()
    assert status["success_count"] == 200
    assert status["failure_count"] == 200


# ── 9. record_failure does not trip breaker ───────────────────────


def test_failure_does_not_trip_breaker(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_failure()
    gp.record_failure()
    gp.record_failure()
    assert gp.get_status()["state"] == "OPEN"
    assert gp.get_status()["consecutive_timeouts"] == 0


# ── 10. Edge: success resets consecutive_timeouts ─────────────────


def test_success_resets_timeout_counter(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2)
    gp.record_timeout()
    gp.record_success()
    gp.record_timeout()
    # Only 1 consecutive timeout now, should still be OPEN
    assert gp.get_status()["state"] == "OPEN"
    assert gp.get_status()["consecutive_timeouts"] == 1


# ── 11. Default parameters ────────────────────────────────────────


def test_default_parameters(mock_time: _MockClock) -> None:
    gp = GatewayPacing()
    assert gp._max_consecutive_timeouts == 2
    assert gp._linear_backoff_max == 300.0
    assert gp._exponential_backoff_max == 1200.0
    assert gp._half_open_probe_interval == 30.0
    assert gp._successes_to_reopen == 3
    assert gp._history_limit == 50


# ── 12. History entries are structured ─────────────────────────────


def test_history_entry_structure(mock_time: _MockClock) -> None:
    gp = GatewayPacing(max_consecutive_timeouts=2, exponential_backoff_max=9999.0)
    gp.record_timeout()
    gp.record_timeout()
    # get_status will transition to HALF_OPEN if we call it,
    # so avoid calling it until we are ready.
    # Instead inspect directly through the private attr.
    raw_history = gp._history
    assert len(raw_history) == 3  # timeout, timeout, state_change
    assert raw_history[0].event == "timeout"
    assert raw_history[1].event == "timeout"
    assert raw_history[2].event == "state_change"
    assert "CLOSED" in raw_history[2].detail
