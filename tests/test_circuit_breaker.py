import pytest
import time
from fleet.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerPanel,
    CircuitState,
)


class TestCircuitBreaker:
    def test_init(self):
        cb = CircuitBreaker("test")
        assert cb.name == "test"
        assert cb.state == CircuitState.CLOSED

    def test_call_success(self):
        cb = CircuitBreaker("test")
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_call_failure(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.failures == 1

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.state == CircuitState.OPEN

    def test_open_raises(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time()
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: 42)

    def test_half_open_then_close(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.001, success_threshold=1
        ))
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time() - 1
        cb.call(lambda: 42)
        assert cb.state == CircuitState.CLOSED

    def test_half_open_too_many_calls(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.001, half_open_max_calls=1
        ))
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time() - 1
        cb.call(lambda: 42)
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: 42)

    def test_get_stats(self):
        cb = CircuitBreaker("test")
        cb.call(lambda: 42)
        stats = cb.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["recent_success_rate"] == 1.0

    def test_to_dict(self):
        cb = CircuitBreaker("test")
        d = cb.to_dict()
        assert d["name"] == "test"


class TestCircuitBreakerPanel:
    def test_get(self):
        panel = CircuitBreakerPanel()
        cb = panel.get("svc")
        assert cb.name == "svc"

    def test_call(self):
        panel = CircuitBreakerPanel()
        result = panel.call("svc", lambda: 42)
        assert result == 42

    def test_get_all_stats(self):
        panel = CircuitBreakerPanel()
        panel.call("svc", lambda: 42)
        stats = panel.get_all_stats()
        assert "svc" in stats

    def test_to_dict(self):
        panel = CircuitBreakerPanel()
        panel.get("svc")
        d = panel.to_dict()
        assert d["breakers"] == 1
