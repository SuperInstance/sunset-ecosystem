import time
import pytest
from fleet.rate_limiter import RateLimiter, RateLimiterPanel


class TestRateLimiter:
    def test_init(self):
        rl = RateLimiter("test", 10.0, 20.0)
        assert rl.name == "test"
        assert rl.tokens == 20.0

    def test_allow(self):
        rl = RateLimiter("test", 10.0, 1.0)
        assert rl.allow() is True
        assert rl.tokens == 0.0
        assert rl.allow() is False

    def test_allow_refill(self):
        rl = RateLimiter("test", 1000.0, 1.0)
        assert rl.allow() is True
        time.sleep(0.01)
        assert rl.allow() is True

    def test_get_status(self):
        rl = RateLimiter("test", 10.0, 5.0)
        status = rl.get_status()
        assert status["name"] == "test"
        assert status["tokens"] == 5.0

    def test_to_dict(self):
        rl = RateLimiter("test", 10.0, 5.0)
        d = rl.to_dict()
        assert d["name"] == "test"


class TestRateLimiterPanel:
    def test_get(self):
        panel = RateLimiterPanel()
        rl = panel.get("svc", rate=5.0, burst=10.0)
        assert rl.name == "svc"
        assert rl.rate == 5.0

    def test_allow(self):
        panel = RateLimiterPanel()
        assert panel.allow("svc", rate=1.0, burst=1.0) is True
        assert panel.allow("svc", rate=1.0, burst=1.0) is False

    def test_get_all_status(self):
        panel = RateLimiterPanel()
        panel.allow("a", rate=1.0, burst=1.0)
        panel.allow("b", rate=1.0, burst=1.0)
        status = panel.get_all_status()
        assert "a" in status
        assert "b" in status

    def test_to_dict(self):
        panel = RateLimiterPanel()
        panel.allow("svc", rate=1.0, burst=1.0)
        d = panel.to_dict()
        assert d["limiters"] == 1
