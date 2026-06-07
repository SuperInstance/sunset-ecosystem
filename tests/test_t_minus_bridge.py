"""Tests for TMinusBridge — Python bridge to t-minus-rs Rust crate.

Reference: fleet/t_minus_bridge.py
Requires: bin/t_minus_bridge binary compiled from t-minus-rs examples
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.t_minus_bridge import CronSchedule, DeadlineTree, RateLimiter, TMinusBridge


@pytest.fixture
def bridge():
    """Provide a TMinusBridge instance."""
    try:
        b = TMinusBridge()
        if not b.is_available():
            pytest.skip("t_minus_bridge binary not available")
        return b
    except FileNotFoundError:
        pytest.skip("t_minus_bridge binary not found — run: cargo build --example t_minus_bridge")


class TestCronScheduling:
    def test_cron_next_basic(self, bridge: TMinusBridge) -> None:
        next_fire = bridge.cron_next("*/15 * * * *", after=0)
        assert next_fire == 900  # 15 minutes = 900 seconds

    def test_cron_next_hourly(self, bridge: TMinusBridge) -> None:
        next_fire = bridge.cron_next("0 * * * *", after=0)
        assert next_fire == 3600  # 1 hour = 3600 seconds

    def test_cron_next_daily(self, bridge: TMinusBridge) -> None:
        next_fire = bridge.cron_next("0 0 * * *", after=0)
        assert next_fire == 86400  # 24 hours = 86400 seconds

    def test_cron_schedule(self, bridge: TMinusBridge) -> None:
        sched = bridge.cron_schedule("*/15 * * * *")
        assert isinstance(sched, CronSchedule)
        assert sched.expr == "*/15 * * * *"
        assert sched.next_fire == 900

    def test_cron_next_with_offset(self, bridge: TMinusBridge) -> None:
        # After 800 seconds, next 15-min fire is at 900
        next_fire = bridge.cron_next("*/15 * * * *", after=800)
        assert next_fire == 900

    def test_cron_invalid_expr(self, bridge: TMinusBridge) -> None:
        with pytest.raises(ValueError):
            bridge.cron_next("invalid", after=0)


class TestDeadlineTrees:
    def test_deadline_remaining_parent_smaller(self, bridge: TMinusBridge) -> None:
        remaining = bridge.deadline_remaining(parent_secs=60, child_secs=120)
        assert remaining == pytest.approx(60.0, abs=1e-3)  # Child inherits parent's 60s

    def test_deadline_remaining_child_smaller(self, bridge: TMinusBridge) -> None:
        remaining = bridge.deadline_remaining(parent_secs=120, child_secs=60)
        assert remaining == pytest.approx(60.0, abs=1e-3)  # Child keeps its own 60s

    def test_deadline_remaining_equal(self, bridge: TMinusBridge) -> None:
        remaining = bridge.deadline_remaining(parent_secs=60, child_secs=60)
        assert remaining == pytest.approx(60.0, abs=1e-3)

    def test_build_deadline_tree(self, bridge: TMinusBridge) -> None:
        tree = bridge.build_deadline_tree(parent_secs=60, child_secs=120)
        assert isinstance(tree, DeadlineTree)
        assert tree.parent_secs == 60.0
        assert tree.child_secs == 120.0
        assert tree.remaining_secs == pytest.approx(60.0, abs=1e-3)

    def test_deadline_tree_type(self, bridge: TMinusBridge) -> None:
        tree = bridge.build_deadline_tree(30, 45)
        assert isinstance(tree.remaining_secs, float)
        assert tree.remaining_secs > 0


class TestRateLimiting:
    def test_token_bucket_acquire(self, bridge: TMinusBridge) -> None:
        limiter = bridge.token_bucket(burst=10.0, rate=2.0, acquire=3.0)
        assert isinstance(limiter, RateLimiter)
        assert limiter.acquired is True
        assert limiter.tokens_remaining == pytest.approx(7.0, abs=1e-3)
        assert limiter.burst == 10.0
        assert limiter.rate == 2.0

    def test_token_bucket_over_acquire(self, bridge: TMinusBridge) -> None:
        limiter = bridge.token_bucket(burst=5.0, rate=1.0, acquire=10.0)
        assert limiter.acquired is False
        assert limiter.tokens_remaining == 5.0

    def test_check_rate_limit(self, bridge: TMinusBridge) -> None:
        assert bridge.check_rate_limit(burst=10.0, rate=2.0, acquire=1.0) is True
        assert bridge.check_rate_limit(burst=1.0, rate=1.0, acquire=5.0) is False

    def test_rate_limiter_fields(self, bridge: TMinusBridge) -> None:
        limiter = bridge.token_bucket(burst=20.0, rate=5.0, acquire=2.0)
        assert limiter.burst == 20.0
        assert limiter.rate == 5.0
        assert limiter.acquired is True
        assert limiter.tokens_remaining == pytest.approx(18.0, abs=1e-3)


class TestIntegrationHelpers:
    def test_schedule_fleet_beat(self, bridge: TMinusBridge) -> None:
        next_beat = bridge.schedule_fleet_beat(interval_mins=15)
        assert next_beat == 900

    def test_schedule_fleet_beat_custom(self, bridge: TMinusBridge) -> None:
        next_beat = bridge.schedule_fleet_beat(interval_mins=5)
        assert next_beat == 300

    def test_propagate_deadline(self, bridge: TMinusBridge) -> None:
        effective = bridge.propagate_deadline(parent_deadline=60.0, child_budget=120.0)
        assert effective == pytest.approx(60.0, abs=1e-3)

    def test_throttle_fleet_operation(self, bridge: TMinusBridge) -> None:
        assert bridge.throttle_fleet_operation(ops_per_sec=2.0, burst=10) is True
        # First call should succeed, but we're not tracking state across calls
        # since each call creates a fresh bucket

    def test_propagate_deadline_child_smaller(self, bridge: TMinusBridge) -> None:
        effective = bridge.propagate_deadline(120.0, 60.0)
        assert effective == pytest.approx(60.0, abs=1e-3)


class TestBridgeAvailability:
    def test_is_available(self, bridge: TMinusBridge) -> None:
        assert bridge.is_available() is True

    def test_repr(self, bridge: TMinusBridge) -> None:
        assert "TMinusBridge" in repr(bridge)
        assert "t_minus_bridge" in repr(bridge)


class TestBinaryResolution:
    def test_resolve_explicit_path(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake_t_minus"
        fake.write_text("#!/bin/sh\necho '{}'".format('{"success":true}'))
        fake.chmod(0o755)
        bridge = TMinusBridge(binary_path=fake)
        assert bridge.binary_path == fake

    def test_resolve_missing_explicit(self) -> None:
        with pytest.raises(FileNotFoundError):
            TMinusBridge(binary_path="/nonexistent/path")

    def test_call_json(self, bridge: TMinusBridge) -> None:
        # Test internal _call with valid request
        resp = bridge._call({"op": "CronNext", "expr": "0 * * * *", "after": 0})
        assert resp["success"] is True
        assert "result" in resp

    def test_call_invalid_op(self, bridge: TMinusBridge) -> None:
        # Invalid op returns JSON but success may be false or the binary may panic
        # We just verify it doesn't crash and returns a response
        resp = bridge._call({"op": "InvalidOp"})
        # The binary may return success:false or just not match any arm
        assert isinstance(resp, dict)


class TestEdgeCases:
    def test_zero_parent_deadline(self, bridge: TMinusBridge) -> None:
        remaining = bridge.deadline_remaining(parent_secs=0, child_secs=120)
        assert remaining == 0.0

    def test_zero_child_budget(self, bridge: TMinusBridge) -> None:
        remaining = bridge.deadline_remaining(parent_secs=60, child_secs=0)
        assert remaining == 0.0

    def test_token_bucket_zero_acquire(self, bridge: TMinusBridge) -> None:
        limiter = bridge.token_bucket(burst=10.0, rate=2.0, acquire=0.0)
        assert limiter.acquired is True
        assert limiter.tokens_remaining == 10.0

    def test_cron_next_zero_interval(self, bridge: TMinusBridge) -> None:
        # Every minute
        next_fire = bridge.cron_next("* * * * *", after=0)
        assert next_fire == 60
