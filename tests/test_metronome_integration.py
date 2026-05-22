"""Tests for MetronomeIntegration."""
from __future__ import annotations

import time

import pytest

from nerve.metronome_integration import DeviceStatus, MetronomeIntegration


class MockGrid:
    """Mock RoomGrid for testing."""
    def __init__(self) -> None:
        self.tick_count = 0

    def tick(self) -> None:
        self.tick_count += 1


class TestRegistration:
    def test_register_device(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.register_device("cuda:0")
        assert "cuda:0" in metro._devices

    def test_unregister_device(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.register_device("cuda:0")
        metro.unregister_device("cuda:0")
        assert "cuda:0" not in metro._devices


class TestHeartbeat:
    def test_heartbeat_updates_timestamp(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.register_device("cuda:0")
        before = metro._devices["cuda:0"].last_heartbeat
        time.sleep(0.01)
        metro.heartbeat("cuda:0")
        assert metro._devices["cuda:0"].last_heartbeat > before

    def test_heartbeat_records_drift(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.register_device("cuda:0")
        metro.heartbeat("cuda:0", drift_ms=5.0)
        assert metro._devices["cuda:0"].drift_ms == 5.0

    def test_heartbeat_unknown_device_warns(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.heartbeat("cuda:99")  # should not raise


class TestDeviceCheck:
    def test_device_goes_offline(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, heartbeat_timeout_sec=0.01)
        metro.register_device("cuda:0", heartbeat_interval_sec=0.001)
        time.sleep(0.02)
        offline = metro.check_devices()
        assert "cuda:0" in offline
        assert not metro._devices["cuda:0"].online

    def test_device_stays_online_with_heartbeat(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, heartbeat_timeout_sec=1.0)
        metro.register_device("cuda:0")
        metro.heartbeat("cuda:0")
        offline = metro.check_devices()
        assert "cuda:0" not in offline
        assert metro._devices["cuda:0"].online


class TestTick:
    def test_disabled_runs_local_only(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        result = metro.tick()
        assert "local" in result
        assert grid.tick_count == 1

    def test_enabled_with_online_devices(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, devices=["cuda:0"])
        metro.enable()
        metro.heartbeat("cuda:0")
        result = metro.tick()
        assert result["results"]["cuda:0"]["status"] == "ticked"
        assert grid.tick_count == 1

    def test_enabled_skips_offline_devices(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, heartbeat_timeout_sec=0.01)
        metro.register_device("cuda:0", heartbeat_interval_sec=0.001)
        metro.enable()
        time.sleep(0.02)
        result = metro.tick()
        assert result["results"]["cuda:0"]["skipped"]
        assert grid.tick_count == 1

    def test_tick_returns_offline_list(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, heartbeat_timeout_sec=0.01)
        metro.register_device("cuda:0", heartbeat_interval_sec=0.001)
        metro.enable()
        time.sleep(0.02)
        result = metro.tick()
        assert "cuda:0" in result["offline_devices"]


class TestDriftCorrection:
    def test_no_devices_zero_drift(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        assert metro.get_drift_correction() == 0.0

    def test_median_drift_calculation(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.register_device("cuda:0")
        metro.register_device("cuda:1")
        metro.register_device("cuda:2")
        metro.heartbeat("cuda:0", drift_ms=1.0)
        metro.heartbeat("cuda:1", drift_ms=5.0)
        metro.heartbeat("cuda:2", drift_ms=10.0)
        assert metro.get_drift_correction() == 5.0  # median

    def test_offline_devices_ignored_for_drift(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, heartbeat_timeout_sec=0.05)
        metro.register_device("cuda:0")
        metro.register_device("cuda:1")
        metro.heartbeat("cuda:0", drift_ms=100.0)
        metro.heartbeat("cuda:1", drift_ms=1.0)
        time.sleep(0.06)
        # cuda:0 and cuda:1 both offline after timeout
        assert metro.get_drift_correction() == 0.0  # no online devices

    def test_median_drift_with_some_offline(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, heartbeat_timeout_sec=0.2)
        metro.register_device("cuda:0")
        metro.register_device("cuda:1")
        metro.register_device("cuda:2")
        metro.heartbeat("cuda:0", drift_ms=1.0)
        metro.heartbeat("cuda:1", drift_ms=5.0)
        metro.heartbeat("cuda:2", drift_ms=100.0)
        # All start online
        assert metro.get_drift_correction() == 5.0  # median of [1, 5, 100]
        # Now cuda:2 goes offline after not heartbeating
        time.sleep(0.15)
        # Re-heartbeat cuda:0 and cuda:1 to keep them online
        metro.heartbeat("cuda:0", drift_ms=1.0)
        metro.heartbeat("cuda:1", drift_ms=5.0)
        # cuda:2 offline, cuda:0 and cuda:1 online
        assert metro.get_drift_correction() == 5.0  # median of [1, 5]


class TestStatus:
    def test_status_counts(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, heartbeat_timeout_sec=0.05)
        metro.register_device("cuda:0")
        metro.register_device("cuda:1")
        metro.heartbeat("cuda:0")
        metro.heartbeat("cuda:1")
        time.sleep(0.06)
        metro.heartbeat("cuda:0")  # only cuda:0 back online
        status = metro.get_status()
        assert status["n_devices"] == 2
        assert status["n_online"] == 1
        assert status["n_offline"] == 1

    def test_status_disabled(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        assert not metro.get_status()["enabled"]

    def test_status_enabled(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.enable()
        assert metro.get_status()["enabled"]


class TestEnableDisable:
    def test_enable_sets_flag(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.enable()
        assert metro._enabled

    def test_disable_clears_flag(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid)
        metro.enable()
        metro.disable()
        assert not metro._enabled


class TestInit:
    def test_init_with_devices(self):
        grid = MockGrid()
        metro = MetronomeIntegration(grid, devices=["cuda:0", "cuda:1"])
        assert len(metro._devices) == 2
