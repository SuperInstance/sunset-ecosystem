#!/usr/bin/env python3
"""tests/test_health_thermal_bridge.py — HealthThermalBridge tests."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sunset.health_thermal_bridge import ThermalReading, HealthThermalBridge


class TestThermalReading:
    def test_pressure_score_cool(self):
        r = ThermalReading(
            "test", cpu_percent=20, gpu_percent=10, memory_percent=30, temperature_c=40
        )
        assert r.pressure_score() < 0.3

    def test_pressure_score_warm(self):
        r = ThermalReading(
            "test", cpu_percent=60, gpu_percent=70, memory_percent=50, temperature_c=75
        )
        assert 0.5 <= r.pressure_score() <= 0.8

    def test_pressure_score_critical(self):
        r = ThermalReading(
            "test", cpu_percent=95, gpu_percent=98, memory_percent=90, temperature_c=95
        )
        assert r.pressure_score() >= 0.9


class TestHealthThermalBridge:
    def test_subscribe_without_bus(self):
        bridge = HealthThermalBridge()
        assert bridge.subscribe() is False

    def test_pressure_history_empty(self):
        bridge = HealthThermalBridge()
        mn, mu, mx = bridge.pressure_history()
        assert mn == mu == mx == 0.0

    def test_status_empty(self):
        bridge = HealthThermalBridge()
        s = bridge.status()
        assert s["subscribed"] is False
        assert s["readings_count"] == 0

    def test_on_thermal_snapshot_updates_pressure(self):
        bridge = HealthThermalBridge()
        event = {
            "source": "cocapn-health",
            "cpu_percent": 80.0,
            "gpu_percent": 90.0,
            "memory_percent": 70.0,
            "temperature_c": 85.0,
        }
        bridge._on_thermal_snapshot(event)
        assert bridge._last_pressure > 0.7
        assert len(bridge._readings) == 1

    def test_on_thermal_snapshot_rolls_window(self):
        bridge = HealthThermalBridge()
        for i in range(105):
            bridge._on_thermal_snapshot(
                {
                    "source": "test",
                    "cpu_percent": 50,
                    "gpu_percent": 50,
                    "memory_percent": 50,
                    "temperature_c": 60,
                }
            )
        assert len(bridge._readings) <= 100

    def test_pressure_history(self):
        bridge = HealthThermalBridge()
        for i in range(5):
            bridge._on_thermal_snapshot(
                {
                    "source": "test",
                    "cpu_percent": i * 20,
                    "gpu_percent": i * 20,
                    "memory_percent": i * 10,
                    "temperature_c": 40 + i * 5,
                }
            )
        mn, mu, mx = bridge.pressure_history(window=5)
        assert mn < mu < mx
        assert 0 <= mn <= mx <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
