"""Tests for ThermalAutoCalibrator."""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from ethos.thermal_auto_calibrate import ThermalAutoCalibrator, ThermalBudget


class TestCalibration:
    def test_calibrate_from_profile_learns_model(self):
        cal = ThermalAutoCalibrator()
        profiles = [
            {"n_agents": 10, "power_w": 60.0, "temp_c": 40.0},
            {"n_agents": 20, "power_w": 110.0, "temp_c": 45.0},
            {"n_agents": 30, "power_w": 160.0, "temp_c": 50.0},
        ]
        cal.calibrate_from_profile(profiles)
        assert cal._calibrated
        assert cal._power_per_agent > 0
        assert cal._temp_per_agent > 0

    def test_calibrate_empty_profiles_uses_defaults(self):
        cal = ThermalAutoCalibrator()
        cal.calibrate_from_profile([])
        assert not cal._calibrated

    def test_calibrate_single_profile_warns(self):
        cal = ThermalAutoCalibrator()
        cal.calibrate_from_profile([{"n_agents": 10, "power_w": 50.0}])
        assert not cal._calibrated


class TestPrediction:
    def test_predict_budget_linear(self):
        cal = ThermalAutoCalibrator()
        cal._calibrated = True
        cal._power_per_agent = 5.0
        cal._temp_per_agent = 0.5
        cal._intercept_power = 10.0
        cal._intercept_temp = 35.0
        cal.temp_threshold = 85.0

        budget = cal.predict_budget(n_agents=50)
        assert isinstance(budget, ThermalBudget)
        assert budget.n_agents == 50
        assert budget.predicted_power_w == 260.0  # 10 + 5*50
        assert budget.predicted_temp_c == 60.0  # 35 + 0.5*50
        assert budget.max_safe_agents == 100  # (85-35)/0.5

    def test_predict_budget_uncalibrated_low_confidence(self):
        cal = ThermalAutoCalibrator()
        budget = cal.predict_budget(n_agents=10)
        assert budget.confidence == 0.0


class TestRebalance:
    def test_rebalance_when_under_threshold(self):
        cal = ThermalAutoCalibrator()
        load = {1: 0.5, 2: 0.6, 3: 0.4}
        result = cal.rebalance_on_alert(load, threshold=0.85)
        assert result["status"] == "ok"
        assert len(result["agents_to_migrate"]) == 0

    def test_rebalance_when_over_threshold(self):
        cal = ThermalAutoCalibrator()
        load = {1: 0.9, 2: 0.9, 3: 0.9}
        result = cal.rebalance_on_alert(load, threshold=0.80)
        assert result["status"] == "rebalance_required"
        assert len(result["agents_to_migrate"]) > 0
        assert result["agents_to_migrate"][0] in load  # hottest agent first

    def test_rebalance_empty_load(self):
        cal = ThermalAutoCalibrator()
        result = cal.rebalance_on_alert({}, threshold=0.85)
        assert result["status"] == "ok"


class TestSerialization:
    def test_save_and_load_model(self):
        cal = ThermalAutoCalibrator(temp_threshold=80.0)
        cal._calibrated = True
        cal._power_per_agent = 4.5
        cal._temp_per_agent = 0.4
        cal._intercept_power = 12.0
        cal._intercept_temp = 30.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cal.save_model(f.name)
            path = f.name

        cal2 = ThermalAutoCalibrator()
        cal2.load_model(path)
        assert cal2.temp_threshold == 80.0
        assert cal2._power_per_agent == 4.5
        assert cal2._calibrated
