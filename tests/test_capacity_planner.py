"""Tests for capacity_planner.py — Capacity planning with forecasting.

Run: python3 -m pytest tests/test_capacity_planner.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.capacity_planner import CapacityPlanner


class TestCapacityPlanner:
    def test_create(self):
        planner = CapacityPlanner()
        assert planner.stats()["resources"] == 0

    def test_add_usage(self):
        planner = CapacityPlanner()
        planner.add_usage("cpu", [10, 20, 30])
        assert planner.stats()["samples"]["cpu"] == 3

    def test_current_utilization(self):
        planner = CapacityPlanner()
        planner.add_usage("cpu", [10, 20, 30])
        assert planner.current_utilization()["cpu"] == 30

    def test_avg_utilization(self):
        planner = CapacityPlanner()
        planner.add_usage("cpu", [10, 20, 30])
        assert planner.avg_utilization()["cpu"] == 20.0

    def test_trend_increasing(self):
        planner = CapacityPlanner()
        planner.add_usage("cpu", [10, 20, 30, 40, 50])
        assert planner.trend("cpu") > 0

    def test_trend_decreasing(self):
        planner = CapacityPlanner()
        planner.add_usage("cpu", [50, 40, 30, 20, 10])
        assert planner.trend("cpu") < 0

    def test_forecast(self):
        planner = CapacityPlanner()
        planner.add_usage("cpu", [10, 20, 30, 40, 50])
        forecast = planner.forecast("cpu", steps=3)
        assert len(forecast) == 3
        assert forecast[0] > 50

    def test_recommend_scale_up(self):
        planner = CapacityPlanner(scale_up_threshold=0.7)
        planner.add_usage("cpu", [0.8])
        rec = planner.recommend()
        assert rec["action"] == "scale_up"
        assert "cpu" in rec["reasons"][0]

    def test_recommend_scale_down(self):
        planner = CapacityPlanner(scale_down_threshold=0.3)
        planner.add_usage("cpu", [0.2])
        rec = planner.recommend()
        assert rec["action"] == "scale_down"

    def test_recommend_stable(self):
        planner = CapacityPlanner(scale_up_threshold=0.8, scale_down_threshold=0.2)
        planner.add_usage("cpu", [0.5])
        rec = planner.recommend()
        assert rec["action"] == "stable"

    def test_add_point(self):
        planner = CapacityPlanner()
        planner.add_point("cpu", 10)
        planner.add_point("cpu", 20)
        assert planner.stats()["samples"]["cpu"] == 2

    def test_repr(self):
        planner = CapacityPlanner()
        planner.add_usage("cpu", [10])
        assert "CapacityPlanner" in repr(planner)
