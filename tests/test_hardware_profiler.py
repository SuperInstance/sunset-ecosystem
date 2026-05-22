#!/usr/bin/env python3
"""Tests for scripts/profile_hardware.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.profile_hardware import HardwareProfiler, HardwareReport


class TestHardwareProfiler:
    def test_detect_devices_returns_at_least_cpu(self):
        profiler = HardwareProfiler()
        assert len(profiler.devices) >= 1
        cpu_devices = [d for d in profiler.devices if d.device_type == "cpu"]
        assert len(cpu_devices) >= 1, "Expected at least one CPU device"

    def test_measure_idle_returns_positive_power(self):
        profiler = HardwareProfiler()
        result = profiler.measure_idle(duration_sec=0.5)
        assert "mean_watts" in result
        assert result["mean_watts"] > 0, f"Expected positive power, got {result['mean_watts']}"
        assert "duration_sec" in result
        assert result["duration_sec"] > 0

    def test_measure_einsum_returns_joules_per_op(self):
        profiler = HardwareProfiler()
        config = {"n_rooms": 100, "n_fibers": 2}
        result = profiler.measure_operation("einsum", config, duration_sec=0.5)
        assert "joules_per_op" in result
        assert result["joules_per_op"] >= 0, f"Expected non-negative joules_per_op, got {result['joules_per_op']}"
        assert "ops_per_second" in result
        assert result["ops_per_second"] > 0

    def test_profile_all_generates_hardware_report_with_all_operations(self):
        profiler = HardwareProfiler()
        config = {"n_rooms": 50, "n_fibers": 2}
        report = profiler.profile_all(config)
        assert isinstance(report, HardwareReport)
        assert report.hostname != ""
        assert len(report.devices) >= 1
        # Should have attempted all 5 operations (some may fail gracefully)
        assert len(report.operations) > 0
        # einsum should definitely succeed since numpy is always available
        einsum_ops = [op for op in report.operations if op.operation == "einsum"]
        assert len(einsum_ops) == 1, "Expected einsum operation to be profiled"
        assert einsum_ops[0].joules_per_op >= 0
        assert einsum_ops[0].ops_per_second > 0

    def test_save_creates_json_and_markdown(self, tmp_path):
        profiler = HardwareProfiler()
        report = profiler.profile_all({"n_rooms": 50, "n_fibers": 2})
        json_path, md_path = profiler.save(report, out_dir=str(tmp_path))
        assert Path(json_path).exists()
        assert Path(md_path).exists()
        assert Path(json_path).stat().st_size > 0
        assert Path(md_path).stat().st_size > 0

    def test_measure_novelty_scoring(self):
        profiler = HardwareProfiler()
        config = {"n_rooms": 100, "n_fibers": 2}
        result = profiler.measure_operation("novelty_scoring", config, duration_sec=0.5)
        assert "joules_per_op" in result
        assert result["ops_per_second"] > 0

    def test_measure_routing(self):
        profiler = HardwareProfiler()
        config = {"n_rooms": 100, "n_fibers": 4}
        result = profiler.measure_operation("routing", config, duration_sec=0.5)
        assert "joules_per_op" in result
        assert result["ops_per_second"] > 0

    def test_measure_thermal_scheduling(self):
        profiler = HardwareProfiler()
        config = {"n_rooms": 100, "n_fibers": 2}
        result = profiler.measure_operation("thermal_scheduling", config, duration_sec=0.5)
        assert "joules_per_op" in result
        assert result["ops_per_second"] > 0
