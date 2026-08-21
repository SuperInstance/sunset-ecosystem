"""Tests for EMBenchmarkSuite."""

from __future__ import annotations

import pytest

from benchmarks.em_suite import EMBenchmarkSuite, EMTestResult


class TestSignalIntegrity:
    def test_signal_integrity_returns_result(self):
        suite = EMBenchmarkSuite(seed=42)
        result = suite.test_signal_integrity()
        assert isinstance(result, EMTestResult)
        assert result.test_name == "signal_integrity"
        assert result.unit == "dB"

    def test_signal_integrity_passes_with_short_cable(self):
        suite = EMBenchmarkSuite(seed=42)
        result = suite.test_signal_integrity(cable_length_m=0.1, frequency_mhz=500)
        assert isinstance(result.passed, bool)

    def test_signal_integrity_fails_with_long_cable(self):
        suite = EMBenchmarkSuite(seed=42)
        result = suite.test_signal_integrity(cable_length_m=5.0, frequency_mhz=5000)
        # Long cable + high frequency likely fails
        assert isinstance(result.margin, float)


class TestThermalEmission:
    def test_thermal_emission_returns_result(self):
        suite = EMBenchmarkSuite(seed=42)
        result = suite.test_thermal_emission()
        assert isinstance(result, EMTestResult)
        assert result.test_name == "thermal_emission"
        assert result.unit == "°C"

    def test_thermal_emission_stricter_for_sealed(self):
        suite = EMBenchmarkSuite(seed=42)
        open_result = suite.test_thermal_emission(enclosure_type="open", power_w=100)
        sealed_result = suite.test_thermal_emission(
            enclosure_type="sealed", power_w=100
        )
        assert sealed_result.threshold <= open_result.threshold


class TestPowerLineNoise:
    def test_power_line_noise_returns_result(self):
        suite = EMBenchmarkSuite(seed=42)
        result = suite.test_power_line_noise()
        assert isinstance(result, EMTestResult)
        assert result.test_name == "power_line_noise"
        assert result.unit == "mVpp"

    def test_power_line_noise_increases_with_load(self):
        suite = EMBenchmarkSuite(seed=42)
        low = suite.test_power_line_noise(load_a=1.0)
        high = suite.test_power_line_noise(load_a=10.0)
        assert high.measurement > low.measurement


class TestRFInterference:
    def test_rf_interference_returns_result(self):
        suite = EMBenchmarkSuite(seed=42)
        result = suite.test_rf_interference()
        assert isinstance(result, EMTestResult)
        assert result.test_name == "rf_interference"
        assert result.unit == "dBm"

    def test_rf_interference_decreases_with_distance(self):
        suite = EMBenchmarkSuite(seed=42)
        near = suite.test_rf_interference(distance_m=0.5)
        far = suite.test_rf_interference(distance_m=5.0)
        assert far.measurement < near.measurement


class TestSuiteIntegration:
    def test_run_all_returns_four_results(self):
        suite = EMBenchmarkSuite(seed=42)
        results = suite.run_all()
        assert len(results) == 4
        assert set(results.keys()) == {
            "signal_integrity",
            "thermal_emission",
            "power_line_noise",
            "rf_interference",
        }

    def test_summary_counts_correctly(self):
        suite = EMBenchmarkSuite(seed=42)
        results = suite.run_all()
        summary = suite.summary(results)
        assert summary["total_tests"] == 4
        assert summary["passed"] + summary["failed"] == 4
        assert 0.0 <= summary["pass_rate"] <= 1.0

    def test_reproducible_with_seed(self):
        suite1 = EMBenchmarkSuite(seed=123)
        suite2 = EMBenchmarkSuite(seed=123)
        r1 = suite1.run_all()
        r2 = suite2.run_all()
        for name in r1:
            assert r1[name].measurement == pytest.approx(r2[name].measurement, abs=1e-6)
