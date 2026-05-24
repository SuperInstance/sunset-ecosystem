"""EM Benchmark Suite — electromagnetic compatibility tests for fleet hardware.

Provides `EMBenchmarkSuite` which runs signal integrity, thermal emission,
power line noise, and RF interference tests on hardware components.

Usage::

    from benchmarks.em_suite import EMBenchmarkSuite
    suite = EMBenchmarkSuite()
    results = suite.run_all()
    # results is a dict of test_name → pass/fail with measurements
"""
from __future__ import annotations

__all__ = ["EMBenchmarkSuite", "EMTestResult"]

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EMTestResult:
    """Result of a single EM compatibility test."""
    test_name: str
    passed: bool
    measurement: float
    unit: str
    threshold: float
    margin: float  # how far from threshold (positive = good)


class EMBenchmarkSuite:
    """Runs electromagnetic compatibility tests for fleet hardware.

    Each test generates synthetic but realistic measurements based on
    component type and operating conditions. Real hardware would
    replace these with actual sensor readings.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # ── Public API ─────────────────────────────────────────────────

    def test_signal_integrity(
        self,
        frequency_mhz: float = 1000.0,
        cable_length_m: float = 0.5,
        impedance_ohm: float = 50.0,
    ) -> EMTestResult:
        """Test signal integrity at a given frequency.

        Measures insertion loss (dB) and reflection (S11).
        Pass if insertion loss < 3dB and reflection < -10dB.
        """
        # Synthetic model: higher frequency = more loss, longer cable = more loss
        insertion_loss_db = (
            0.5 * (frequency_mhz / 1000.0)
            + 0.3 * cable_length_m
            + self._np_rng.normal(0, 0.1)
        )
        reflection_db = -15.0 + self._np_rng.normal(0, 2.0)

        passed = insertion_loss_db < 3.0 and reflection_db < -10.0
        margin = min(3.0 - insertion_loss_db, -10.0 - reflection_db)

        return EMTestResult(
            test_name="signal_integrity",
            passed=passed,
            measurement=insertion_loss_db,
            unit="dB",
            threshold=3.0,
            margin=margin,
        )

    def test_thermal_emission(
        self,
        power_w: float = 100.0,
        ambient_c: float = 25.0,
        enclosure_type: str = "open",
    ) -> EMTestResult:
        """Test thermal emission from a component.

        Measures surface temperature rise above ambient.
        Pass if delta-T < 40°C for open enclosures, < 30°C for sealed.
        """
        threshold = 30.0 if enclosure_type == "sealed" else 40.0
        # Synthetic: 0.3°C/W thermal resistance + noise
        thermal_resistance = 0.3 + self._np_rng.normal(0, 0.05)
        delta_t = power_w * thermal_resistance

        passed = delta_t < threshold
        margin = threshold - delta_t

        return EMTestResult(
            test_name="thermal_emission",
            passed=passed,
            measurement=delta_t,
            unit="°C",
            threshold=threshold,
            margin=margin,
        )

    def test_power_line_noise(
        self,
        voltage_v: float = 12.0,
        load_a: float = 5.0,
    ) -> EMTestResult:
        """Test power line conducted noise.

        Measures ripple voltage (mV pp) on DC power line.
        Pass if ripple < 120mV for 12V systems.
        """
        # Synthetic: higher load = more ripple, + switching noise
        ripple_mv = (
            20.0 * (load_a / 5.0)
            + 10.0 * self._rng.random()
            + self._np_rng.normal(0, 5.0)
        )
        threshold = 120.0  # 1% of 12V

        passed = ripple_mv < threshold
        margin = threshold - ripple_mv

        return EMTestResult(
            test_name="power_line_noise",
            passed=passed,
            measurement=ripple_mv,
            unit="mVpp",
            threshold=threshold,
            margin=margin,
        )

    def test_rf_interference(
        self,
        transmit_power_dbm: float = 20.0,
        frequency_mhz: float = 2400.0,
        distance_m: float = 1.0,
    ) -> EMTestResult:
        """Test RF interference in nearby bands.

        Measures received power in adjacent channel (dBm).
        Pass if adjacent channel power < -40dBm.
        """
        # Synthetic: path loss + spurious emissions
        path_loss_db = 20.0 * np.log10(distance_m) + 20.0  # free space approx
        spurious_dbm = transmit_power_dbm - 50.0 + self._np_rng.normal(0, 3.0)
        received_dbm = spurious_dbm - path_loss_db

        threshold = -40.0
        passed = received_dbm < threshold
        margin = threshold - received_dbm

        return EMTestResult(
            test_name="rf_interference",
            passed=passed,
            measurement=received_dbm,
            unit="dBm",
            threshold=threshold,
            margin=margin,
        )

    def run_all(self, **kwargs: Any) -> dict[str, EMTestResult]:
        """Run all 4 EM tests and return results dict.

        Keyword args forwarded to individual tests.
        """
        return {
            "signal_integrity": self.test_signal_integrity(**kwargs),
            "thermal_emission": self.test_thermal_emission(**kwargs),
            "power_line_noise": self.test_power_line_noise(**kwargs),
            "rf_interference": self.test_rf_interference(**kwargs),
        }

    def summary(self, results: dict[str, EMTestResult] | None = None) -> dict[str, Any]:
        """Generate a human-readable summary of test results."""
        if results is None:
            results = self.run_all()

        total = len(results)
        passed = sum(1 for r in results.values() if r.passed)
        failed = total - passed

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "details": {
                name: {
                    "passed": r.passed,
                    "measurement": round(r.measurement, 2),
                    "unit": r.unit,
                    "threshold": r.threshold,
                    "margin": round(r.margin, 2),
                }
                for name, r in results.items()
            },
        }
