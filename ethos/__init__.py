"""Ethos-level modules for hardware survey and thermal calibration."""

from __future__ import annotations

__version__ = "0.1.0"

from ethos.hardware_survey import HardwareProfile, survey_hardware
from ethos.thermal_auto_calibrate import ThermalAutoCalibrator

__all__ = [
    "HardwareProfile",
    "survey_hardware",
    "ThermalAutoCalibrator",
]
