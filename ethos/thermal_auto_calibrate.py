"""Thermal Auto-Calibrator — learns thermal models from hardware profiler data.

Provides `ThermalAutoCalibrator` which:
  1. Calibrates from historical thermal + power + energy profiles
  2. Predicts thermal budget for N agents before spawning
  3. Rebalances load when thermal threshold is breached

Usage::

    from ethos.thermal_auto_calibrate import ThermalAutoCalibrator
    cal = ThermalAutoCalibrator()
    cal.calibrate_from_profile(profiles)
    budget = cal.predict_budget(n_agents=50)
    cal.rebalance_on_alert(current_load, threshold=0.85)
"""

from __future__ import annotations

__all__ = ["ThermalAutoCalibrator"]

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ThermalProfile:
    """Single observation of thermal state."""

    n_agents: int
    power_w: float
    temp_c: float
    energy_j: float
    timestamp: float


@dataclass
class ThermalBudget:
    """Predicted thermal budget for a given agent count."""

    n_agents: int
    max_safe_agents: int
    predicted_power_w: float
    predicted_temp_c: float
    confidence: float  # 0-1 based on calibration data density


class ThermalAutoCalibrator:
    """Learns thermal dynamics from fleet hardware profiler data.

    Uses linear regression with safety margins to predict thermal
    load and recommend rebalance actions.
    """

    def __init__(self, temp_threshold: float = 85.0) -> None:
        self.temp_threshold = temp_threshold
        self._profiles: list[ThermalProfile] = []
        self._power_per_agent: float = 5.0  # default 5W/agent
        self._temp_per_agent: float = 0.5  # default 0.5°C/agent
        self._intercept_power: float = 10.0
        self._intercept_temp: float = 35.0
        self._calibrated: bool = False

    # ── Public API ─────────────────────────────────────────────────

    def calibrate_from_profile(self, profiles: list[dict[str, Any]]) -> None:
        """Learn thermal model from hardware profiler history.

        Each profile dict must contain:
          - n_agents: int
          - power_w: float
          - temp_c: float (optional)
          - energy_j: float (optional)
        """
        if not profiles:
            log.warning("No profiles provided, using defaults")
            return

        self._profiles = [
            ThermalProfile(
                n_agents=p["n_agents"],
                power_w=p.get("power_w", 0.0),
                temp_c=p.get("temp_c", 0.0),
                energy_j=p.get("energy_j", 0.0),
                timestamp=p.get("timestamp", 0.0),
            )
            for p in profiles
        ]

        if len(self._profiles) < 2:
            log.warning("Need ≥2 profiles for calibration, using defaults")
            return

        # Simple linear regression: power = intercept + slope * n_agents
        n = np.array([p.n_agents for p in self._profiles], dtype=np.float64)
        power = np.array([p.power_w for p in self._profiles], dtype=np.float64)
        temps = np.array([p.temp_c for p in self._profiles], dtype=np.float64)

        # Fit power model
        A = np.vstack([n, np.ones(len(n))]).T
        power_slope, power_intercept = np.linalg.lstsq(A, power, rcond=None)[0]
        self._power_per_agent = max(power_slope, 0.1)
        self._intercept_power = max(power_intercept, 0.0)

        # Fit temperature model (only if we have temperature data)
        if np.any(temps > 0):
            temp_slope, temp_intercept = np.linalg.lstsq(A, temps, rcond=None)[0]
            self._temp_per_agent = max(temp_slope, 0.01)
            self._intercept_temp = max(temp_intercept, 20.0)

        self._calibrated = True
        log.info(
            "Calibrated: power=%.2fW/agent + %.2fW, temp=%.2f°C/agent + %.2f°C",
            self._power_per_agent,
            self._intercept_power,
            self._temp_per_agent,
            self._intercept_temp,
        )

    def predict_budget(self, n_agents: int) -> ThermalBudget:
        """Predict thermal budget for N agents.

        Returns `ThermalBudget` with max_safe_agents based on temp threshold.
        """
        predicted_power = self._intercept_power + self._power_per_agent * n_agents
        predicted_temp = self._intercept_temp + self._temp_per_agent * n_agents

        # Max safe agents before hitting threshold
        if self._temp_per_agent > 0:
            max_safe = int(
                (self.temp_threshold - self._intercept_temp) / self._temp_per_agent
            )
        else:
            max_safe = n_agents * 2  # generous fallback

        confidence = min(len(self._profiles) / 20.0, 1.0) if self._calibrated else 0.0

        return ThermalBudget(
            n_agents=n_agents,
            max_safe_agents=max_safe,
            predicted_power_w=predicted_power,
            predicted_temp_c=predicted_temp,
            confidence=confidence,
        )

    def rebalance_on_alert(
        self,
        current_load: dict[int, float],
        threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Recommend rebalance actions when thermal threshold is breached.

        `current_load` maps agent_id → thermal_load (0-1 scale).
        Returns dict with `actions` list and `agents_to_migrate` list.
        """
        total_load = sum(current_load.values())
        n_agents = len(current_load)
        avg_load = total_load / n_agents if n_agents > 0 else 0.0

        actions: list[str] = []
        agents_to_migrate: list[int] = []

        if avg_load < threshold:
            return {
                "status": "ok",
                "avg_load": avg_load,
                "actions": actions,
                "agents_to_migrate": agents_to_migrate,
            }

        # Sort by load descending, migrate hottest agents
        sorted_agents = sorted(current_load.items(), key=lambda x: x[1], reverse=True)
        target_load = threshold * n_agents
        current_total = total_load

        for agent_id, load in sorted_agents:
            if current_total <= target_load:
                break
            agents_to_migrate.append(agent_id)
            current_total -= load
            actions.append(f"migrate_agent_{agent_id}")

        return {
            "status": "rebalance_required",
            "avg_load": avg_load,
            "actions": actions,
            "agents_to_migrate": agents_to_migrate,
        }

    def save_model(self, path: str | Path) -> None:
        """Serialize calibration to JSON."""
        data = {
            "temp_threshold": self.temp_threshold,
            "power_per_agent": self._power_per_agent,
            "temp_per_agent": self._temp_per_agent,
            "intercept_power": self._intercept_power,
            "intercept_temp": self._intercept_temp,
            "calibrated": self._calibrated,
            "n_profiles": len(self._profiles),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_model(self, path: str | Path) -> None:
        """Deserialize calibration from JSON."""
        data = json.loads(Path(path).read_text())
        self.temp_threshold = data["temp_threshold"]
        self._power_per_agent = data["power_per_agent"]
        self._temp_per_agent = data["temp_per_agent"]
        self._intercept_power = data["intercept_power"]
        self._intercept_temp = data["intercept_temp"]
        self._calibrated = data["calibrated"]
