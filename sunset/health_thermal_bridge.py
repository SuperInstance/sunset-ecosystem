"""sunset/health_thermal_bridge.py — Wire cocapn-health into RoomGrid thermal budget.

Bidirectional bridge: cocapn-health emits thermal_snapshot events on the
FleetEventBus; this subscriber normalizes them into RoomGrid's thermal
model and triggers emergency breeding policies when pressure exceeds
threshold.

Usage:
    from nexus.fleet_event_bus import FleetEventBus
    from sunset.health_thermal_bridge import HealthThermalBridge

    bus = FleetEventBus()
    bridge = HealthThermalBridge(bus)
    bridge.subscribe()   # starts listening

    # When cocapn-health emits a thermal_snapshot, RoomGrid's thermal
    # budget is updated and emergency policies may trigger.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Optional nexus event bus ───────────────────────────────────
try:
    from nexus.fleet_event_bus import FleetEventBus
    _HAS_BUS = True
except Exception:
    FleetEventBus = None  # type: ignore[misc,assignment]
    _HAS_BUS = False

logger = logging.getLogger(__name__)


@dataclass
class ThermalReading:
    """Normalized thermal reading from any source."""
    source: str           # "cocapn-health", "roomgrid-profiler", etc.
    cpu_percent: float
    gpu_percent: float
    memory_percent: float
    temperature_c: float  # Highest sensor reading
    timestamp_ns: int = field(default_factory=time.time_ns)

    def pressure_score(self) -> float:
        """0.0 = cool, 1.0 = critical.  Weighted composite."""
        # GPU matters most for breeding (CUDA inference)
        return min(1.0, (
            0.4 * (self.gpu_percent / 100.0) +
            0.3 * (self.cpu_percent / 100.0) +
            0.2 * (self.memory_percent / 100.0) +
            0.1 * max(0.0, (self.temperature_c - 70.0) / 30.0)
        ))


class HealthThermalBridge:
    """Subscribe to FleetEventBus thermal events, update RoomGrid budget."""

    # Pressure thresholds (0.0–1.0)
    NORMAL = 0.5
    ELEVATED = 0.7
    CRITICAL = 0.9

    def __init__(
        self,
        bus: Any | None = None,
        grid: Any | None = None,
        emergency_callback: Callable | None = None,
    ) -> None:
        self._bus = bus
        self._grid = grid
        self._emergency_callback = emergency_callback
        self._readings: List[ThermalReading] = []
        self._last_pressure: float = 0.0
        self._subscribed: bool = False

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------
    def subscribe(self) -> bool:
        """Start listening for thermal_snapshot events."""
        if not _HAS_BUS or self._bus is None:
            logger.warning("FleetEventBus not available — HealthThermalBridge offline")
            return False
        self._bus.on("thermal_snapshot", self._on_thermal_snapshot)
        self._subscribed = True
        logger.info("HealthThermalBridge subscribed to thermal_snapshot")
        return True

    def unsubscribe(self) -> None:
        if self._subscribed and self._bus:
            self._bus.off("thermal_snapshot", self._on_thermal_snapshot)
            self._subscribed = False

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------
    def _on_thermal_snapshot(self, event: Dict[str, Any]) -> None:
        """Called when cocapn-health emits a thermal_snapshot."""
        payload = event.get("payload", event)
        reading = ThermalReading(
            source=payload.get("source", "cocapn-health"),
            cpu_percent=payload.get("cpu_percent", 0.0),
            gpu_percent=payload.get("gpu_percent", 0.0),
            memory_percent=payload.get("memory_percent", 0.0),
            temperature_c=payload.get("temperature_c", 0.0),
        )
        self._readings.append(reading)
        if len(self._readings) > 100:
            self._readings.pop(0)

        pressure = reading.pressure_score()
        self._last_pressure = pressure
        logger.debug("Thermal pressure=%.2f from %s", pressure, reading.source)

        # Update grid thermal model if available
        if self._grid and hasattr(self._grid, "thermal_pressure"):
            self._grid.thermal_pressure = pressure

        # Trigger emergency policies
        if pressure >= self.CRITICAL:
            self._trigger_emergency("CRITICAL", reading)
        elif pressure >= self.ELEVATED:
            self._trigger_emergency("ELEVATED", reading)

    # ------------------------------------------------------------------
    # Emergency response
    # ------------------------------------------------------------------
    def _trigger_emergency(self, level: str, reading: ThermalReading) -> None:
        logger.warning("THERMAL %s: pressure=%.2f", level, reading.pressure_score())
        if self._emergency_callback:
            self._emergency_callback(level, reading)
        elif self._grid and hasattr(self._grid, "thermal_policy"):
            self._grid.thermal_policy = level

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def pressure_history(self, window: int = 10) -> Tuple[float, float, float]:
        """Return (min, mean, max) pressure over last `window` readings."""
        recent = self._readings[-window:] if self._readings else []
        if not recent:
            return 0.0, 0.0, 0.0
        scores = [r.pressure_score() for r in recent]
        return float(min(scores)), float(np.mean(scores)), float(max(scores))

    def status(self) -> Dict[str, Any]:
        mn, mu, mx = self.pressure_history()
        return {
            "subscribed": self._subscribed,
            "last_pressure": self._last_pressure,
            "readings_count": len(self._readings),
            "pressure_min": mn,
            "pressure_mean": mu,
            "pressure_max": mx,
        }


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    bridge = HealthThermalBridge()
    r = ThermalReading("test", cpu=80, gpu=90, mem=70, temp=85)
    print(f"Pressure score = {r.pressure_score():.2f}")
    assert 0.7 <= r.pressure_score() <= 1.0
    print("HealthThermalBridge self-test passed.")
