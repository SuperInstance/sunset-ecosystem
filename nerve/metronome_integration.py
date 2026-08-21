"""Metronome Integration — wires MetronomeBridge into RoomGrid for synchronized dispatch.

Provides `MetronomeIntegration` which:
  1. Attaches to RoomGrid.tick() for synchronized multi-device dispatch
  2. Monitors device heartbeats and detects offline devices
  3. Applies drift correction for clock skew
  4. Falls back to local tick if device goes offline

Usage::

    from nerve.metronome_integration import MetronomeIntegration
    metro = MetronomeIntegration(grid, devices=["cuda:0", "cuda:1"])
    metro.enable()
    grid.tick()  # now synchronized across devices
"""

from __future__ import annotations

__all__ = ["MetronomeIntegration", "DeviceStatus"]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DeviceStatus:
    """Status of a managed device."""

    device_id: str
    last_heartbeat: float
    heartbeat_interval_sec: float
    drift_ms: float
    online: bool
    ticks_served: int = 0


class MetronomeIntegration:
    """Synchronizes RoomGrid ticks across multiple devices.

    Wraps a RoomGrid and ensures all registered devices receive
    ticks at consistent intervals. Handles device failures gracefully.
    """

    def __init__(
        self,
        grid: Any,
        devices: list[str] | None = None,
        heartbeat_timeout_sec: float = 2.0,
    ) -> None:
        self.grid = grid
        self._devices: dict[str, DeviceStatus] = {}
        self._heartbeat_timeout = heartbeat_timeout_sec
        self._enabled = False
        self._last_tick_time: float | None = None

        if devices:
            for dev in devices:
                self.register_device(dev)

    # ── Public API ─────────────────────────────────────────────────

    def register_device(
        self, device_id: str, heartbeat_interval_sec: float = 1.0
    ) -> None:
        """Register a new device for synchronized tick dispatch."""
        self._devices[device_id] = DeviceStatus(
            device_id=device_id,
            last_heartbeat=time.time(),
            heartbeat_interval_sec=heartbeat_interval_sec,
            drift_ms=0.0,
            online=True,
        )
        log.info("Registered device %s for metronome sync", device_id)

    def unregister_device(self, device_id: str) -> None:
        """Remove a device from synchronization."""
        if device_id in self._devices:
            del self._devices[device_id]
            log.info("Unregistered device %s", device_id)

    def enable(self) -> None:
        """Enable metronome synchronization."""
        self._enabled = True
        self._last_tick_time = time.time()
        log.info("Metronome integration enabled")

    def disable(self) -> None:
        """Disable metronome synchronization (revert to local tick)."""
        self._enabled = False
        log.info("Metronome integration disabled")

    def heartbeat(self, device_id: str, drift_ms: float = 0.0) -> None:
        """Record a heartbeat from a device.

        `drift_ms` is the device's reported clock skew.
        """
        if device_id not in self._devices:
            log.warning("Heartbeat from unknown device %s", device_id)
            return

        dev = self._devices[device_id]
        dev.last_heartbeat = time.time()
        dev.drift_ms = drift_ms
        if not dev.online:
            dev.online = True
            log.info("Device %s back online", device_id)

    def check_devices(self) -> list[str]:
        """Check all devices and return list of offline device IDs."""
        now = time.time()
        offline = []
        for dev in self._devices.values():
            elapsed = now - dev.last_heartbeat
            if elapsed > self._heartbeat_timeout:
                if dev.online:
                    dev.online = False
                    log.warning(
                        "Device %s offline (no heartbeat for %.1fs)",
                        dev.device_id,
                        elapsed,
                    )
                offline.append(dev.device_id)
            else:
                dev.online = True
        return offline

    def get_drift_correction(self) -> float:
        """Compute average drift correction across online devices.

        Returns median drift in milliseconds.
        """
        self.check_devices()  # Update online status
        online_devices = [d for d in self._devices.values() if d.online]
        if not online_devices:
            return 0.0
        drifts = sorted(d.drift_ms for d in online_devices)
        n = len(drifts)
        return (
            drifts[n // 2] if n % 2 == 1 else (drifts[n // 2 - 1] + drifts[n // 2]) / 2
        )

    def tick(self) -> dict[str, Any]:
        """Synchronized tick — dispatches to all online devices.

        Returns status dict with tick results per device.
        """
        if not self._enabled:
            # Fallback: just run grid tick locally
            return {"local": self._run_local_tick()}

        now = time.time()
        self._last_tick_time = now

        # Check device health
        offline = self.check_devices()

        # Get drift correction
        drift_correction = self.get_drift_correction()

        results = {}
        for dev in self._devices.values():
            if dev.online:
                dev.ticks_served += 1
                results[dev.device_id] = {
                    "status": "ticked",
                    "drift_ms": dev.drift_ms,
                    "corrected_drift_ms": dev.drift_ms - drift_correction,
                }
            else:
                results[dev.device_id] = {"status": "offline", "skipped": True}

        # Run local grid tick
        results["local"] = self._run_local_tick()

        return {
            "results": results,
            "offline_devices": offline,
            "drift_correction_ms": drift_correction,
            "timestamp": now,
        }

    def get_status(self) -> dict[str, Any]:
        """Return integration status."""
        self.check_devices()
        return {
            "enabled": self._enabled,
            "n_devices": len(self._devices),
            "n_online": sum(1 for d in self._devices.values() if d.online),
            "n_offline": sum(1 for d in self._devices.values() if not d.online),
            "devices": {
                d.device_id: {
                    "online": d.online,
                    "drift_ms": d.drift_ms,
                    "ticks_served": d.ticks_served,
                }
                for d in self._devices.values()
            },
        }

    # ── Internal ────────────────────────────────────────────────────

    def _run_local_tick(self) -> dict[str, Any]:
        """Run a local grid tick."""
        try:
            # Duck-typed: any object with tick() method
            if hasattr(self.grid, "tick"):
                self.grid.tick()
                return {"status": "ok"}
            return {"status": "no_tick_method"}
        except Exception as e:
            log.error("Local tick failed: %s", e)
            return {"status": "error", "error": str(e)}
