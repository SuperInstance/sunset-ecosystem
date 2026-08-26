"""RoomGrid Tick Integration — wires Metronome + Compiler + EventBus into RoomGrid.

Provides `RoomGridTickIntegration` which:
  1. Hooks CompilerHotSwap into RoomGrid.tick() for JIT recompilation
  2. Wires MetronomeIntegration for synchronized multi-device tick dispatch
  3. Emits per-tick metrics via FleetEventBus

Usage::

    from nerve.room_grid_tick_integration import RoomGridTickIntegration
    from nerve.room_grid import RoomGrid
    from nexus.fleet_event_bus import FleetEventBus

    grid = RoomGrid(250)
    bus = FleetEventBus()
    integration = RoomGridTickIntegration(grid, event_bus=bus)

    # Single tick with full instrumentation
    result = integration.tick(np.random.randn(64))

    # Batch tick with metronome synchronization
    integration = RoomGridTickIntegration(
        grid,
        metronome=MetronomeIntegration(grid, devices=["cuda:0"]),
        event_bus=bus,
    )
    results = integration.tick_batch(signals)
"""

from __future__ import annotations

__all__ = ["RoomGridTickIntegration", "TickMetrics"]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TickMetrics:
    """Metrics emitted per tick cycle."""

    tick: int
    n_rooms: int
    fired_count: int
    active_ratio: float
    thermal_pressure: float
    backend: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "n_rooms": self.n_rooms,
            "fired_count": self.fired_count,
            "active_ratio": self.active_ratio,
            "thermal_pressure": self.thermal_pressure,
            "backend": self.backend,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class RoomGridTickIntegration:
    """Orchestrates RoomGrid ticks with metronome sync, compiler hot-swap,
    and fleet-wide event telemetry.

    The integration is **non-invasive**: it wraps RoomGrid but does not
    monkey-patch it.  Existing tests and standalone usage of RoomGrid
    are unaffected.
    """

    def __init__(
        self,
        grid: Any,
        metronome: Any | None = None,
        compiler_swap: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.grid = grid
        self.metronome = metronome
        self.compiler_swap = compiler_swap
        self.event_bus = event_bus
        self._tick_count = 0
        self._total_duration_ms = 0.0
        self._enabled = True

    # ── Public API ─────────────────────────────────────────────────

    def enable(self) -> None:
        """Enable integration (default)."""
        self._enabled = True
        log.info("RoomGridTickIntegration enabled")

    def disable(self) -> None:
        """Disable integration — tick() falls back to raw grid.tick()."""
        self._enabled = False
        log.info("RoomGridTickIntegration disabled")

    def tick(self, x: np.ndarray) -> dict[str, Any]:
        """Run a single tick with compiler + metronome + event bus hooks.

        Falls back to ``grid.tick(x)`` if the integration is disabled
        or if any optional component is unavailable.
        """
        if not self._enabled:
            return self.grid.tick(x)

        t0 = time.perf_counter()

        # 1. Compiler hot-swap check — trigger recompile if config changed
        self._check_compiler()

        # 2. Metronome pre-tick (device health check, drift correction)
        metronome_status = self._pre_metronome()

        # 3. Run the actual grid tick
        try:
            result = self.grid.tick(x)
        except Exception as e:
            log.error("RoomGrid tick failed: %s", e)
            self._emit_event("grid_tick_error", {"error": str(e)})
            raise

        # 4. Post-tick: metronome sync
        if metronome_status:
            result["metronome"] = metronome_status

        # 5. Emit metrics
        duration_ms = (time.perf_counter() - t0) * 1000
        self._tick_count += 1
        self._total_duration_ms += duration_ms
        metrics = self._build_metrics(result, duration_ms)
        self._emit_metrics(metrics)

        result["duration_ms"] = duration_ms
        return result

    def tick_batch(self, signals: np.ndarray) -> list[dict[str, Any]]:
        """Run a batch tick with synchronized dispatch.

        If ``metronome`` is enabled, this performs a synchronized tick
        across all registered devices before running the local batch.
        Falls back to ``grid.tick_batch(signals)`` when unavailable.
        """
        if not self._enabled:
            return self.grid.tick_batch(signals)

        t0 = time.perf_counter()

        # 1. Compiler hot-swap check
        self._check_compiler()

        # 2. Metronome synchronized dispatch
        metronome_results: dict[str, Any] | None = None
        if self.metronome is not None and hasattr(self.metronome, "tick"):
            try:
                metronome_results = self.metronome.tick()
            except Exception as e:
                log.warning("Metronome tick failed: %s", e)
                self._emit_event("metronome_tick_error", {"error": str(e)})

        # 3. Run local batch tick
        try:
            results = self.grid.tick_batch(signals)
        except Exception as e:
            log.error("RoomGrid tick_batch failed: %s", e)
            self._emit_event("grid_tick_batch_error", {"error": str(e)})
            raise

        # 4. Attach metronome metadata
        if metronome_results:
            for r in results:
                r["metronome"] = metronome_results

        # 5. Emit aggregate metrics
        duration_ms = (time.perf_counter() - t0) * 1000
        self._tick_count += 1
        self._total_duration_ms += duration_ms

        # Aggregate results for metrics
        total_fired = sum(r.get("fired", 0) for r in results)
        aggregated = {
            "tick": getattr(self.grid, "ticks", 0),
            "fired": total_fired,
            "ids": [],
            "batch_size": len(results),
        }
        metrics = self._build_metrics(aggregated, duration_ms)
        self._emit_metrics(metrics)

        # Add duration to each result
        for r in results:
            r["duration_ms"] = duration_ms / max(len(results), 1)
        return results

    def get_status(self) -> dict[str, Any]:
        """Return integration health and statistics."""
        avg_duration = (
            self._total_duration_ms / self._tick_count if self._tick_count > 0 else 0.0
        )
        return {
            "enabled": self._enabled,
            "tick_count": self._tick_count,
            "avg_duration_ms": round(avg_duration, 3),
            "has_metronome": self.metronome is not None,
            "has_compiler_swap": self.compiler_swap is not None,
            "has_event_bus": self.event_bus is not None,
            "grid_n_rooms": getattr(self.grid, "n", None),
            "grid_ticks": getattr(self.grid, "ticks", None),
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _check_compiler(self) -> None:
        """Trigger compiler hot-swap if config changed."""
        if self.compiler_swap is None:
            return
        if not hasattr(self.compiler_swap, "check_and_compile"):
            return
        try:
            result = self.compiler_swap.check_and_compile()
            if result is not None and getattr(result, "success", False):
                log.debug(
                    "Compiler hot-swap triggered: compile_time_ms=%.2f",
                    getattr(result, "compile_time_ms", 0),
                )
                self._emit_event(
                    "compiler_hot_swap",
                    {
                        "success": result.success,
                        "compile_time_ms": getattr(result, "compile_time_ms", 0),
                    },
                )
        except Exception as e:
            log.warning("Compiler check failed (non-fatal): %s", e)

    def _pre_metronome(self) -> dict[str, Any] | None:
        """Run metronome pre-tick checks and return status."""
        if self.metronome is None:
            return None
        status: dict[str, Any] = {}

        # Device health check
        if hasattr(self.metronome, "check_devices"):
            try:
                offline = self.metronome.check_devices()
                status["offline_devices"] = offline
            except Exception as e:
                log.warning("Metronome device check failed: %s", e)

        # Drift correction
        if hasattr(self.metronome, "get_drift_correction"):
            try:
                drift = self.metronome.get_drift_correction()
                status["drift_correction_ms"] = drift
            except Exception as e:
                log.warning("Metronome drift check failed: %s", e)

        # Enable metronome if not already enabled
        if hasattr(self.metronome, "_enabled") and not self.metronome._enabled:
            if hasattr(self.metronome, "enable"):
                self.metronome.enable()

        return status if status else None

    def _build_metrics(
        self, tick_result: dict[str, Any], duration_ms: float
    ) -> TickMetrics:
        """Build TickMetrics from a tick result dict."""
        n = getattr(self.grid, "n", 0)
        active = getattr(self.grid, "activity", None)
        chaos = getattr(self.grid, "chaos", None)

        fired_count = tick_result.get("fired", 0)
        active_count = int((active > 0).sum()) if active is not None else 0
        active_ratio = active_count / n if n > 0 else 0.0

        # Thermal pressure = mean chaos level (proxy for thermal load)
        thermal_pressure = float(chaos.mean()) if chaos is not None else 0.0

        # Detect backend from grid
        backend = "numpy"
        if hasattr(self.grid, "_cuda_grid"):
            backend = "cuda"
        elif hasattr(self.grid, "_rust_grid"):
            backend = "rust_persistent"
        elif hasattr(self.grid, "_select_backend"):
            # Try calling the module-level function
            try:
                from nerve.room_grid import _select_backend

                backend = _select_backend(n)
            except Exception:
                backend = "numpy"

        return TickMetrics(
            tick=tick_result.get("tick", getattr(self.grid, "ticks", 0)),
            n_rooms=n,
            fired_count=fired_count,
            active_ratio=round(active_ratio, 3),
            thermal_pressure=round(thermal_pressure, 4),
            backend=backend,
            duration_ms=round(duration_ms, 3),
        )

    def _emit_metrics(self, metrics: TickMetrics) -> None:
        """Emit metrics via FleetEventBus if available."""
        self._emit_event("grid_tick_metrics", metrics.to_dict())

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget event emit (safe if no bus attached)."""
        if self.event_bus is None:
            return
        if not hasattr(self.event_bus, "emit"):
            return
        try:
            self.event_bus.emit(
                {"type": event_type, **payload}, source="room_grid_tick_integration"
            )
        except Exception as e:
            log.warning("EventBus emit failed (non-fatal): %s", e)
