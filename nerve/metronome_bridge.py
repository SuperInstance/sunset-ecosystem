"""Metronome Bridge — Synchronized multi-device dispatch for RoomGrid.

Maps metronome beats to selective RoomGrid ticks, enabling:
- Beat 0: Full grid tick (all rooms)
- Beat 1: Thermal-critical rooms only
- Beat 2: Breeding-ready rooms only
- Beat 3: Perception-active rooms only

Device-aware dispatch tracks latency per device type (cpu, gpu, rust).
"""

from __future__ import annotations

__all__ = ["MetronomeBridge"]

import logging
import time
from typing import List, Optional

import numpy as np

from nerve.room_grid import forward_one

log = logging.getLogger(__name__)


class MetronomeBridge:
    """Bridge FM's Metronome to sunset-ecosystem RoomGrid tick loop.

    The metronome runs at a fixed BPM. Each beat maps to a different
    dispatch pattern across the room grid, amortizing full-grid
    computation cost across the beat cycle while keeping critical
    rooms "fresh" on their dedicated beats.
    """

    def __init__(self, grid, scheduler):
        self.grid = grid
        self.scheduler = scheduler
        self._tick_counter = 0
        self._last_signal: Optional[np.ndarray] = None
        self._dispatched_rooms: List[int] = []
        self._latencies = {"cpu": [], "gpu": [], "rust": []}
        self._pending_ops = 0

    def on_metronome_beat(self, beat_number: int, tempo_ms: float) -> List[int]:
        """Called on every metronome beat.

        Maps metronome beats to RoomGrid ticks:
        - Beat 0: dispatch to all rooms (full grid tick)
        - Beat 1: dispatch to thermal-critical rooms
        - Beat 2: dispatch to breeding-ready rooms
        - Beat 3: dispatch to perception-active rooms

        Returns the list of room IDs dispatched this beat.
        """
        signal = self.scheduler.signal_source.next_signal(beat_number)
        self._last_signal = signal
        self._dispatched_rooms = []
        self._tick_counter += 1

        beat_mod = beat_number % 4

        if beat_mod == 0:
            # Full grid tick — all rooms
            result = self.grid.tick(signal)
            self._dispatched_rooms = list(range(self.grid.n))
            log.debug("Beat %d (full): ticked all %d rooms, fired=%d",
                      beat_number, self.grid.n, result.get("fired", 0))
        elif beat_mod == 1:
            # Thermal-critical rooms — highest chaos
            critical = self._get_thermal_critical_rooms()
            for room_id in critical:
                self.dispatch_room(room_id, "cpu")
            log.debug("Beat %d (thermal): dispatched %d critical rooms",
                      beat_number, len(critical))
        elif beat_mod == 2:
            # Breeding-ready rooms — moderate chaos, some activity
            ready = self._get_breeding_ready_rooms()
            for room_id in ready:
                self.dispatch_room(room_id, "cpu")
            log.debug("Beat %d (breed): dispatched %d ready rooms",
                      beat_number, len(ready))
        elif beat_mod == 3:
            # Perception-active rooms — highest activity
            active = self._get_perception_active_rooms()
            for room_id in active:
                self.dispatch_room(room_id, "cpu")
            log.debug("Beat %d (perception): dispatched %d active rooms",
                      beat_number, len(active))

        # After a full 4-beat cycle, synchronize all device queues
        if beat_mod == 3:
            self.sync_devices()

        return self._dispatched_rooms

    # ── Room selection heuristics ───────────────────────────

    def _get_thermal_critical_rooms(self, top_k: int = 10) -> List[int]:
        """Rooms with highest chaos values (thermal stress indicator)."""
        if self.grid.n == 0:
            return []
        # Sort by chaos descending, filter threshold
        idx = np.argsort(self.grid.chaos)[::-1]
        critical = [int(i) for i in idx[:top_k] if self.grid.chaos[i] > 0.15]
        return critical

    def _get_breeding_ready_rooms(self, top_k: int = 10) -> List[int]:
        """Rooms with moderate chaos, some activity history, ready to breed."""
        if self.grid.n == 0:
            return []
        mask = (self.grid.chaos > 0.05) & (self.grid.chaos < 0.35) & (self.grid.activity >= 1)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            return []
        if len(candidates) > top_k:
            # Prefer highest activity within the moderate-chaos band
            candidates = candidates[np.argsort(self.grid.activity[candidates])[-top_k:]]
        return [int(i) for i in candidates]

    def _get_perception_active_rooms(self, top_k: int = 10) -> List[int]:
        """Rooms with highest activity (perception / firing history)."""
        if self.grid.n == 0:
            return []
        idx = np.argsort(self.grid.activity)[::-1]
        active = [int(i) for i in idx[:top_k] if self.grid.activity[i] > 0]
        return active

    # ── Core dispatch ─────────────────────────────────────────

    def dispatch_room(self, room_id: int, device: str = "cpu") -> None:
        """Dispatch a single room's computation to specified device.

        Performs a lightweight forward pass for the room, updating
        its latent vector in-place. Full novelty/chaus/activity
        bookkeeping is deferred to the next full-grid tick (beat 0).
        """
        start = time.perf_counter()
        signal = self._last_signal
        if signal is None:
            signal = np.zeros(64, dtype=np.float32)

        # Single-room forward pass (lightweight)
        latent = forward_one(self.grid.w, room_id, signal)
        self.grid.latents[room_id] = latent

        # Track this room as dispatched
        self._dispatched_rooms.append(room_id)
        self._pending_ops += 1

        elapsed = time.perf_counter() - start
        self._latencies[device].append(elapsed)

    def sync_devices(self) -> None:
        """Synchronize all device queues after a full beat cycle.

        In a real multi-device system this would block until all
        pending GPU/Rust operations complete. Here we clear the
        pending-op counter and mark all queues as synchronized.
        """
        self._pending_ops = 0
        log.debug("Device sync complete: all queues flushed")

    def get_latency_report(self) -> dict:
        """Report tick latency per device type.

        Returns a dict keyed by device ('cpu', 'gpu', 'rust') with
        count, mean, min, max, and std in milliseconds.
        """
        report = {}
        for device, times in self._latencies.items():
            if times:
                arr = np.array(times)
                report[device] = {
                    "count": int(len(times)),
                    "mean_ms": float(np.mean(arr) * 1000),
                    "min_ms": float(np.min(arr) * 1000),
                    "max_ms": float(np.max(arr) * 1000),
                    "std_ms": float(np.std(arr) * 1000),
                }
            else:
                report[device] = {
                    "count": 0,
                    "mean_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "std_ms": 0.0,
                }
        return report
