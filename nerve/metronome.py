"""Metronome Bridge — Periodic pulse generator for the nerve grid.

Wraps RoomGrid.tick() into a tempo-driven scheduler.
Each beat = one call to grid.tick(). Harmonics trigger routing,
breeding, and FLUX checks on sub-multiples of the beat.
"""

from __future__ import annotations

__all__ = [
    "LocalMetronome",
    "MetronomeScheduler",
    "SignalSource",
    "RandomSignalSource",
]

import logging
import threading
import time
from collections import deque
from typing import Optional, Protocol

import numpy as np

from nerve.room_grid import RoomGrid
from nerve.routing import RoutingLayer
from swarm.breeder_daemon import AutoBreeder

log = logging.getLogger(__name__)


# ── SignalSource Protocol ─────────────────────────────────

class SignalSource(Protocol):
    """Pluggable signal generator for the grid."""

    def next_signal(self, beat_number: int) -> np.ndarray:
        """Return a (64,) float32 signal for this beat."""


class RandomSignalSource:
    """White-noise signal source (default / test)."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)

    def next_signal(self, beat_number: int) -> np.ndarray:
        return self._rng.randn(64).astype(np.float32)


# ── LocalMetronome ────────────────────────────────────────

class LocalMetronome:
    """Per-room sub-oscillator.

    Divides the global BPM so the room only fires on a
    sub-multiple of the master beat. Phase offsets the fire
    point within the cycle (0.0-1.0).
    """

    def __init__(self, global_bpm: float, divider: int = 1, phase: float = 0.0):
        self.global_bpm = float(global_bpm)
        self.divider = max(1, int(divider))
        self.phase = float(phase)
        self.local_bpm = self.global_bpm / self.divider

    def should_fire(self, beat_number: int) -> bool:
        offset = int(self.phase * self.divider)
        return (beat_number + offset) % self.divider == 0

    def __repr__(self) -> str:
        return (
            f"LocalMetronome("
            f"global_bpm={self.global_bpm:.1f}, "
            f"divider={self.divider}, "
            f"phase={self.phase:.2f}, "
            f"local_bpm={self.local_bpm:.2f})"
        )


# ── MetronomeScheduler ────────────────────────────────────

class MetronomeScheduler:
    """Drives the nerve grid on a periodic beat.

    Responsibilities:
      1. Maintain master BPM and beat counter
      2. Call grid.tick(signal) on every beat
      3. Trigger routing on compiled + exploratory routes
      4. Fire breeding on harmonic multiples
      5. Report timing statistics (actual BPM, missed beats)
    """

    def __init__(
        self,
        grid: RoomGrid,
        router: RoutingLayer,
        breeder: AutoBreeder,
        bpm: float = 120.0,
        breeding_harmonic: int = 4,
        flux_harmonic: int = 16,
        signal_source: Optional[SignalSource] = None,
    ):
        self.grid = grid
        self.router = router
        self.breeder = breeder
        self.bpm = float(bpm)
        self.breeding_harmonic = max(1, int(breeding_harmonic))
        self.flux_harmonic = max(1, int(flux_harmonic))
        self.signal_source = signal_source or RandomSignalSource()

        self._beat_number = 0
        self._beat_times: deque[float] = deque(maxlen=10)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_tick_result: Optional[dict] = None

    # ── public API ───────────────────────────────────────

    @property
    def beat_number(self) -> int:
        """Monotonic beat counter (starts at 0, increments each tick)."""
        with self._lock:
            return self._beat_number

    @property
    def actual_bpm(self) -> float:
        """Measured BPM over the last 10 beat intervals."""
        with self._lock:
            if len(self._beat_times) < 2:
                return 0.0
            intervals = [
                self._beat_times[i] - self._beat_times[i - 1]
                for i in range(1, len(self._beat_times))
            ]
            avg_interval = sum(intervals) / len(intervals)
            return 60.0 / avg_interval if avg_interval > 0 else 0.0

    @property
    def beat_duration(self) -> float:
        """Target beat duration in seconds."""
        return 60.0 / self.bpm

    def start(self) -> None:
        """Begin the background thread loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="metronome-scheduler",
            daemon=True,
        )
        self._thread.start()
        log.info("MetronomeScheduler started (bpm=%.1f)", self.bpm)

    def stop(self) -> None:
        """Graceful shutdown of the background loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("MetronomeScheduler stopped")

    def tick_now(self) -> dict:
        """Manual beat — useful for testing and deterministic sync."""
        start = time.perf_counter()
        result = self._on_beat()
        elapsed = time.perf_counter() - start
        beat_dur = self.beat_duration
        if elapsed > beat_dur * 0.80:
            log.warning(
                "Metronome beat exceeded 80%% of beat duration: "
                "%.3fms / %.3fms",
                elapsed * 1000,
                beat_dur * 1000,
            )
        return result

    # ── internal phases ──────────────────────────────────

    def _on_beat(self) -> dict:
        with self._lock:
            beat = self._beat_number
            self._beat_number += 1
            self._beat_times.append(time.perf_counter())

        self._compute_phase(beat)
        self._gate_phase(beat)
        self._route_phase(beat)

        if beat % self.breeding_harmonic == 0:
            self._breed_phase(beat)

        if beat % self.flux_harmonic == 0:
            self._flux_phase(beat)

        return self._last_tick_result or {}

    def _compute_phase(self, beat_number: int) -> None:
        """Fetch signal and run grid forward pass."""
        signal = self.signal_source.next_signal(beat_number)
        self._last_tick_result = self.grid.tick(signal)

    def _gate_phase(self, beat_number: int) -> None:
        """Novelty/chaos gating.

        RoomGrid.tick() already performs vectorized novelty and chaos
        fire checks. Future: per-room LocalMetronome masking goes here.
        """
        pass

    def _route_phase(self, beat_number: int) -> None:
        """Fire compiled (deterministic) and exploratory routes."""
        # Compiled routes (strength > 0.9) fire every beat
        compiled = [
            r for r in self.router.routes.values() if r.strength > 0.9
        ]
        for r in compiled:
            r.fires += 1
            r.last_fired = time.time()

        # Exploratory routes via vectorized fire_fast
        self.router.fire_fast(source="grid", chaos=self.router.chaos)

    def _breed_phase(self, beat_number: int) -> None:
        """Trigger one breeding cycle."""
        try:
            self.breeder.cycle()
        except Exception:
            log.exception("Breeding cycle failed on beat %d", beat_number)

    def _flux_phase(self, beat_number: int) -> None:
        """Run FLUX constraint feedback if a checker is attached."""
        if self.grid._flux_checker is not None:
            try:
                from sunset.flux_integration import apply_constraint_feedback
                apply_constraint_feedback(self.grid, self.grid._flux_checker)
            except Exception:
                log.exception("FLUX phase failed on beat %d", beat_number)

    # ── background loop ──────────────────────────────────

    def _run_loop(self) -> None:
        beat_dur = self.beat_duration
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            self.tick_now()
            elapsed = time.perf_counter() - loop_start
            sleep_time = beat_dur - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
            else:
                log.warning(
                    "Metronome missed beat: overload "
                    "(elapsed %.3fms > beat %.3fms)",
                    elapsed * 1000,
                    beat_dur * 1000,
                )
