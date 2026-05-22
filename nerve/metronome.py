"""Metronome Bridge — Periodic pulse generator for the nerve grid.

Wraps RoomGrid.tick() into a tempo-driven scheduler.
Each beat = one call to grid.tick(). Harmonics trigger routing,
breeding, and FLUX checks on sub-multiples of the beat.
"""

from __future__ import annotations

__all__ = [
    "A2ASignalSource",
    "LocalMetronome",
    "MetronomeScheduler",
    "SignalSource",
    "RandomSignalSource",
    "TickAsTask",
]

import json
import logging
import threading
import time
import urllib.request
from collections import deque
from typing import Optional, Protocol

import numpy as np

from logos.intent_protocol import FleetState, IntentConfirmationProtocol
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


# ── A2ASignalSource ─────────────────────────────────────

class A2ASignalSource:
    """Fetch input signals from an A2A agent via HTTP.

    On every beat, POST a SignalRequest to the A2A endpoint and
    expect a 64-dim float32 vector in the response payload.
    """

    def __init__(
        self,
        endpoint_url: str,
        agent_card_path: Optional[str] = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.agent_card_path = agent_card_path
        self._agent_card: Optional[dict] = None

    def next_signal(self, beat_number: int) -> np.ndarray:
        """POST to the A2A agent and return the signal vector."""
        payload = json.dumps({
            "id": f"signal-beat-{beat_number}",
            "type": "get_signal",
            "input": {"beat_number": beat_number, "dimensions": 64},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.endpoint_url}/a2a/tasks/send",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            log.warning("A2ASignalSource fetch failed: %s", exc)
            # Fallback: return zeros so the grid keeps ticking
            return np.zeros(64, dtype=np.float32)

        # Extract signal from A2A response artefacts
        artefacts = data.get("artefacts", [])
        for art in artefacts:
            content = art.get("content", {})
            signal = content.get("signal")
            if signal is not None:
                vec = np.array(signal, dtype=np.float32)
                if vec.shape == (64,):
                    return vec
                # Try to reshape / pad to 64
                if vec.size >= 64:
                    return vec[:64].astype(np.float32)
                padded = np.zeros(64, dtype=np.float32)
                padded[:vec.size] = vec
                return padded

        # No signal artefact found — fallback to zeros
        return np.zeros(64, dtype=np.float32)

    def get_metadata(self) -> dict:
        """Return source metadata (endpoint + cached agent card)."""
        return {
            "endpoint_url": self.endpoint_url,
            "agent_card_path": self.agent_card_path,
            "agent_card": self._agent_card,
        }


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


# ── TickAsTask ────────────────────────────────────────────

class TickAsTask:
    """Convert each metronome beat into an A2A task payload.

    When ``task_mode=True`` on the scheduler, every beat is packaged
    as an A2A ``tasks/send`` request rather than running compute
    phases directly in-process.
    """

    def __init__(
        self,
        scheduler: MetronomeScheduler,
        task_queue: Optional[list] = None,
    ):
        self.scheduler = scheduler
        self.task_queue = task_queue or []
        self._pending_task_ids: deque[str] = deque()

    def on_beat(self, beat_number: int) -> dict:
        """Build an A2A task payload for this beat."""
        signal = self.scheduler.signal_source.next_signal(beat_number)
        signal_list = signal.tolist()

        payload = {
            "id": f"tick-{beat_number}",
            "type": "tick",
            "input": {
                "signal": signal_list,
                "beat_number": beat_number,
                "force": False,
            },
        }
        return payload

    def submit_task(self, payload: dict) -> dict:
        """POST the payload to the A2A ``/a2a/tasks/send`` endpoint."""
        endpoint = "http://nexus.fleet.local:4047/metronome"
        if hasattr(self.scheduler, "a2a_endpoint"):
            endpoint = self.scheduler.a2a_endpoint

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/a2a/tasks/send",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            log.warning("TickAsTask submit failed: %s", exc)
            result = {"status": "failed", "error": str(exc)}

        task_id = result.get("id", payload.get("id"))
        if task_id:
            self._pending_task_ids.append(task_id)
        return result

    def collect_results(self) -> list[dict]:
        """Gather async results from previously submitted ticks.

        Returns a list of result dicts for all pending task IDs.
        In a full implementation this would poll ``/a2a/tasks/{id}``.
        For now we return synthetic completions for any pending IDs
        and clear the backlog.
        """
        results = []
        while self._pending_task_ids:
            task_id = self._pending_task_ids.popleft()
            # Synthetic completion — real impl would GET /a2a/tasks/{task_id}
            results.append({
                "id": task_id,
                "status": "completed",
                "artefacts": [{
                    "type": "TickResult",
                    "content": {
                        "beat_number": int(task_id.split("-")[-1]) if "-" in task_id else 0,
                        "fired_rooms": [],
                        "fired_count": 0,
                        "missed_beat": False,
                    },
                }],
            })
        return results

    def __repr__(self) -> str:
        return (
            f"TickAsTask(pending={len(self._pending_task_ids)}, "
            f"queue_size={len(self.task_queue)})"
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
        task_mode: bool = False,
        a2a_endpoint: Optional[str] = None,
    ):
        self.grid = grid
        self.router = router
        self.breeder = breeder
        self.bpm = float(bpm)
        self.breeding_harmonic = max(1, int(breeding_harmonic))
        self.flux_harmonic = max(1, int(flux_harmonic))
        self.signal_source = signal_source or RandomSignalSource()
        self.task_mode = bool(task_mode)
        self.a2a_endpoint = a2a_endpoint or "http://nexus.fleet.local:4047/metronome"

        self._tick_as_task: Optional[TickAsTask] = None
        if self.task_mode:
            self._tick_as_task = TickAsTask(self)

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

        if self.task_mode and self._tick_as_task is not None:
            # In task mode: package beat as A2A task, submit, and return
            payload = self._tick_as_task.on_beat(beat)
            result = self._tick_as_task.submit_task(payload)
            self._last_tick_result = result
            return result

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
