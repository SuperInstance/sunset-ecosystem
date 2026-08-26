"""A2A Metronome Tasks — Agent-to-Agent task types for metronome sync.

Defines four A2A task classes that enable distributed beat negotiation
across the Cocapn fleet:

  - BeatSyncTask        : "sync to my beat N at timestamp T"
  - BPMNegotiationTask  : "propose BPM=X, respond with accept/counter"
  - DriftAlertTask      : "my drift exceeds threshold, request partition or nudge"
  - TickAsTask          : wrap a single tick computation as an A2A task

Each task implements the A2A wire protocol:
    to_dict() / from_dict()  ↔ JSON payloads
    validate()               → schema + semantic checks
    execute(grid)            → run against a RoomGrid and return a Result
"""

from __future__ import annotations

__all__ = [
    "A2AMetronomeResult",
    "BeatSyncTask",
    "BPMNegotiationTask",
    "DriftAlertTask",
    "TickAsTask",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Result wrapper ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class A2AMetronomeResult:
    """Outcome of executing an A2A metronome task on a RoomGrid."""

    status: str  # "completed" | "failed" | "rejected"
    task_type: str
    beat_number: int = 0
    payload: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "task_type": self.task_type,
            "beat_number": self.beat_number,
            "payload": self.payload,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "A2AMetronomeResult":
        return cls(
            status=data.get("status", "unknown"),
            task_type=data.get("task_type", "unknown"),
            beat_number=data.get("beat_number", 0),
            payload=data.get("payload", {}),
            error=data.get("error"),
        )


# ── Base A2A Task Protocol ─────────────────────────────────


class A2AMetronomeTask:
    """Base class for all A2A metronome tasks.

    Subclasses must implement:
      - to_dict()     → serialise to A2A JSON payload
      - from_dict()   → deserialise from A2A JSON payload
      - validate()    → raise ValueError on malformed data
      - execute(grid) → run computation and return A2AMetronomeResult
    """

    TASK_TYPE: str = "base"

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "A2AMetronomeTask":
        raise NotImplementedError

    def validate(self) -> None:
        """Raise ValueError if this task is malformed."""
        raise NotImplementedError

    def execute(self, grid: Any) -> A2AMetronomeResult:
        """Run the task against a RoomGrid and return a Result."""
        raise NotImplementedError


# ── BeatSyncTask ───────────────────────────────────────────


class BeatSyncTask(A2AMetronomeTask):
    """ "Sync to my beat N at timestamp T."

    A peer node announces its current beat number and the wall-clock
    time when that beat occurred.  The receiver should align its own
    scheduler to match (or report how far off it is).
    """

    TASK_TYPE = "beat_sync"

    def __init__(
        self,
        node_id: str,
        target_beat: int,
        timestamp_ns: int,
        bpm: float = 120.0,
    ):
        self.node_id = node_id
        self.target_beat = target_beat
        self.timestamp_ns = timestamp_ns
        self.bpm = float(bpm)

    def to_dict(self) -> dict:
        return {
            "id": f"beat-sync-{self.node_id}-{self.target_beat}",
            "type": self.TASK_TYPE,
            "input": {
                "node_id": self.node_id,
                "target_beat": self.target_beat,
                "timestamp_ns": self.timestamp_ns,
                "bpm": self.bpm,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BeatSyncTask":
        inp = data.get("input", {})
        return cls(
            node_id=inp.get("node_id", "unknown"),
            target_beat=inp.get("target_beat", 0),
            timestamp_ns=inp.get("timestamp_ns", 0),
            bpm=inp.get("bpm", 120.0),
        )

    def validate(self) -> None:
        if not self.node_id or not isinstance(self.node_id, str):
            raise ValueError("beat_sync requires a non-empty node_id string")
        if self.target_beat < 0:
            raise ValueError("beat_sync target_beat must be >= 0")
        if self.timestamp_ns <= 0:
            raise ValueError("beat_sync timestamp_ns must be > 0")
        if self.bpm <= 0:
            raise ValueError("beat_sync bpm must be > 0")

    def execute(self, grid: Any) -> A2AMetronomeResult:
        """Return a result that advertises the local beat delta.

        In a real integration the scheduler would be nudged here.
        For the A2A wire layer we return the delta so the conductor
        can decide whether to apply a phase nudge or skip-jump.
        """
        now_ns = time.time_ns()
        delta_ns = now_ns - self.timestamp_ns
        beat_duration_ns = int((60.0 / self.bpm) * 1_000_000_000.0)
        delta_beats = delta_ns / beat_duration_ns if beat_duration_ns else 0.0

        return A2AMetronomeResult(
            status="completed",
            task_type=self.TASK_TYPE,
            beat_number=self.target_beat,
            payload={
                "delta_ns": delta_ns,
                "delta_beats": round(delta_beats, 4),
                "node_id": self.node_id,
                "beat_duration_ns": beat_duration_ns,
            },
        )


# ── BPMNegotiationTask ─────────────────────────────────────


class BPMNegotiationTask(A2AMetronomeTask):
    """ "Propose BPM=X, respond with accept/counter."

    One node proposes a new fleet-wide BPM.  The receiver can:
      - accept  → returns status="completed" with accepted_bpm
      - counter → returns status="completed" with counter_bpm
      - reject  → returns status="rejected" with reason
    """

    TASK_TYPE = "bpm_negotiation"

    def __init__(
        self,
        node_id: str,
        proposed_bpm: float,
        ramp_ms: int = 2000,
        reason: str = "",
        response_action: str = "propose",  # propose | accept | counter | reject
        counter_bpm: Optional[float] = None,
    ):
        self.node_id = node_id
        self.proposed_bpm = float(proposed_bpm)
        self.ramp_ms = int(ramp_ms)
        self.reason = reason
        self.response_action = response_action
        self.counter_bpm = float(counter_bpm) if counter_bpm is not None else None

    def to_dict(self) -> dict:
        payload: dict = {
            "id": f"bpm-neg-{self.node_id}-{self.response_action}",
            "type": self.TASK_TYPE,
            "input": {
                "node_id": self.node_id,
                "proposed_bpm": self.proposed_bpm,
                "ramp_ms": self.ramp_ms,
                "reason": self.reason,
                "response_action": self.response_action,
            },
        }
        if self.counter_bpm is not None:
            payload["input"]["counter_bpm"] = self.counter_bpm
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "BPMNegotiationTask":
        inp = data.get("input", {})
        return cls(
            node_id=inp.get("node_id", "unknown"),
            proposed_bpm=inp.get("proposed_bpm", 120.0),
            ramp_ms=inp.get("ramp_ms", 2000),
            reason=inp.get("reason", ""),
            response_action=inp.get("response_action", "propose"),
            counter_bpm=inp.get("counter_bpm"),
        )

    def validate(self) -> None:
        if not self.node_id or not isinstance(self.node_id, str):
            raise ValueError("bpm_negotiation requires a non-empty node_id")
        if self.proposed_bpm <= 0:
            raise ValueError("bpm_negotiation proposed_bpm must be > 0")
        if self.ramp_ms < 0:
            raise ValueError("bpm_negotiation ramp_ms must be >= 0")
        valid_actions = {"propose", "accept", "counter", "reject"}
        if self.response_action not in valid_actions:
            raise ValueError(
                f"bpm_negotiation response_action must be one of {valid_actions}"
            )
        if self.response_action == "counter" and self.counter_bpm is None:
            raise ValueError("bpm_negotiation counter action requires counter_bpm")

    def execute(self, grid: Any) -> A2AMetronomeResult:
        """Evaluate the proposal against local capabilities.

        If the grid has a scheduler attached, check that the proposed BPM
        is within the hardware-enforced range (60..480 by default).
        """
        # Determine local min/max BPM from scheduler if available
        min_bpm = 60.0
        max_bpm = 480.0
        if hasattr(grid, "_scheduler") and grid._scheduler is not None:
            sched = grid._scheduler
            min_bpm = getattr(sched, "min_bpm", min_bpm)
            max_bpm = getattr(sched, "max_bpm", max_bpm)

        action = self.response_action
        if action == "propose":
            # Evaluate proposal
            if min_bpm <= self.proposed_bpm <= max_bpm:
                return A2AMetronomeResult(
                    status="completed",
                    task_type=self.TASK_TYPE,
                    payload={
                        "action": "accept",
                        "accepted_bpm": self.proposed_bpm,
                        "ramp_ms": self.ramp_ms,
                    },
                )
            else:
                # Counter with clamped BPM
                counter = max(min_bpm, min(max_bpm, self.proposed_bpm))
                return A2AMetronomeResult(
                    status="completed",
                    task_type=self.TASK_TYPE,
                    payload={
                        "action": "counter",
                        "counter_bpm": counter,
                        "reason": f"proposed {self.proposed_bpm} outside [{min_bpm},{max_bpm}]",
                    },
                )

        if action == "accept":
            return A2AMetronomeResult(
                status="completed",
                task_type=self.TASK_TYPE,
                payload={"action": "accept", "accepted_bpm": self.proposed_bpm},
            )

        if action == "counter":
            return A2AMetronomeResult(
                status="completed",
                task_type=self.TASK_TYPE,
                payload={"action": "counter", "counter_bpm": self.counter_bpm},
            )

        # reject
        return A2AMetronomeResult(
            status="rejected",
            task_type=self.TASK_TYPE,
            payload={"action": "reject", "reason": self.reason or "no reason given"},
        )


# ── DriftAlertTask ─────────────────────────────────────────


class DriftAlertTask(A2AMetronomeTask):
    """ "My drift exceeds threshold, request partition or nudge."

    A node reports that its measured drift relative to the fleet
    consensus has crossed a threshold.  The conductor decides which
    correction strategy to apply:
      - phase_nudge  : smooth adjustment < 5 % of beat duration
      - skip_jump    : hard snap if drift >= 1 full beat
      - partition    : isolate node if quorum is lost or drift is extreme
    """

    TASK_TYPE = "drift_alert"

    # Actions the conductor may take in response
    ACTIONS = {"phase_nudge", "skip_jump", "partition"}

    def __init__(
        self,
        node_id: str,
        drift_ms: float,
        drift_beats: float,
        threshold_ms: float = 5.0,
        requested_action: str = "phase_nudge",
        target_beat: Optional[int] = None,
    ):
        self.node_id = node_id
        self.drift_ms = float(drift_ms)
        self.drift_beats = float(drift_beats)
        self.threshold_ms = float(threshold_ms)
        self.requested_action = requested_action
        self.target_beat = target_beat

    def to_dict(self) -> dict:
        payload: dict = {
            "id": f"drift-alert-{self.node_id}",
            "type": self.TASK_TYPE,
            "input": {
                "node_id": self.node_id,
                "drift_ms": self.drift_ms,
                "drift_beats": self.drift_beats,
                "threshold_ms": self.threshold_ms,
                "requested_action": self.requested_action,
            },
        }
        if self.target_beat is not None:
            payload["input"]["target_beat"] = self.target_beat
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "DriftAlertTask":
        inp = data.get("input", {})
        return cls(
            node_id=inp.get("node_id", "unknown"),
            drift_ms=inp.get("drift_ms", 0.0),
            drift_beats=inp.get("drift_beats", 0.0),
            threshold_ms=inp.get("threshold_ms", 5.0),
            requested_action=inp.get("requested_action", "phase_nudge"),
            target_beat=inp.get("target_beat"),
        )

    def validate(self) -> None:
        if not self.node_id or not isinstance(self.node_id, str):
            raise ValueError("drift_alert requires a non-empty node_id")
        if self.drift_ms < 0:
            raise ValueError("drift_alert drift_ms must be >= 0")
        if self.threshold_ms <= 0:
            raise ValueError("drift_alert threshold_ms must be > 0")
        if self.requested_action not in self.ACTIONS:
            raise ValueError(
                f"drift_alert requested_action must be one of {self.ACTIONS}"
            )

    def execute(self, grid: Any) -> A2AMetronomeResult:
        """Classify the alert and recommend a concrete action.

        Returns the action the conductor *should* take, not what the
        node requested.  The conductor may override (e.g. upgrade a
        nudge to a skip-jump when drift_beats >= 1).
        """
        # Conductor-level decision logic
        if self.drift_beats >= 1.0:
            recommended = "skip_jump"
        elif self.drift_ms > self.threshold_ms * 3:
            recommended = "partition"
        elif self.drift_ms > self.threshold_ms:
            recommended = "phase_nudge"
        else:
            recommended = "none"

        return A2AMetronomeResult(
            status="completed",
            task_type=self.TASK_TYPE,
            beat_number=self.target_beat or 0,
            payload={
                "recommended_action": recommended,
                "requested_action": self.requested_action,
                "drift_ms": self.drift_ms,
                "drift_beats": self.drift_beats,
                "threshold_ms": self.threshold_ms,
                "node_id": self.node_id,
            },
        )


# ── TickAsTask ─────────────────────────────────────────────


class TickAsTask(A2AMetronomeTask):
    """Wrap a single tick computation as an A2A task for remote execution.

    Instead of running ``grid.tick(signal)`` locally, the payload is
    serialised and sent to a peer node.  That node executes the tick
    and returns the fired-room list, beat number, and timing metadata.
    """

    TASK_TYPE = "tick_as_task"

    def __init__(
        self,
        beat_number: int,
        signal: Optional[np.ndarray] = None,
        room_ids: Optional[list[int]] = None,
        force: bool = False,
    ):
        self.beat_number = int(beat_number)
        self.signal = signal
        self.room_ids = room_ids
        self.force = bool(force)

    def to_dict(self) -> dict:
        sig_list = None
        if self.signal is not None:
            sig_list = self.signal.tolist()
        payload: dict = {
            "id": f"tick-task-{self.beat_number}",
            "type": self.TASK_TYPE,
            "input": {
                "beat_number": self.beat_number,
                "force": self.force,
            },
        }
        if sig_list is not None:
            payload["input"]["signal"] = sig_list
        if self.room_ids is not None:
            payload["input"]["room_ids"] = self.room_ids
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "TickAsTask":
        inp = data.get("input", {})
        sig = inp.get("signal")
        if sig is not None:
            sig = np.array(sig, dtype=np.float32)
        return cls(
            beat_number=inp.get("beat_number", 0),
            signal=sig,
            room_ids=inp.get("room_ids"),
            force=inp.get("force", False),
        )

    def validate(self) -> None:
        if self.beat_number < 0:
            raise ValueError("tick_as_task beat_number must be >= 0")
        if self.signal is not None:
            if not isinstance(self.signal, np.ndarray):
                raise ValueError("tick_as_task signal must be a numpy array")
            if self.signal.shape != (64,):
                raise ValueError("tick_as_task signal must be shape (64,)")
        if self.room_ids is not None:
            if not isinstance(self.room_ids, list):
                raise ValueError("tick_as_task room_ids must be a list")
            if any(not isinstance(r, int) or r < 0 for r in self.room_ids):
                raise ValueError("tick_as_task room_ids must be non-negative ints")

    def execute(self, grid: Any) -> A2AMetronomeResult:
        """Run grid.tick() (or a subset tick) and package the result."""
        import time as time_mod

        start = time_mod.perf_counter()

        # Build signal — zero-fill if none provided
        signal = self.signal
        if signal is None:
            signal = np.zeros(64, dtype=np.float32)

        # Full-grid or selective tick
        if self.room_ids is None:
            tick_result = grid.tick(signal)
        else:
            # Subset tick — tick the grid but mask to requested rooms
            # RoomGrid.tick() always processes all rooms; we report subset
            tick_result = grid.tick(signal)
            fired_subset = [r for r in tick_result.get("ids", []) if r in self.room_ids]
            tick_result = {
                "fired": len(fired_subset),
                "ids": fired_subset,
                "tick": self.beat_number,
            }

        elapsed_ms = (time_mod.perf_counter() - start) * 1000.0

        return A2AMetronomeResult(
            status="completed",
            task_type=self.TASK_TYPE,
            beat_number=self.beat_number,
            payload={
                "fired": tick_result.get("fired", 0),
                "ids": tick_result.get("ids", []),
                "tick": tick_result.get("tick", self.beat_number),
                "elapsed_ms": round(elapsed_ms, 3),
                "room_ids": self.room_ids,
                "force": self.force,
            },
        )
