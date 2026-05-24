"""Tests for nerve/a2a_metronome_tasks.py and nerve/a2a_conductor_integration.py.

Minimum 8 tests covering:
  1. BeatSyncTask validates
  2. BPMNegotiation accept
  3. BPMNegotiation counter
  4. DriftAlert triggers nudge
  5. DriftAlert triggers partition
  6. TickAsTask executes on grid
  7. Conductor dispatch to peer
  8. Conductor handles unknown task
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from nerve.a2a_metronome_tasks import (
    A2AMetronomeResult,
    BeatSyncTask,
    BPMNegotiationTask,
    DriftAlertTask,
    TickAsTask,
)
from nerve.a2a_conductor_integration import (
    FleetConductorA2AExtension,
    handle_unknown_task,
    register_a2a_handlers,
)
from nerve.room_grid import RoomGrid
from nexus.fleet_conductor import BeatState, FleetConductor


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def small_grid():
    """Return a small RoomGrid for tick execution tests."""
    np.random.seed(42)
    return RoomGrid(50)


@pytest.fixture
def conductor():
    """Return a FleetConductor with a mock scheduler attached."""
    fc = FleetConductor("test-node", "http://nexus.test:4047", max_drift_ms=5.0)
    scheduler = _MockScheduler()
    fc.register_local_scheduler(scheduler)
    return fc


class _MockScheduler:
    """Minimal scheduler stand-in for conductor tests."""

    def __init__(self, beat_number: int = 100, bpm: float = 120.0):
        self.beat_number = beat_number
        self.bpm = bpm
        self.last_nudge_ms: float | None = None
        self.last_jump_beat: int | None = None
        self.grid = None  # set externally if needed

    def nudge_phase(self, nudge_ms: float) -> None:
        self.last_nudge_ms = nudge_ms

    def jump_to_beat(self, beat_number: int) -> None:
        self.last_jump_beat = beat_number


# ═══════════════════════════════════════════════════════════════
# 1. BeatSyncTask validates
# ═══════════════════════════════════════════════════════════════

class TestBeatSyncTask:
    def test_beat_sync_task_validates(self):
        """BeatSyncTask.validate() must raise on malformed input."""
        valid = BeatSyncTask("node-a", 42, time.time_ns(), 120.0)
        valid.validate()  # should not raise

        invalid = BeatSyncTask("", -1, 0, -10.0)
        with pytest.raises(ValueError):
            invalid.validate()

    def test_beat_sync_roundtrip(self):
        """to_dict → from_dict must reconstruct an equivalent task."""
        original = BeatSyncTask("node-a", 42, 1_000_000_000, 240.0)
        d = original.to_dict()
        restored = BeatSyncTask.from_dict(d)
        assert restored.node_id == original.node_id
        assert restored.target_beat == original.target_beat
        assert restored.timestamp_ns == original.timestamp_ns
        assert restored.bpm == original.bpm

    def test_beat_sync_execute_returns_delta(self):
        """execute() must return delta_beats and delta_ns."""
        task = BeatSyncTask("node-a", 100, time.time_ns(), 120.0)
        result = task.execute(None)
        assert result.status == "completed"
        assert result.task_type == "beat_sync"
        assert "delta_beats" in result.payload
        assert "delta_ns" in result.payload


# ═══════════════════════════════════════════════════════════════
# 2. BPMNegotiation accept
# ═══════════════════════════════════════════════════════════════

class TestBPMNegotiationAccept:
    def test_bpm_negotiation_accept(self):
        """A proposal inside the valid range must be accepted."""
        task = BPMNegotiationTask(
            node_id="node-a",
            proposed_bpm=240.0,
            response_action="propose",
        )
        result = task.execute(None)
        assert result.status == "completed"
        assert result.payload["action"] == "accept"
        assert result.payload["accepted_bpm"] == 240.0

    def test_bpm_negotiation_counter(self):
        """A proposal outside the valid range must trigger a counter."""
        task = BPMNegotiationTask(
            node_id="node-a",
            proposed_bpm=999.0,
            response_action="propose",
        )
        result = task.execute(None)
        assert result.status == "completed"
        assert result.payload["action"] == "counter"
        assert "counter_bpm" in result.payload
        # Default max BPM is 480
        assert result.payload["counter_bpm"] == 480.0

    def test_bpm_negotiation_reject(self):
        """Explicit reject action must return rejected status."""
        task = BPMNegotiationTask(
            node_id="node-a",
            proposed_bpm=120.0,
            response_action="reject",
            reason="thermal overload",
        )
        result = task.execute(None)
        assert result.status == "rejected"
        assert result.payload["action"] == "reject"


# ═══════════════════════════════════════════════════════════════
# 4. DriftAlert triggers nudge
# 5. DriftAlert triggers partition
# ═══════════════════════════════════════════════════════════════

class TestDriftAlert:
    def test_drift_alert_triggers_nudge(self):
        """Moderate drift (> threshold but < 3× threshold, < 1 beat)
        must recommend phase_nudge."""
        task = DriftAlertTask(
            node_id="node-a",
            drift_ms=10.0,
            drift_beats=0.2,
            threshold_ms=5.0,
            requested_action="phase_nudge",
        )
        result = task.execute(None)
        assert result.status == "completed"
        assert result.payload["recommended_action"] == "phase_nudge"

    def test_drift_alert_triggers_partition(self):
        """Extreme drift (> 3× threshold) must recommend partition."""
        task = DriftAlertTask(
            node_id="node-a",
            drift_ms=20.0,
            drift_beats=0.5,
            threshold_ms=5.0,
            requested_action="phase_nudge",
        )
        result = task.execute(None)
        assert result.status == "completed"
        assert result.payload["recommended_action"] == "partition"

    def test_drift_alert_triggers_skip_jump(self):
        """Drift >= 1 full beat must recommend skip_jump."""
        task = DriftAlertTask(
            node_id="node-a",
            drift_ms=500.0,
            drift_beats=1.0,
            threshold_ms=5.0,
            requested_action="phase_nudge",
        )
        result = task.execute(None)
        assert result.status == "completed"
        assert result.payload["recommended_action"] == "skip_jump"

    def test_drift_alert_validates(self):
        """DriftAlertTask.validate() must reject bad inputs."""
        bad = DriftAlertTask(
            node_id="",
            drift_ms=-5.0,
            drift_beats=0.0,
            threshold_ms=0.0,
            requested_action="invalid_action",
        )
        with pytest.raises(ValueError):
            bad.validate()


# ═══════════════════════════════════════════════════════════════
# 6. TickAsTask executes on grid
# ═══════════════════════════════════════════════════════════════

class TestTickAsTask:
    def test_tick_as_task_executes_on_grid(self, small_grid):
        """TickAsTask.execute(grid) must tick the grid and return metadata."""
        signal = np.random.randn(64).astype(np.float32)
        task = TickAsTask(beat_number=7, signal=signal, room_ids=[0, 1, 2])
        result = task.execute(small_grid)
        assert result.status == "completed"
        assert result.task_type == "tick_as_task"
        assert result.beat_number == 7
        assert "elapsed_ms" in result.payload
        assert "fired" in result.payload
        assert "ids" in result.payload

    def test_tick_as_task_validates_bad_signal(self):
        """TickAsTask must reject a signal with wrong shape."""
        task = TickAsTask(beat_number=0, signal=np.zeros(32, dtype=np.float32))
        with pytest.raises(ValueError, match="signal must be shape"):
            task.validate()

    def test_tick_as_task_validates_negative_beat(self):
        """TickAsTask must reject a negative beat_number."""
        task = TickAsTask(beat_number=-1)
        with pytest.raises(ValueError, match="beat_number must be"):
            task.validate()


# ═══════════════════════════════════════════════════════════════
# 7. Conductor dispatch to peer
# ═══════════════════════════════════════════════════════════════

class TestConductorDispatch:
    def test_conductor_dispatch_sync_task(self, conductor):
        """FleetConductorA2AExtension.dispatch_sync_task must return a
        serialised BeatSyncTask payload."""
        payload = FleetConductorA2AExtension.dispatch_sync_task(
            conductor, "peer-b", target_beat=200
        )
        assert payload["type"] == "beat_sync"
        assert payload["input"]["target_beat"] == 200
        assert payload["input"]["node_id"] == "test-node"

    def test_conductor_dispatch_bpm_proposal(self, conductor):
        """dispatch_bpm_proposal must return a serialised BPMNegotiationTask."""
        payload = FleetConductorA2AExtension.dispatch_bpm_proposal(
            conductor, "peer-b", proposed_bpm=240.0, ramp_ms=1000
        )
        assert payload["type"] == "bpm_negotiation"
        assert payload["input"]["proposed_bpm"] == 240.0
        assert payload["input"]["response_action"] == "propose"

    def test_conductor_dispatch_drift_alert(self, conductor):
        """dispatch_drift_alert must return a serialised DriftAlertTask."""
        payload = FleetConductorA2AExtension.dispatch_drift_alert(
            conductor, "peer-b", drift_ms=12.0, drift_beats=0.3
        )
        assert payload["type"] == "drift_alert"
        assert payload["input"]["drift_ms"] == 12.0
        assert payload["input"]["threshold_ms"] == 5.0  # from conductor


# ═══════════════════════════════════════════════════════════════
# 8. Conductor handles unknown task
# ═══════════════════════════════════════════════════════════════

class TestConductorUnknownTask:
    def test_conductor_handles_unknown_task(self, conductor):
        """An unrecognised task type must return a failed result with an
        informative error message."""
        register_a2a_handlers(conductor)

        bad_task = {"type": "unicorn_dance", "input": {}}
        result = FleetConductorA2AExtension.handle_incoming_task(
            conductor, bad_task
        )
        assert result.status == "failed"
        assert "unicorn_dance" in (result.error or "")

    def test_handle_unknown_task_directly(self):
        """The free handle_unknown_task must return a failed result."""
        result = handle_unknown_task(None, {"type": "ghost_task"})
        assert result.status == "failed"
        assert "ghost_task" in (result.error or "")


# ═══════════════════════════════════════════════════════════════
# Integration: conductor + handler wiring
# ═══════════════════════════════════════════════════════════════

class TestConductorHandlerWiring:
    def test_register_a2a_handlers_populates_registry(self, conductor):
        """register_a2a_handlers must attach all four task handlers."""
        register_a2a_handlers(conductor)
        assert hasattr(conductor, "_a2a_handlers")
        assert "beat_sync" in conductor._a2a_handlers
        assert "bpm_negotiation" in conductor._a2a_handlers
        assert "drift_alert" in conductor._a2a_handlers
        assert "tick_as_task" in conductor._a2a_handlers

    def test_handle_incoming_beat_sync(self, conductor):
        """handle_incoming_task must route a beat_sync dict to the
        correct handler and return a completed result."""
        register_a2a_handlers(conductor)

        task_dict = BeatSyncTask("peer-c", 150, time.time_ns(), 120.0).to_dict()
        result = FleetConductorA2AExtension.handle_incoming_task(
            conductor, task_dict
        )
        assert isinstance(result, A2AMetronomeResult)
        assert result.status == "completed"
        assert result.task_type == "beat_sync"

    def test_handle_incoming_bpm_negotiation(self, conductor):
        """handle_incoming_task must accept a valid BPM proposal."""
        register_a2a_handlers(conductor)

        task_dict = BPMNegotiationTask(
            node_id="peer-c",
            proposed_bpm=180.0,
            response_action="propose",
        ).to_dict()
        result = FleetConductorA2AExtension.handle_incoming_task(
            conductor, task_dict
        )
        assert result.status == "completed"
        assert result.payload["action"] == "accept"

    def test_handle_incoming_drift_alert_applies_nudge(self, conductor):
        """A moderate drift alert must trigger _apply_phase_nudge on the
        conductor (via the handler)."""
        register_a2a_handlers(conductor)

        task_dict = DriftAlertTask(
            node_id="peer-c",
            drift_ms=12.0,
            drift_beats=0.2,
            threshold_ms=5.0,
            requested_action="phase_nudge",
            target_beat=200,
        ).to_dict()

        with patch.object(conductor, "_apply_phase_nudge") as mock_nudge:
            result = FleetConductorA2AExtension.handle_incoming_task(
                conductor, task_dict
            )
            mock_nudge.assert_called_once()

        assert result.status == "completed"
        assert result.payload["recommended_action"] == "phase_nudge"

    def test_handle_incoming_drift_alert_applies_skip_jump(self, conductor):
        """A large drift alert (drift_beats >= 1) must trigger skip_jump."""
        register_a2a_handlers(conductor)

        task_dict = DriftAlertTask(
            node_id="peer-c",
            drift_ms=550.0,
            drift_beats=1.1,
            threshold_ms=5.0,
            requested_action="phase_nudge",
            target_beat=300,
        ).to_dict()

        with patch.object(conductor, "_apply_skip_jump") as mock_jump:
            result = FleetConductorA2AExtension.handle_incoming_task(
                conductor, task_dict
            )
            mock_jump.assert_called_once()

        assert result.status == "completed"
        assert result.payload["recommended_action"] == "skip_jump"

    def test_handle_incoming_tick_as_task(self, conductor, small_grid):
        """TickAsTask sent to the conductor must execute on the
        scheduler's attached grid."""
        scheduler = _MockScheduler()
        scheduler.grid = small_grid
        conductor.register_local_scheduler(scheduler)
        register_a2a_handlers(conductor)

        signal = np.random.randn(64).astype(np.float32)
        task_dict = TickAsTask(beat_number=5, signal=signal).to_dict()
        result = FleetConductorA2AExtension.handle_incoming_task(
            conductor, task_dict
        )
        assert result.status == "completed"
        assert result.task_type == "tick_as_task"
        assert result.beat_number == 5
        assert "fired" in result.payload
        assert "elapsed_ms" in result.payload

    def test_handle_incoming_tick_as_task_no_grid(self, conductor):
        """TickAsTask with no grid available must return failed."""
        conductor.register_local_scheduler(_MockScheduler())
        register_a2a_handlers(conductor)

        task_dict = TickAsTask(beat_number=1).to_dict()
        result = FleetConductorA2AExtension.handle_incoming_task(
            conductor, task_dict
        )
        assert result.status == "failed"
        assert "No grid available" in (result.error or "")
