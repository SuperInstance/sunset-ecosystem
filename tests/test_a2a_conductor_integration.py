"""Tests for a2a_conductor_integration.py — A2A task handlers."""

import os

import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from nexus.fleet_conductor import FleetConductor
from nerve.a2a_metronome_tasks import (
    BeatSyncTask,
    BPMNegotiationTask,
    DriftAlertTask,
    TickAsTask,
    A2AMetronomeResult,
)
from nerve.a2a_conductor_integration import (
    register_a2a_handlers,
    handle_beat_sync_task,
    handle_bpm_proposal_task,
    handle_drift_alert_task,
    handle_tick_as_task,
    handle_unknown_task,
    FleetConductorA2AExtension,
    _A2A_HANDLER_REGISTRY,
)


@pytest.fixture
def conductor():
    """Return a FleetConductor with a mocked scheduler."""
    fc = FleetConductor("test-node", "http://test:4047", max_drift_ms=10.0)
    fc._scheduler = MagicMock()
    fc._scheduler.bpm = 120.0
    fc._scheduler.beat_number = 42
    fc._scheduler.grid = MagicMock()
    return fc


class TestRegisterHandlers:
    def test_registers_on_conductor(self, conductor):
        register_a2a_handlers(conductor)
        assert hasattr(conductor, "_a2a_handlers")
        assert "beat_sync" in conductor._a2a_handlers
        assert "bpm_negotiation" in conductor._a2a_handlers
        assert "drift_alert" in conductor._a2a_handlers
        assert "tick_as_task" in conductor._a2a_handlers

    def test_updates_module_registry(self, conductor):
        _A2A_HANDLER_REGISTRY.clear()
        register_a2a_handlers(conductor)
        assert "beat_sync" in _A2A_HANDLER_REGISTRY


class TestHandleBeatSyncTask:
    def test_no_scheduler(self):
        fc = FleetConductor("n", "http://test:4047")
        task = BeatSyncTask(node_id="peer", target_beat=5, timestamp_ns=1000, bpm=120.0)
        result = handle_beat_sync_task(fc, task)
        assert result.status == "completed"

    def test_small_drift(self, conductor):
        register_a2a_handlers(conductor)
        task = BeatSyncTask(node_id="peer", target_beat=5, timestamp_ns=1000, bpm=120.0)
        result = handle_beat_sync_task(conductor, task)
        assert result.status == "completed"
        # Small drift should not trigger skip-jump or phase nudge

    def test_large_drift_skip_jump(self, conductor):
        register_a2a_handlers(conductor)
        # Mock execute to return large drift
        with patch.object(
            BeatSyncTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="beat_sync",
                payload={
                    "delta_beats": 2.0,
                    "delta_ns": 20_000_000,
                    "beat_duration_ns": 500_000_000,
                },
            ),
        ):
            task = BeatSyncTask(
                node_id="peer", target_beat=5, timestamp_ns=1000, bpm=120.0
            )
            result = handle_beat_sync_task(conductor, task)
            assert result.status == "completed"

    def test_large_drift_phase_nudge(self, conductor):
        register_a2a_handlers(conductor)
        with patch.object(
            BeatSyncTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="beat_sync",
                payload={
                    "delta_beats": 0.5,
                    "delta_ns": 20_000_000,
                    "beat_duration_ns": 500_000_000,
                },
            ),
        ):
            task = BeatSyncTask(
                node_id="peer", target_beat=5, timestamp_ns=1000, bpm=120.0
            )
            result = handle_beat_sync_task(conductor, task)
            assert result.status == "completed"


class TestHandleBPMProposalTask:
    def test_accept(self, conductor):
        with patch.object(
            BPMNegotiationTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="bpm_negotiation",
                payload={"action": "accept", "accepted_bpm": 140.0},
            ),
        ):
            task = BPMNegotiationTask(
                node_id="peer",
                proposed_bpm=140.0,
                ramp_ms=2000,
                reason="test",
                response_action="propose",
            )
            result = handle_bpm_proposal_task(conductor, task)
            assert result.status == "completed"
            assert conductor._scheduler.bpm == 140.0

    def test_reject(self, conductor):
        old_bpm = conductor._scheduler.bpm
        with patch.object(
            BPMNegotiationTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="bpm_negotiation",
                payload={"action": "reject"},
            ),
        ):
            task = BPMNegotiationTask(
                node_id="peer",
                proposed_bpm=200.0,
                ramp_ms=2000,
                reason="test",
                response_action="propose",
            )
            result = handle_bpm_proposal_task(conductor, task)
            assert result.status == "completed"
            assert conductor._scheduler.bpm == old_bpm

    def test_no_scheduler(self):
        fc = FleetConductor("n", "http://test:4047")
        with patch.object(
            BPMNegotiationTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="bpm_negotiation",
                payload={"action": "accept", "accepted_bpm": 140.0},
            ),
        ):
            task = BPMNegotiationTask(
                node_id="peer",
                proposed_bpm=140.0,
                ramp_ms=2000,
                reason="test",
                response_action="propose",
            )
            result = handle_bpm_proposal_task(fc, task)
            assert result.status == "completed"


class TestHandleDriftAlertTask:
    def test_skip_jump(self, conductor):
        with patch.object(
            DriftAlertTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="drift_alert",
                payload={"recommended_action": "skip_jump"},
            ),
        ):
            task = DriftAlertTask(
                node_id="peer",
                drift_ms=15.0,
                drift_beats=2.0,
                threshold_ms=10.0,
                requested_action="skip_jump",
                target_beat=5,
            )
            result = handle_drift_alert_task(conductor, task)
            assert result.status == "completed"

    def test_partition(self, conductor):
        with patch.object(
            DriftAlertTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="drift_alert",
                payload={"recommended_action": "partition"},
            ),
        ):
            task = DriftAlertTask(
                node_id="peer",
                drift_ms=50.0,
                drift_beats=5.0,
                threshold_ms=10.0,
                requested_action="partition",
                target_beat=5,
            )
            result = handle_drift_alert_task(conductor, task)
            assert result.status == "completed"

    def test_phase_nudge(self, conductor):
        with patch.object(
            DriftAlertTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="drift_alert",
                payload={"recommended_action": "phase_nudge"},
            ),
        ):
            task = DriftAlertTask(
                node_id="peer",
                drift_ms=3.0,
                drift_beats=0.2,
                threshold_ms=10.0,
                requested_action="phase_nudge",
                target_beat=5,
            )
            result = handle_drift_alert_task(conductor, task)
            assert result.status == "completed"

    def test_no_action(self, conductor):
        with patch.object(
            DriftAlertTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="drift_alert",
                payload={"recommended_action": "none"},
            ),
        ):
            task = DriftAlertTask(
                node_id="peer",
                drift_ms=0.5,
                drift_beats=0.01,
                threshold_ms=10.0,
                requested_action="phase_nudge",
                target_beat=5,
            )
            result = handle_drift_alert_task(conductor, task)
            assert result.status == "completed"


class TestHandleTickAsTask:
    def test_with_grid(self, conductor):
        grid = MagicMock()
        conductor._scheduler.grid = grid
        with patch.object(
            TickAsTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="tick_as",
                payload={"tick": 1},
            ),
        ):
            task = TickAsTask(beat_number=1, signal=np.zeros(64, dtype=np.float32))
            result = handle_tick_as_task(conductor, task)
            assert result.status == "completed"

    def test_no_scheduler(self):
        fc = FleetConductor("n", "http://test:4047")
        task = TickAsTask(beat_number=1, signal=np.zeros(64, dtype=np.float32))
        result = handle_tick_as_task(fc, task)
        assert result.status == "failed"
        assert "No grid" in result.error

    def test_no_grid_attribute(self, conductor):
        del conductor._scheduler.grid
        task = TickAsTask(beat_number=1, signal=np.zeros(64, dtype=np.float32))
        result = handle_tick_as_task(conductor, task)
        assert result.status == "failed"
        assert "No grid" in result.error


class TestHandleUnknownTask:
    def test_dict_task(self):
        result = handle_unknown_task(None, {"type": "weird_task"})
        assert result.status == "failed"
        assert "weird_task" in result.error

    def test_object_task(self):
        class FakeTask:
            TASK_TYPE = "fake"

        result = handle_unknown_task(None, FakeTask())
        assert result.status == "failed"
        assert "fake" in result.error


class TestFleetConductorA2AExtension:
    def test_dispatch_sync_task(self, conductor):
        register_a2a_handlers(conductor)
        payload = FleetConductorA2AExtension.dispatch_sync_task(
            conductor, "peer-1", target_beat=10
        )
        assert payload["type"] == "beat_sync"
        assert payload["input"]["target_beat"] == 10

    def test_dispatch_sync_task_no_scheduler(self):
        fc = FleetConductor("n", "http://test:4047")
        payload = FleetConductorA2AExtension.dispatch_sync_task(
            fc, "peer-1", target_beat=10
        )
        assert payload["type"] == "beat_sync"
        assert payload["input"]["bpm"] == 120.0  # default

    def test_dispatch_bpm_proposal(self, conductor):
        register_a2a_handlers(conductor)
        payload = FleetConductorA2AExtension.dispatch_bpm_proposal(
            conductor,
            "peer-1",
            proposed_bpm=140.0,
            ramp_ms=3000,
            reason="load",
        )
        assert payload["type"] == "bpm_negotiation"
        assert payload["input"]["proposed_bpm"] == 140.0

    def test_dispatch_drift_alert(self, conductor):
        register_a2a_handlers(conductor)
        payload = FleetConductorA2AExtension.dispatch_drift_alert(
            conductor,
            "peer-1",
            drift_ms=15.0,
            drift_beats=2.0,
            requested_action="skip_jump",
        )
        assert payload["type"] == "drift_alert"
        assert payload["input"]["drift_ms"] == 15.0

    def test_dispatch_drift_alert_no_scheduler(self):
        fc = FleetConductor("n", "http://test:4047")
        payload = FleetConductorA2AExtension.dispatch_drift_alert(
            fc,
            "peer-1",
            drift_ms=15.0,
            drift_beats=2.0,
        )
        assert payload["type"] == "drift_alert"
        assert payload["input"]["target_beat"] == 0  # default

    def test_handle_incoming_task_beat_sync(self, conductor):
        register_a2a_handlers(conductor)
        task_dict = BeatSyncTask(
            node_id="peer", target_beat=5, timestamp_ns=1000, bpm=120.0
        ).to_dict()
        with patch.object(
            BeatSyncTask,
            "execute",
            return_value=A2AMetronomeResult(
                status="completed",
                task_type="beat_sync",
                payload={},
            ),
        ):
            result = FleetConductorA2AExtension.handle_incoming_task(
                conductor, task_dict
            )
            assert result.status == "completed"

    def test_handle_incoming_task_unknown(self, conductor):
        register_a2a_handlers(conductor)
        result = FleetConductorA2AExtension.handle_incoming_task(
            conductor, {"type": "unknown"}
        )
        assert result.status == "failed"
        assert "unknown" in result.error

    def test_handle_incoming_task_no_handlers(self):
        fc = FleetConductor("n", "http://test:4047")
        # No handlers registered - falls back to module registry
        _A2A_HANDLER_REGISTRY.clear()
        result = FleetConductorA2AExtension.handle_incoming_task(
            fc, {"type": "beat_sync"}
        )
        assert result.status == "failed"
