"""Tests for Metronome P2 — A2ASignalSource + tick-as-task.

P2 A2A-First Integration: metronome is a first-class A2A agent.
"""

from __future__ import annotations

import json
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nerve.metronome import (
    A2ASignalSource,
    MetronomeScheduler,
    RandomSignalSource,
    TickAsTask,
)
from nerve.room_grid import RoomGrid
from nerve.routing import RoutingLayer
from swarm.breeder_daemon import AutoBreeder
from swarm.thermal import ThermalBudget


@pytest.fixture
def scheduler():
    """Return a MetronomeScheduler wired to a small grid."""
    np.random.seed(42)
    grid = RoomGrid(100)
    router = RoutingLayer()
    for i in range(5):
        router.add_route("grid", f"dst_{i}")
    thermal = ThermalBudget()
    breeder = AutoBreeder(grid, thermal, interval=4)
    return MetronomeScheduler(
        grid=grid,
        router=router,
        breeder=breeder,
        bpm=120.0,
        breeding_harmonic=4,
        flux_harmonic=16,
        signal_source=RandomSignalSource(seed=42),
    )


# ── A2ASignalSource ────────────────────────────────────────


def test_a2a_signal_source_fetch():
    """A2ASignalSource should POST to endpoint and extract the signal vector."""
    src = A2ASignalSource(
        endpoint_url="http://test.local:8080",
        agent_card_path="/.well-known/agent-signal.json",
    )

    # Build a mock A2A response with a 64-dim signal
    mock_response = MagicMock()
    expected_signal = np.linspace(-1, 1, 64).astype(np.float32)
    mock_response.read.return_value = json.dumps(
        {
            "id": "task-001",
            "status": "completed",
            "artefacts": [
                {
                    "type": "SignalPayload",
                    "content": {"signal": expected_signal.tolist()},
                }
            ],
        }
    ).encode("utf-8")
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        signal = src.next_signal(beat_number=7)

    assert signal.shape == (64,)
    assert signal.dtype == np.float32
    np.testing.assert_allclose(signal, expected_signal, atol=1e-6)

    meta = src.get_metadata()
    assert meta["endpoint_url"] == "http://test.local:8080"
    assert meta["agent_card_path"] == "/.well-known/agent-signal.json"


def test_a2a_signal_source_fallback_on_error():
    """If the A2A endpoint fails, A2ASignalSource should return zeros."""
    src = A2ASignalSource(endpoint_url="http://down.local:9999")

    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        signal = src.next_signal(beat_number=0)

    assert signal.shape == (64,)
    assert signal.dtype == np.float32
    assert np.all(signal == 0.0)


def test_a2a_signal_source_pad_short_vector():
    """If the returned signal is shorter than 64 dims, pad with zeros."""
    src = A2ASignalSource(endpoint_url="http://test.local:8080")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "artefacts": [{"content": {"signal": [1.0, 2.0, 3.0]}}],
        }
    ).encode("utf-8")
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        signal = src.next_signal(beat_number=0)

    assert signal.shape == (64,)
    assert signal[0] == 1.0
    assert signal[1] == 2.0
    assert signal[2] == 3.0
    assert np.all(signal[3:] == 0.0)


# ── TickAsTask ───────────────────────────────────────────


def test_tick_as_task_payload_shape(scheduler):
    """TickAsTask.on_beat() should produce a valid A2A task payload."""
    task = TickAsTask(scheduler)
    payload = task.on_beat(beat_number=42)

    assert payload["id"] == "tick-42"
    assert payload["type"] == "tick"
    assert payload["input"]["beat_number"] == 42
    assert payload["input"]["force"] is False
    assert isinstance(payload["input"]["signal"], list)
    assert len(payload["input"]["signal"]) == 64


def test_tick_as_task_submit_and_collect(scheduler):
    """TickAsTask.submit_task() should POST and collect_results() should return completions."""
    task = TickAsTask(scheduler)
    payload = task.on_beat(beat_number=1)

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "id": "tick-1",
            "status": "submitted",
        }
    ).encode("utf-8")
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = task.submit_task(payload)

    assert result["id"] == "tick-1"
    assert task._pending_task_ids == deque(["tick-1"])

    # collect_results should synthesize completions and clear backlog
    results = task.collect_results()
    assert len(results) == 1
    assert results[0]["id"] == "tick-1"
    assert results[0]["status"] == "completed"
    assert len(task._pending_task_ids) == 0


def test_tick_as_task_repr(scheduler):
    """TickAsTask repr should show pending count."""
    task = TickAsTask(scheduler)
    assert "pending=0" in repr(task)


# ── MetronomeScheduler task_mode ─────────────────────────


def test_task_mode_scheduler(scheduler):
    """When task_mode=True, tick_now() should submit an A2A task instead of direct phases."""
    # Create a task-mode scheduler reusing the same components
    task_scheduler = MetronomeScheduler(
        grid=scheduler.grid,
        router=scheduler.router,
        breeder=scheduler.breeder,
        bpm=240.0,
        signal_source=RandomSignalSource(seed=99),
        task_mode=True,
        a2a_endpoint="http://mock.nexus:4047/metronome",
    )

    assert task_scheduler.task_mode is True
    assert task_scheduler._tick_as_task is not None
    assert task_scheduler.a2a_endpoint == "http://mock.nexus:4047/metronome"

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "id": "tick-0",
            "status": "completed",
            "artefacts": [
                {
                    "type": "TickResult",
                    "content": {
                        "beat_number": 0,
                        "fired_rooms": [1, 2, 3],
                        "fired_count": 3,
                    },
                }
            ],
        }
    ).encode("utf-8")
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = task_scheduler.tick_now()

    # Result should come from the A2A response, not from grid.tick()
    assert result["status"] == "completed"
    assert result["artefacts"][0]["content"]["fired_count"] == 3

    # Beat counter should still advance
    assert task_scheduler.beat_number == 1


def test_task_mode_false_runs_direct_phases(scheduler):
    """When task_mode=False (default), tick_now() runs compute/route/breed directly."""
    assert scheduler.task_mode is False
    assert scheduler._tick_as_task is None

    result = scheduler.tick_now()
    # Direct tick returns the grid tick result dict, not an A2A envelope
    assert (
        "fired" in result
        or "fired_rooms" in result
        or "ids" in result
        or "tick" in result
    )
    assert scheduler.beat_number == 1


# ── collect_results async ─────────────────────────────────


def test_collect_results_async(scheduler):
    """collect_results() should handle multiple pending tasks in order."""
    task = TickAsTask(scheduler)

    # Simulate submitting 3 tasks without HTTP mocking (synthetic failures)
    for i in range(3):
        payload = task.on_beat(beat_number=i)
        # Force-add pending IDs manually to simulate submitted state
        task._pending_task_ids.append(f"tick-{i}")

    assert len(task._pending_task_ids) == 3

    results = task.collect_results()
    assert len(results) == 3
    assert [r["id"] for r in results] == ["tick-0", "tick-1", "tick-2"]
    assert all(r["status"] == "completed" for r in results)
    assert len(task._pending_task_ids) == 0


# ── Integration: A2ASignalSource + task_mode scheduler ─────


def test_integration_a2a_signal_source_in_task_mode():
    """An A2ASignalSource can feed a task-mode scheduler end-to-end."""
    grid = RoomGrid(50)
    router = RoutingLayer()
    thermal = ThermalBudget()
    breeder = AutoBreeder(grid, thermal, interval=4)

    a2a_signal = A2ASignalSource(endpoint_url="http://signal.agent:9000")
    sched = MetronomeScheduler(
        grid=grid,
        router=router,
        breeder=breeder,
        bpm=240.0,
        signal_source=a2a_signal,
        task_mode=True,
    )

    # Mock both the signal fetch and the task submission
    signal_response = MagicMock()
    signal_response.read.return_value = json.dumps(
        {
            "artefacts": [{"content": {"signal": np.ones(64).tolist()}}],
        }
    ).encode("utf-8")
    signal_response.__enter__ = MagicMock(return_value=signal_response)
    signal_response.__exit__ = MagicMock(return_value=False)

    task_response = MagicMock()
    task_response.read.return_value = json.dumps(
        {
            "id": "tick-0",
            "status": "completed",
            "artefacts": [{"content": {"beat_number": 0, "fired_rooms": []}}],
        }
    ).encode("utf-8")
    task_response.__enter__ = MagicMock(return_value=task_response)
    task_response.__exit__ = MagicMock(return_value=False)

    call_count = {"signal": 0, "task": 0}

    def mock_urlopen(req, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "signal" in url:
            call_count["signal"] += 1
            return signal_response
        call_count["task"] += 1
        return task_response

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = sched.tick_now()

    assert call_count["signal"] == 1
    assert call_count["task"] == 1
    assert result["status"] == "completed"
    assert sched.beat_number == 1
