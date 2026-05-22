"""Tests for nerve/metronome_bridge.py — MetronomeBridge selective dispatch.

Bridge maps metronome beats to RoomGrid ticks:
  Beat 0: full grid tick
  Beat 1: thermal-critical rooms
  Beat 2: breeding-ready rooms
  Beat 3: perception-active rooms
"""

from __future__ import annotations

import numpy as np
import pytest

from nerve.metronome import MetronomeScheduler, RandomSignalSource
from nerve.metronome_bridge import MetronomeBridge
from nerve.room_grid import RoomGrid
from nerve.routing import RoutingLayer
from swarm.breeder_daemon import AutoBreeder
from swarm.thermal import ThermalBudget


@pytest.fixture
def bridge():
    """Return a MetronomeBridge wired to a small grid + scheduler."""
    np.random.seed(42)
    grid = RoomGrid(100)
    router = RoutingLayer()
    for i in range(5):
        router.add_route("grid", f"dst_{i}")
    thermal = ThermalBudget()
    breeder = AutoBreeder(grid, thermal, interval=4)
    scheduler = MetronomeScheduler(
        grid=grid,
        router=router,
        breeder=breeder,
        bpm=120.0,
        signal_source=RandomSignalSource(seed=42),
    )
    return scheduler._bridge


# ── Beat 0: full grid tick ───────────────────────────────


def test_beat_zero_dispatches_all_rooms(bridge):
    """Beat 0 must dispatch every room in the grid."""
    grid = bridge.grid
    n = grid.n

    # Pre-seed some chaos/activity so selective beats have something to pick
    grid.chaos[:20] = np.linspace(0.1, 0.9, 20)
    grid.activity[10:30] = np.arange(20)

    dispatched = bridge.on_metronome_beat(beat_number=0, tempo_ms=500.0)

    assert len(dispatched) == n
    assert set(dispatched) == set(range(n))
    # After a full tick, latents should be populated for all rooms
    assert grid.latents.shape == (n, 16)
    assert not np.all(grid.latents == 0)


# ── Beat 1: thermal-critical rooms ─────────────────────


def test_beat_one_dispatches_thermal_critical_only(bridge):
    """Beat 1 should only touch rooms with high chaos values."""
    grid = bridge.grid

    # Set up a clear chaos gradient
    grid.chaos[:] = 0.01  # baseline
    grid.chaos[5] = 0.99
    grid.chaos[7] = 0.95
    grid.chaos[12] = 0.88
    grid.chaos[80] = 0.50

    dispatched = bridge.on_metronome_beat(beat_number=1, tempo_ms=500.0)

    # Should NOT dispatch all rooms
    assert len(dispatched) < grid.n
    # Every dispatched room must have chaos above the threshold
    for rid in dispatched:
        assert grid.chaos[rid] > 0.15, f"room {rid} chaos={grid.chaos[rid]:.3f}"

    # The highest-chaos rooms should be included
    assert 5 in dispatched
    assert 7 in dispatched


# ── Beat cycle: different rooms on each beat ─────────────


def test_beat_cycle_dispatches_different_rooms_each_beat(bridge):
    """A full 4-beat cycle should dispatch different subsets."""
    grid = bridge.grid

    # Seed diverse chaos and activity profiles
    grid.chaos[:] = np.random.rand(grid.n) * 0.4
    grid.activity[:] = np.random.randint(0, 50, size=grid.n)

    beats = []
    for beat in range(4):
        dispatched = bridge.on_metronome_beat(beat_number=beat, tempo_ms=500.0)
        beats.append(set(dispatched))

    # Beat 0 is the full grid; beats 1-3 are subsets
    assert beats[0] == set(range(grid.n))

    # Beats 1, 2, 3 should be proper subsets (or at least not identical)
    assert len(beats[1]) < grid.n
    assert len(beats[2]) < grid.n
    assert len(beats[3]) < grid.n

    # They may overlap, but they should not all be identical
    assert not (beats[1] == beats[2] == beats[3])


# ── Device dispatch: GPU latency tracking ────────────────


def test_dispatch_room_tracks_gpu_latency(bridge):
    """dispatch_room(device='gpu') should record a GPU latency entry."""
    grid = bridge.grid

    # Provide a signal so dispatch_room has something to compute
    bridge._last_signal = np.random.randn(64).astype(np.float32)

    before = len(bridge._latencies["gpu"])
    bridge.dispatch_room(room_id=3, device="gpu")
    after = len(bridge._latencies["gpu"])

    assert after == before + 1
    report = bridge.get_latency_report()
    assert report["gpu"]["count"] >= 1
    assert report["gpu"]["mean_ms"] > 0.0


# ── sync_devices: no pending ops ─────────────────────────


def test_sync_devices_clears_pending_ops(bridge):
    """After sync_devices(), pending_ops must be zero."""
    grid = bridge.grid
    bridge._last_signal = np.random.randn(64).astype(np.float32)

    # Dispatch a few rooms to create pending ops
    bridge.dispatch_room(0, "cpu")
    bridge.dispatch_room(1, "gpu")
    bridge.dispatch_room(2, "rust")

    assert bridge._pending_ops > 0

    bridge.sync_devices()

    assert bridge._pending_ops == 0
    report = bridge.get_latency_report()
    for device in ("cpu", "gpu", "rust"):
        assert report[device]["count"] >= 1


# ── Integration: scheduler compute_phase via bridge ─────


def test_scheduler_uses_bridge_for_compute_phase():
    """MetronomeScheduler._compute_phase delegates to the bridge."""
    np.random.seed(42)
    grid = RoomGrid(50)
    router = RoutingLayer()
    breeder = AutoBreeder(grid, ThermalBudget(), interval=4)
    scheduler = MetronomeScheduler(
        grid=grid,
        router=router,
        breeder=breeder,
        bpm=120.0,
        signal_source=RandomSignalSource(seed=42),
    )

    assert scheduler._bridge is not None
    assert isinstance(scheduler._bridge, MetronomeBridge)

    # Tick beat 0 → full grid via bridge
    scheduler.tick_now()
    assert scheduler.beat_number == 1
    result = scheduler._last_tick_result
    assert result is not None
    assert result["fired"] == 50  # all rooms dispatched

    # Tick beat 1 → selective dispatch
    scheduler.tick_now()
    result = scheduler._last_tick_result
    assert result["fired"] < 50  # subset dispatched
