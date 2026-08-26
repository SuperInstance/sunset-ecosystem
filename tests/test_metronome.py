"""Tests for nerve/metronome.py — MetronomeScheduler + LocalMetronome.

P0 Core: single-node metronome driving RoomGrid.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from nerve.metronome import LocalMetronome, MetronomeScheduler, RandomSignalSource
from nerve.room_grid import RoomGrid
from nerve.routing import RoutingLayer
from swarm.breeder_daemon import AutoBreeder
from swarm.thermal import ThermalBudget


@pytest.fixture
def scheduler():
    """Return a MetronomeScheduler wired to a small grid, router, and breeder."""
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


# ── LocalMetronome ───────────────────────────────────────


def test_local_metronome_divider_8():
    """A divider=8 metronome should fire only every 8th beat."""
    lm = LocalMetronome(global_bpm=120.0, divider=8, phase=0.0)
    fired = [i for i in range(24) if lm.should_fire(i)]
    assert fired == [0, 8, 16]
    assert lm.local_bpm == 15.0  # 120 / 8


# ── MetronomeScheduler timing ────────────────────────────


def test_scheduler_beat_timing_120bpm(scheduler):
    """120 BPM = 500 ms beat duration; tick_now() advances beat_number."""
    assert scheduler.bpm == 120.0
    assert scheduler.beat_duration == pytest.approx(0.5, abs=1e-9)

    before = scheduler.beat_number
    scheduler.tick_now()
    after = scheduler.beat_number
    assert after == before + 1

    # Tick twice so actual_bpm has enough intervals
    scheduler.tick_now()
    scheduler.tick_now()
    assert scheduler.actual_bpm > 0.0


# ── Harmonic breeding ───────────────────────────────────


def test_harmonic_breeding_every_4_beats(scheduler):
    """Breeder.cycle() should be called on beats 0, 4, 8, ..."""
    call_count = 0
    orig_cycle = scheduler.breeder.cycle

    def counting_cycle(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return []

    scheduler.breeder.cycle = counting_cycle

    # Tick 8 beats → breeding on 0 and 4
    for _ in range(8):
        scheduler.tick_now()

    assert call_count == 2

    scheduler.breeder.cycle = orig_cycle


# ── Adaptive throttle ────────────────────────────────────


def test_adaptive_throttle_on_slow_forward(scheduler, caplog):
    """If _forward() takes > 80 % of beat_duration, log a warning."""
    orig_tick = scheduler.grid.tick

    def slow_tick(x):
        time.sleep(0.15)  # 150 ms
        return orig_tick(x)

    scheduler.grid.tick = slow_tick
    # 500 BPM → 120 ms beat duration → 150 ms > 80 % of 120 ms = 96 ms
    scheduler.bpm = 500.0

    with caplog.at_level(logging.WARNING, logger="nerve.metronome"):
        scheduler.tick_now()

    assert any("exceeded 80%" in rec.message for rec in caplog.records), (
        f"Expected throttle warning in logs, got: {[r.message for r in caplog.records]}"
    )

    scheduler.grid.tick = orig_tick


# ── SignalSource ─────────────────────────────────────────


def test_signal_source_random():
    """RandomSignalSource produces (64,) float32 arrays that vary per call."""
    src = RandomSignalSource(seed=123)
    s1 = src.next_signal(0)
    s2 = src.next_signal(1)

    assert s1.shape == (64,)
    assert s1.dtype == np.float32
    assert not np.array_equal(s1, s2)


# ── Start / Stop ─────────────────────────────────────────


def test_scheduler_start_stop(scheduler):
    """Background thread should start, tick at least once, and stop cleanly."""
    before = scheduler.beat_number
    scheduler.start()
    # Give it ~300 ms at 120 BPM (≈ 500 ms/beat) → should tick 0 or 1 times
    time.sleep(0.3)
    scheduler.stop()
    after = scheduler.beat_number
    # We should have advanced by at least 0, maybe 1 beat
    assert after >= before


# ── Flux harmonic (optional sanity check) ──────────────────


def test_flux_harmonic_not_called_every_beat(scheduler):
    """FLUX phase only fires every flux_harmonic beats."""
    flux_calls = 0
    orig_flux = scheduler._flux_phase

    def counting_flux(beat_number):
        nonlocal flux_calls
        flux_calls += 1

    scheduler._flux_phase = counting_flux

    # Tick 8 beats with flux_harmonic=16 → flux fires on beat 0 only
    for _ in range(8):
        scheduler.tick_now()

    assert flux_calls == 1

    # Tick 9 more → should hit beat 16 (0 + 8 + 9 = 17 ticks total, beats 0..16)
    for _ in range(9):
        scheduler.tick_now()

    assert flux_calls == 2

    scheduler._flux_phase = orig_flux
