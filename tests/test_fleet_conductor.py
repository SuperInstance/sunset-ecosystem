"""Tests for FleetConductor distributed metronome sync."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nexus.fleet_conductor import BeatState, FleetConductor


# ═══════════════════════════════════════════════════════════════
# Mock scheduler
# ═══════════════════════════════════════════════════════════════

class MockScheduler:
    """Minimal scheduler stand-in for conductor tests."""

    def __init__(self, beat_number: int = 0, bpm: float = 120.0):
        self.beat_number = beat_number
        self.bpm = bpm
        self.last_nudge_ms: float | None = None
        self.last_jump_beat: int | None = None

    def nudge_phase(self, nudge_ms: float) -> None:
        self.last_nudge_ms = nudge_ms

    def jump_to_beat(self, beat_number: int) -> None:
        self.last_jump_beat = beat_number


# ═══════════════════════════════════════════════════════════════
# BeatState CRDT merge
# ═══════════════════════════════════════════════════════════════

class TestBeatStateMergeCRDT:
    """BeatState.merge must satisfy CRDT semantics."""

    def test_higher_beat_number_wins(self):
        old = BeatState(beat_number=10, wall_time_ns=100, perf_counter_ns=100)
        new = BeatState(beat_number=11, wall_time_ns=50, perf_counter_ns=50)
        assert BeatState.merge(old, new) == new

    def test_tiebreak_earlier_wall_time_wins(self):
        a = BeatState(beat_number=10, wall_time_ns=100, perf_counter_ns=200)
        b = BeatState(beat_number=10, wall_time_ns=50, perf_counter_ns=300)
        # Earlier wall_time wins (emitted first)
        assert BeatState.merge(a, b) == b

    def test_perf_counter_tiebreak(self):
        a = BeatState(beat_number=10, wall_time_ns=100, perf_counter_ns=200)
        b = BeatState(beat_number=10, wall_time_ns=100, perf_counter_ns=300)
        assert BeatState.merge(a, b) == a

    def test_merge_is_commutative(self):
        x = BeatState(beat_number=5, wall_time_ns=10, perf_counter_ns=20)
        y = BeatState(beat_number=7, wall_time_ns=5, perf_counter_ns=15)
        assert BeatState.merge(x, y) == BeatState.merge(y, x)

    def test_merge_is_idempotent(self):
        s = BeatState(beat_number=3, wall_time_ns=1, perf_counter_ns=2)
        assert BeatState.merge(s, s) == s


# ═══════════════════════════════════════════════════════════════
# Drift detection
# ═══════════════════════════════════════════════════════════════

class TestDriftDetection:
    def test_drift_detection_exceeds_threshold(self):
        """Drift > max_drift_ms with >=1 beat delta must trigger skip-jump."""
        conductor = FleetConductor(
            "node-a", "http://nexus.test:4047", max_drift_ms=5.0
        )
        scheduler = MockScheduler(beat_number=100, bpm=120)  # 500 ms/beat
        conductor.register_local_scheduler(scheduler)

        conductor._local_beat_state = BeatState(
            beat_number=100, wall_time_ns=0, perf_counter_ns=0
        )
        # Peer 3 beats ahead → 1500 ms drift > 5 ms threshold
        peer = BeatState(beat_number=103, wall_time_ns=0, perf_counter_ns=0)

        with patch.object(conductor, "_apply_skip_jump") as mock_jump:
            conductor.correct_drift({"node-b": peer})
            mock_jump.assert_called_once()

    def test_drift_within_threshold_no_correction(self):
        """Zero drift must not trigger any correction."""
        conductor = FleetConductor(
            "node-a", "http://nexus.test:4047", max_drift_ms=5.0
        )
        scheduler = MockScheduler(beat_number=100, bpm=120)
        conductor.register_local_scheduler(scheduler)

        conductor._local_beat_state = BeatState(
            beat_number=100, wall_time_ns=0, perf_counter_ns=0
        )
        peer = BeatState(beat_number=100, wall_time_ns=0, perf_counter_ns=0)

        with patch.object(conductor, "_apply_phase_nudge") as mock_nudge, \
             patch.object(conductor, "_apply_skip_jump") as mock_jump:
            conductor.correct_drift({"node-b": peer})
            mock_nudge.assert_not_called()
            mock_jump.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# Phase nudge
# ═══════════════════════════════════════════════════════════════

class TestPhaseNudge:
    def test_phase_nudge_correction(self):
        """Sub-beat drift > max_drift_ms must trigger phase nudge (not skip-jump)."""
        conductor = FleetConductor(
            "node-a", "http://nexus.test:4047", max_drift_ms=5.0
        )
        scheduler = MockScheduler(beat_number=100, bpm=120)
        conductor.register_local_scheduler(scheduler)

        conductor._local_beat_state = BeatState(
            beat_number=100, wall_time_ns=10_000_000, perf_counter_ns=0
        )
        # Same beat number, 10 ms wall-time delta → 10 ms drift > 5 ms,
        # but 0 beat delta → phase nudge, not skip-jump
        peer = BeatState(beat_number=100, wall_time_ns=0, perf_counter_ns=0)

        with patch.object(conductor, "_apply_skip_jump") as mock_jump, \
             patch.object(conductor, "_apply_phase_nudge") as mock_nudge:
            conductor.correct_drift({"node-b": peer})
            mock_jump.assert_not_called()
            mock_nudge.assert_called_once()

    def test_nudge_ratio_capped_at_five_percent(self):
        """Phase nudge must never exceed 5 % of beat duration."""
        conductor = FleetConductor(
            "node-a", "http://nexus.test:4047", max_drift_ms=5.0
        )
        scheduler = MockScheduler(beat_number=100, bpm=120)  # 500 ms/beat
        conductor.register_local_scheduler(scheduler)

        conductor._local_beat_state = BeatState(
            beat_number=100, wall_time_ns=400_000_000, perf_counter_ns=0
        )
        # 400 ms wall drift > 5 ms, same beat → phase nudge
        peer = BeatState(beat_number=100, wall_time_ns=0, perf_counter_ns=0)

        conductor.correct_drift({"node-b": peer})
        # nudge_ratio = min(0.05, (400/500)*0.5) = min(0.05, 0.4) = 0.05
        assert scheduler.last_nudge_ms == pytest.approx(0.05 * 500.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# Partition fallback
# ═══════════════════════════════════════════════════════════════

class TestPartitionFallback:
    def test_partition_fallback_no_quorum(self):
        """Empty peer dict must flag solo mode."""
        conductor = FleetConductor("node-a", "http://nexus.test:4047")
        scheduler = MockScheduler(beat_number=100)
        conductor.register_local_scheduler(scheduler)

        conductor._local_beat_state = BeatState(
            beat_number=100, wall_time_ns=0, perf_counter_ns=0
        )
        conductor.correct_drift({})
        assert conductor._running_solo is True

    def test_partition_recovered_when_quorum_returns(self):
        """Receiving peers again must clear solo flag."""
        conductor = FleetConductor("node-a", "http://nexus.test:4047")
        scheduler = MockScheduler(beat_number=100)
        conductor.register_local_scheduler(scheduler)

        conductor._local_beat_state = BeatState(
            beat_number=100, wall_time_ns=0, perf_counter_ns=0
        )
        # First: no peers → partition
        conductor.correct_drift({})
        assert conductor._running_solo is True

        # Then: peer appears → quorum met (2 nodes, need 2)
        peer = BeatState(beat_number=100, wall_time_ns=0, perf_counter_ns=0)
        conductor.correct_drift({"node-b": peer})
        assert conductor._running_solo is False


# ═══════════════════════════════════════════════════════════════
# Async sync_beat
# ═══════════════════════════════════════════════════════════════

class TestSyncBeat:
    @pytest.mark.asyncio
    async def test_sync_beat_returns_consensus(self):
        """sync_beat must return a BeatState (local when no peers)."""
        conductor = FleetConductor("node-a", "http://nexus.test:4047")
        scheduler = MockScheduler(beat_number=42)
        conductor.register_local_scheduler(scheduler)

        consensus = await conductor.sync_beat()
        assert isinstance(consensus, BeatState)
        assert consensus.beat_number == 42

    @pytest.mark.asyncio
    async def test_sync_beat_merges_peers(self):
        """sync_beat must merge peer states into consensus."""
        conductor = FleetConductor("node-a", "http://nexus.test:4047")
        scheduler = MockScheduler(beat_number=10)
        conductor.register_local_scheduler(scheduler)

        peer = BeatState(beat_number=20, wall_time_ns=0, perf_counter_ns=0)

        with patch.object(conductor, "_fetch_peer_beats", return_value={"node-b": peer}):
            consensus = await conductor.sync_beat()

        # Higher beat number wins
        assert consensus.beat_number == 20
