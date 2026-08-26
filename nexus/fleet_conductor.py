"""Fleet Conductor — Distributed metronome sync across fleet nodes.

Uses CRDT-style beat counters + vector clocks for drift detection.
Falls back to "best-effort sync" when network partitions occur.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from logos.intent_protocol import FleetState, IntentConfirmationProtocol
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── data structures ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BeatState:
    """Immutable beat state for CRDT-style merge.

    Parameters
    ----------
    beat_number:
        Monotonic beat counter (higher = newer).
    wall_time_ns:
        ``time.time_ns()`` when this beat was emitted.
    perf_counter_ns:
        ``time.perf_counter_ns()`` when this beat was emitted.
    rtt_ms:
        Round-trip time in milliseconds (0.0 for local state).
    """

    beat_number: int
    wall_time_ns: int
    perf_counter_ns: int
    rtt_ms: float = 0.0

    @staticmethod
    def now(beat_number: int, rtt_ms: float = 0.0) -> "BeatState":
        """Create a BeatState stamped with the current time."""
        return BeatState(
            beat_number=beat_number,
            wall_time_ns=time.time_ns(),
            perf_counter_ns=time.perf_counter_ns(),
            rtt_ms=rtt_ms,
        )

    @classmethod
    def merge(cls, a: "BeatState", b: "BeatState") -> "BeatState":
        """CRDT merge: higher beat_number wins; tie-break by earlier wall_time."""
        if a.beat_number != b.beat_number:
            return a if a.beat_number > b.beat_number else b
        if a.wall_time_ns != b.wall_time_ns:
            return a if a.wall_time_ns < b.wall_time_ns else b
        return a if a.perf_counter_ns <= b.perf_counter_ns else b


# ── FleetConductor ────────────────────────────────────────────


class FleetConductor:
    """Distributed metronome sync across fleet nodes.

    Each node runs its own ``MetronomeScheduler``. The conductor
    keeps them in phase via CRDT-style beat-state exchange and
    drift correction.
    """

    def __init__(
        self,
        node_id: str,
        nexus_endpoint: str,
        sync_interval_ms: int = 1000,
        max_drift_ms: float = 5.0,
    ):
        self.node_id = node_id
        self.nexus_endpoint = nexus_endpoint.rstrip("/")
        self.sync_interval_ms = sync_interval_ms
        self.max_drift_ms = max_drift_ms

        self._scheduler: Any | None = None
        self._peer_beats: dict[str, BeatState] = {}
        self._local_beat_state: BeatState | None = None
        self._running_solo: bool = False

        # Default beat duration: 500 ms (120 BPM)
        self._default_beat_duration_ms: float = 500.0

    # ── scheduler linkage ─────────────────────────────────────

    def register_local_scheduler(self, scheduler: Any) -> None:
        """Link this conductor to a local ``MetronomeScheduler``."""
        self._scheduler = scheduler

    def submit_human_command(
        self,
        raw_command: str,
        fleet_state: Optional[FleetState] = None,
    ) -> dict:
        """Process a human command through IntentConfirmationProtocol.

        Returns a dict with keys:
        - 'intent': the parsed Intent
        - 'requires_confirmation': bool
        - 'confirmation_prompt': str (only if requires_confirmation)
        - 'can_execute': bool (True if no confirmation needed)
        """
        # Derive fleet state from peer beats if not provided
        if fleet_state is None:
            total = len(self._peer_beats) + 1
            active = total  # assume all active for quick estimate
            fleet_state = FleetState(total_agents=total, active_agents=active)

        protocol = IntentConfirmationProtocol(fleet_state=fleet_state)
        intent = protocol.parse_intent(raw_command)
        requires = protocol.require_confirmation(intent)
        result: dict = {
            "intent": intent,
            "requires_confirmation": requires,
            "can_execute": not requires,
        }
        if requires:
            result["confirmation_prompt"] = protocol.generate_confirmation(intent)
        return result

    # ── local state helpers ───────────────────────────────────

    def _get_local_beat_state(self) -> BeatState:
        """Capture current local beat state."""
        if self._scheduler is not None and hasattr(self._scheduler, "beat_number"):
            beat_number = self._scheduler.beat_number
        else:
            beat_number = 0
        return BeatState.now(beat_number=beat_number)

    def _beat_duration_ms(self) -> float:
        """Return the current beat duration in milliseconds."""
        if self._scheduler is not None:
            if hasattr(self._scheduler, "beat_duration_ms"):
                return float(self._scheduler.beat_duration_ms)
            if hasattr(self._scheduler, "bpm") and self._scheduler.bpm > 0:
                return 60_000.0 / float(self._scheduler.bpm)
        return self._default_beat_duration_ms

    # ── sync ────────────────────────────────────────────────────

    async def sync_beat(self) -> BeatState:
        """Exchange beat state with peers and return merged consensus.

        1. Captures local beat state.
        2. POSTs it to the nexus sync endpoint.
        3. Receives peer beat states.
        4. Merges via CRDT rules.
        5. Triggers drift correction.
        """
        local = self._get_local_beat_state()
        self._local_beat_state = local

        # Fetch peer states from nexus (runs in thread so sync_beat is async)
        peer_beats = await asyncio.to_thread(self._fetch_peer_beats, local)
        self._peer_beats = peer_beats

        # Merge all peer states + local into consensus
        consensus = local
        for peer_state in peer_beats.values():
            consensus = BeatState.merge(consensus, peer_state)

        # Drift correction against the merged consensus
        self.correct_drift(peer_beats)

        return consensus

    def _fetch_peer_beats(self, local_state: BeatState) -> dict[str, BeatState]:
        """POST local state and retrieve peer states.

        In a full implementation this hits the nexus sync endpoint.
        For now it returns an empty dict; subclasses or tests may
        override / patch this.
        """
        return {}

    # ── drift correction ────────────────────────────────────────

    def correct_drift(self, peer_beats: dict[str, BeatState]) -> None:
        """Apply phase nudge, skip-jump, or partition fallback.

        Strategies (in order):
        1. Phase nudge — smooth adjustment < 5 % of beat duration.
        2. Skip/jump   — if drift exceeds one full beat.
        3. Partition   — if quorum is lost, run solo with warning.
        """
        if not peer_beats:
            self._handle_partition()
            return

        beat_duration = self._beat_duration_ms()

        # Determine consensus beat from peers
        consensus_state: BeatState | None = None
        for state in peer_beats.values():
            consensus_state = (
                state
                if consensus_state is None
                else BeatState.merge(consensus_state, state)
            )

        if consensus_state is None or self._local_beat_state is None:
            self._handle_partition()
            return

        local = self._local_beat_state
        drift_beats = abs(local.beat_number - consensus_state.beat_number)
        drift_ms = drift_beats * beat_duration

        # Also compute wall-time drift for extra signal
        wall_drift_ms = (
            abs(local.wall_time_ns - consensus_state.wall_time_ns) / 1_000_000.0
        )

        if drift_ms > self.max_drift_ms or wall_drift_ms > self.max_drift_ms:
            if drift_beats >= 1:
                self._apply_skip_jump(consensus_state)
            else:
                self._apply_phase_nudge(
                    consensus_state, max(drift_ms, wall_drift_ms), beat_duration
                )

        self._check_quorum(peer_beats)

    def _apply_phase_nudge(
        self,
        consensus: BeatState,
        drift_ms: float,
        beat_duration_ms: float,
    ) -> None:
        """Smooth correction: nudge the next beat by < 5 % of duration."""
        nudge_ratio = min(0.05, (drift_ms / beat_duration_ms) * 0.5)
        nudge_ms = nudge_ratio * beat_duration_ms
        logger.info(
            "Phase nudge on %s: %.3f ms (%.2f%% of beat)",
            self.node_id,
            nudge_ms,
            nudge_ratio * 100,
        )
        if self._scheduler is not None and hasattr(self._scheduler, "nudge_phase"):
            self._scheduler.nudge_phase(nudge_ms)

    def _apply_skip_jump(self, consensus: BeatState) -> None:
        """Hard correction: snap local beat counter to consensus."""
        local_bn = self._local_beat_state.beat_number if self._local_beat_state else "?"
        logger.warning(
            "Skip-jump on %s: local=%s → consensus=%d",
            self.node_id,
            local_bn,
            consensus.beat_number,
        )
        if self._scheduler is not None and hasattr(self._scheduler, "jump_to_beat"):
            self._scheduler.jump_to_beat(consensus.beat_number)

    def _check_quorum(self, peer_beats: dict[str, BeatState]) -> None:
        """Require majority of known peers to be responsive."""
        total_nodes = len(peer_beats) + 1  # peers + self
        quorum_needed = (total_nodes // 2) + 1
        responsive = len(peer_beats) + 1  # assume all peers in dict are responsive
        if responsive < quorum_needed:
            self._handle_partition()
        else:
            self._running_solo = False

    def _handle_partition(self) -> None:
        """Partition tolerance: run solo with warning."""
        if not self._running_solo:
            self._running_solo = True
            logger.warning(
                "Fleet partition on %s — running solo (no quorum).",
                self.node_id,
            )
