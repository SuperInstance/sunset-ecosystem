"""Distributed Metronome Bridge — Cross-node beat synchronization for the Cocapn Fleet.

Enables multiple fleet nodes to maintain a unified metronome beat via
A2A-signed sync messages and PID-style drift correction.

Key classes
-----------
* ``UnifiedBeat``          — snapshot of global + local beat state
* ``DriftCorrection``        — PID controller for BPM adjustment
* ``MetronomeBridge``        — orchestrates ticks, sync, drift compute, correction

Integration
-----------
``FleetConductor`` instantiates ``MetronomeBridge`` on startup.
``MeshVectorGossip`` carries sync payloads of type ``"metronome_sync"``.
``AgentIdentity.sign_task()`` signs each sync message.
``OperationalTrap`` monitors drift and alerts when > 50 ms.
"""

from __future__ import annotations

__all__ = [
    "UnifiedBeat",
    "DriftCorrection",
    "MetronomeBridge",
    "SyncMessage",
    "MetronomeBridgeError",
]

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Exceptions ──────────────────────────────────────────────

class MetronomeBridgeError(Exception):
    """Raised when the bridge encounters an unrecoverable state."""


class SignatureValidationError(MetronomeBridgeError):
    """Raised when a sync message signature fails verification."""


# ── Data structures ─────────────────────────────────────────

@dataclass(frozen=True)
class UnifiedBeat:
    """Snapshot of beat alignment across the fleet.

    Attributes
    ----------
    global_beat_count:
        Consensus beat number (max across all known peers + local).
    local_beat_count:
        This node's own monotonic beat counter.
    drift_ms:
        Maximum absolute drift observed across all peers (ms).
    peer_count:
        Number of peers we have timestamps for.
    timestamp:
        Wall-clock time when this snapshot was taken (ns).
    """

    global_beat_count: int
    local_beat_count: int
    drift_ms: float
    peer_count: int
    timestamp: int = field(default_factory=time.time_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_beat_count": self.global_beat_count,
            "local_beat_count": self.local_beat_count,
            "drift_ms": round(self.drift_ms, 3),
            "peer_count": self.peer_count,
            "timestamp": self.timestamp,
        }


@dataclass
class SyncMessage:
    """Payload for a single metronome sync exchange.

    Fields are deliberately plain JSON-serialisable types so that
    ``AgentIdentity.sign_task`` can canonicalise them deterministically.
    """

    node_id: str
    beat_count: int
    timestamp_ns: int
    bpm: float
    signature: str = ""

    def payload_for_signing(self) -> dict[str, Any]:
        """Return the dict that is signed / verified."""
        return {
            "node_id": self.node_id,
            "beat_count": self.beat_count,
            "timestamp_ns": self.timestamp_ns,
            "bpm": self.bpm,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload_for_signing(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyncMessage":
        return cls(
            node_id=data["node_id"],
            beat_count=data["beat_count"],
            timestamp_ns=data["timestamp_ns"],
            bpm=data["bpm"],
            signature=data.get("signature", ""),
        )


# ── DriftCorrection (PID) ───────────────────────────────────

@dataclass
class DriftCorrection:
    """PID-style BPM corrector.

    Parameters
    ----------
    threshold_ms:
        Drift above this value triggers correction (default 10 ms).
    kp, ki, kd:
        Proportional, integral, derivative gains.
    min_factor, max_factor:
        Clamp for the correction multiplier (default 0.95–1.05).
    integral_window_ms:
        Maximum accumulated integral error before wind-up clamp.
    """

    threshold_ms: float = 10.0
    kp: float = 0.01
    ki: float = 0.001
    kd: float = 0.005
    min_factor: float = 0.95
    max_factor: float = 1.05
    integral_window_ms: float = 500.0

    # Mutable state
    _integral: float = field(default=0.0, repr=False)
    _last_drift: float = field(default=0.0, repr=False)
    _last_time: float = field(default_factory=time.perf_counter, repr=False)

    def should_correct(self, drift_ms: float) -> bool:
        """True if ``abs(drift_ms) > threshold_ms``."""
        return abs(drift_ms) > self.threshold_ms

    def correction_factor(self, drift_ms: float) -> float:
        """Return a BPM multiplier in ``[min_factor, max_factor]``.

        Uses a discrete PID:
            P = kp * drift_ms
            I = ki * integral
            D = kd * (drift_ms - last_drift) / dt
            factor = 1.0 + (P + I + D)
        """
        now = time.perf_counter()
        dt = now - self._last_time
        if dt <= 0:
            dt = 1.0  # safety guard on first call or clock glitch

        # Proportional
        p_term = self.kp * drift_ms

        # Integral (with wind-up clamp)
        self._integral += drift_ms * dt
        self._integral = max(
            -self.integral_window_ms,
            min(self.integral_window_ms, self._integral),
        )
        i_term = self.ki * self._integral

        # Derivative
        d_term = self.kd * (drift_ms - self._last_drift) / dt

        # Update state
        self._last_drift = drift_ms
        self._last_time = now

        factor = 1.0 + p_term + i_term + d_term
        return max(self.min_factor, min(self.max_factor, factor))

    def reset(self) -> None:
        """Clear accumulated PID state."""
        self._integral = 0.0
        self._last_drift = 0.0
        self._last_time = time.perf_counter()


# ── MetronomeBridge ─────────────────────────────────────────

class MetronomeBridge:
    """Cross-node metronome synchronisation engine.

    Each node maintains its own local beat.  Periodically it broadcasts
    a signed ``SyncMessage`` to every peer.  Peers validate the signature,
    record the timestamp, and the bridge computes drift.  If drift exceeds
    the threshold the local BPM is nudged via ``DriftCorrection``.

    Thread-safe: all public methods acquire an internal lock.
    """

    def __init__(
        self,
        local_bpm: float,
        node_id: str,
        peers: List[str],
        identity: Optional[Any] = None,
        drift_threshold_ms: float = 10.0,
        max_bpm_delta_ratio: float = 0.05,
        send_fn: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ):
        """Initialise the bridge.

        Parameters
        ----------
        local_bpm:
            Starting beats-per-minute for this node.
        node_id:
            Unique identifier for this node.
        peers:
            List of peer node IDs we synchronise with.
        identity:
            An ``AgentIdentity``-like object with ``sign_task()`` and
            ``verify_task()`` methods.  If *None*, signatures are skipped
            (testing / no-crypto mode).
        drift_threshold_ms:
            Passed through to ``DriftCorrection``.
        max_bpm_delta_ratio:
            Maximum allowed BPM change per adjustment (default 5%).
        send_fn:
            Optional callback ``send_fn(peer_id, payload_dict)`` used by
            ``sync_with_peers()`` to dispatch sync messages.
        """
        self.node_id = node_id
        self.peers = list(peers)
        self._bpm = float(local_bpm)
        self._identity = identity
        self._max_bpm_delta_ratio = max_bpm_delta_ratio
        self._send_fn = send_fn

        # Beat state
        self._beat_count = 0
        self._last_tick_ns: int = time.time_ns()

        # Peer state: node_id → (timestamp_ns, beat_count, bpm, received_at_ns)
        self._peer_state: dict[str, Tuple[int, int, float, int]] = {}

        # Adjustment history: list of (timestamp_ns, old_bpm, new_bpm, reason)
        self._adjustment_history: list[dict[str, Any]] = []

        # Drift corrector
        self._drift_correction = DriftCorrection(threshold_ms=drift_threshold_ms)

        # Thread safety
        self._lock = threading.RLock()

    # ── Beat ticking ────────────────────────────────────────

    def tick(self) -> int:
        """Increment the local beat counter with microsecond precision.

        Returns the new local beat count.
        """
        with self._lock:
            self._beat_count += 1
            self._last_tick_ns = time.time_ns()
        return self._beat_count

    # ── Sync ────────────────────────────────────────────────

    def sync_with_peers(self) -> List[SyncMessage]:
        """Build signed sync messages and dispatch them to all peers.

        If ``send_fn`` was provided at construction, each message is
        delivered via that callback.  Either way the list of constructed
        ``SyncMessage`` objects is returned for inspection / testing.
        """
        with self._lock:
            messages: List[SyncMessage] = []
            payload = SyncMessage(
                node_id=self.node_id,
                beat_count=self._beat_count,
                timestamp_ns=self._last_tick_ns,
                bpm=self._bpm,
            )

            # Sign
            if self._identity is not None and hasattr(self._identity, "sign_task"):
                payload.signature = self._identity.sign_task(payload.payload_for_signing())

            for peer_id in self.peers:
                msg = SyncMessage(
                    node_id=payload.node_id,
                    beat_count=payload.beat_count,
                    timestamp_ns=payload.timestamp_ns,
                    bpm=payload.bpm,
                    signature=payload.signature,
                )
                messages.append(msg)

                if self._send_fn is not None:
                    try:
                        self._send_fn(peer_id, msg.to_dict())
                    except Exception as exc:
                        logger.warning("Sync send to %s failed: %s", peer_id, exc)

            return messages

    def receive_sync(
        self,
        node_id: str,
        timestamp: int,
        signature: str,
        beat_count: Optional[int] = None,
        bpm: Optional[float] = None,
        public_key_pem: Optional[str] = None,
    ) -> bool:
        """Validate and record a sync message from a peer.

        Parameters
        ----------
        node_id:
            Identifier of the peer that sent the sync.
        timestamp:
            The ``timestamp_ns`` field from the sync message.
        signature:
            Base64-encoded signature to verify.
        beat_count:
            Optional peer beat count (defaults to 0 if omitted).
        bpm:
            Optional peer BPM (defaults to local BPM if omitted).
        public_key_pem:
            If provided, verify against this key rather than the local
            identity's public key.

        Returns
        -------
        True if the signature was valid and the state was recorded,
        False otherwise.
        """
        payload_dict = {
            "node_id": node_id,
            "beat_count": beat_count if beat_count is not None else 0,
            "timestamp_ns": timestamp,
            "bpm": bpm if bpm is not None else self._bpm,
        }

        # Validate signature
        if self._identity is not None and hasattr(self._identity, "verify_task"):
            valid = self._identity.verify_task(payload_dict, signature, public_key_pem)
            if not valid:
                logger.warning("Sync signature invalid from %s", node_id)
                return False
        elif signature:
            # We have a signature but no identity to verify it — reject
            logger.warning("Cannot verify sync signature from %s (no identity)", node_id)
            return False
        # If both identity and signature are absent, we accept (no-crypto mode)

        with self._lock:
            self._peer_state[node_id] = (
                timestamp,
                payload_dict["beat_count"],
                payload_dict["bpm"],
                time.time_ns(),
            )

        return True

    # ── Drift computation ───────────────────────────────────

    def compute_drift(self) -> float:
        """Calculate the maximum absolute drift across all known peers (ms).

        Drift is measured as the absolute difference between the local
        ``last_tick_ns`` and each peer's recorded ``timestamp_ns``,
        converted to milliseconds.

        Returns 0.0 when no peers have been heard from.
        """
        with self._lock:
            if not self._peer_state:
                return 0.0

            local_ts = self._last_tick_ns
            max_drift = 0.0
            for peer_id, (peer_ts, _bc, _bpm, _received) in self._peer_state.items():
                drift_ns = abs(local_ts - peer_ts)
                drift_ms = drift_ns / 1_000_000.0
                if drift_ms > max_drift:
                    max_drift = drift_ms

            return max_drift

    # ── BPM adjustment ──────────────────────────────────────

    def adjust_bpm(self, factor: float) -> float:
        """Nudge local BPM by *factor* (multiplicative), clamped to ±5%.

        The actual clamp uses ``max_bpm_delta_ratio`` passed at
        construction, so the new BPM is always within
        ``[old * (1 - ratio), old * (1 + ratio)]``.

        Returns the new BPM.
        """
        with self._lock:
            old_bpm = self._bpm
            min_bpm = old_bpm * (1.0 - self._max_bpm_delta_ratio)
            max_bpm = old_bpm * (1.0 + self._max_bpm_delta_ratio)
            new_bpm = max(min_bpm, min(max_bpm, old_bpm * factor))
            self._bpm = new_bpm

            self._adjustment_history.append({
                "timestamp_ns": time.time_ns(),
                "old_bpm": old_bpm,
                "new_bpm": new_bpm,
                "factor": factor,
                "reason": "drift_correction",
            })

            logger.info(
                "BPM adjusted on %s: %.3f → %.3f (factor %.4f)",
                self.node_id,
                old_bpm,
                new_bpm,
                factor,
            )

            return new_bpm

    def maybe_correct_drift(self) -> Tuple[bool, float]:
        """High-level helper: compute drift, decide, adjust BPM if needed.

        Returns
        -------
        (did_adjust, new_bpm)
        """
        drift_ms = self.compute_drift()
        if self._drift_correction.should_correct(drift_ms):
            factor = self._drift_correction.correction_factor(drift_ms)
            new_bpm = self.adjust_bpm(factor)
            return True, new_bpm
        return False, self._bpm

    # ── Status ──────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return a status dict with unified beat count, max drift,
        peer count, and adjustment history.
        """
        with self._lock:
            global_beat = self._beat_count
            for _peer_id, (_ts, peer_bc, _bpm, _received) in self._peer_state.items():
                if peer_bc > global_beat:
                    global_beat = peer_bc

            return {
                "node_id": self.node_id,
                "unified_beat_count": global_beat,
                "local_beat_count": self._beat_count,
                "max_drift_ms": round(self.compute_drift(), 3),
                "peer_count": len(self._peer_state),
                "bpm": round(self._bpm, 3),
                "adjustment_history": list(self._adjustment_history),
            }

    def get_unified_beat(self) -> UnifiedBeat:
        """Return a ``UnifiedBeat`` snapshot."""
        with self._lock:
            global_beat = self._beat_count
            for _peer_id, (_ts, peer_bc, _bpm, _received) in self._peer_state.items():
                if peer_bc > global_beat:
                    global_beat = peer_bc

            return UnifiedBeat(
                global_beat_count=global_beat,
                local_beat_count=self._beat_count,
                drift_ms=self.compute_drift(),
                peer_count=len(self._peer_state),
            )

    # ── Properties ──────────────────────────────────────────

    @property
    def bpm(self) -> float:
        with self._lock:
            return self._bpm

    @property
    def beat_count(self) -> int:
        with self._lock:
            return self._beat_count

    @property
    def peer_state(self) -> dict[str, Tuple[int, int, float, int]]:
        """Read-only copy of peer state."""
        with self._lock:
            return dict(self._peer_state)
