#!/usr/bin/env python3
"""Metronome Mesh Gossip Bridge — transport metronome sync over mesh gossip.

Unifies two reverse-actualization P0 modules:
  • Distributed Metronome Bridge (nerve/distributed_metronome_bridge.py)
  • Mesh Vector Gossip (swarm/mesh_vector_gossip.py)

Metronome sync messages (beat ticks, drift corrections) are wrapped as
gossip payloads and propagated through the mesh gossip protocol. This
means one gossip channel handles both:
  - Vector table CRDT updates
  - Metronome beat synchronization

Reference: docs/METRONOME_MESH_BRIDGE.md
"""

from __future__ import annotations

__all__ = [
    "MetronomeGossipBridge",
    "GossipMessageType",
    "SyncPayload",
    "BridgeConfig",
]

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# ── data structures ───────────────────────────────────────────


class GossipMessageType(Enum):
    BEAT = auto()
    DRIFT_CORRECTION = auto()
    NODE_ANNOUNCE = auto()
    VECTOR_UPDATE = auto()  # passthrough for mesh_vector_gossip


@dataclass(frozen=True)
class SyncPayload:
    """Payload for a metronome sync message transported over gossip."""

    msg_type: GossipMessageType
    node_id: str
    bpm: float
    beat_number: int
    timestamp: float
    drift_ms: float = 0.0
    signature: str = ""  # SignedWAL signature
    vector_update: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "msg_type": self.msg_type.name,
                "node_id": self.node_id,
                "bpm": self.bpm,
                "beat_number": self.beat_number,
                "timestamp": self.timestamp,
                "drift_ms": self.drift_ms,
                "signature": self.signature,
                "vector_update": self.vector_update,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "SyncPayload":
        d = json.loads(raw)
        return cls(
            msg_type=GossipMessageType[d["msg_type"]],
            node_id=d["node_id"],
            bpm=d["bpm"],
            beat_number=d["beat_number"],
            timestamp=d["timestamp"],
            drift_ms=d.get("drift_ms", 0.0),
            signature=d.get("signature", ""),
            vector_update=d.get("vector_update", {}),
        )


@dataclass
class BridgeConfig:
    """Configuration for the metronome-gossip bridge."""

    enable_drift_gossip: bool = True
    enable_beat_gossip: bool = True
    enable_vector_passthrough: bool = True
    max_gossip_age_sec: float = 30.0
    dedup_window_sec: float = 60.0


# ── bridge ────────────────────────────────────────────────────


class MetronomeGossipBridge:
    """Bridge between metronome and mesh gossip.

    Usage
    -----
    1. Create bridge: ``bridge = MetronomeGossipBridge(config)``
    2. Attach metronome: ``bridge.attach_metronome(metronome)``
    3. Attach gossip: ``bridge.attach_gossip(gossip)``
    4. Start: ``bridge.start()``
    """

    def __init__(self, config: BridgeConfig | None = None) -> None:
        self.config = config or BridgeConfig()
        self._metronome: Any | None = None
        self._gossip: Any | None = None
        self._node_id: str = "unknown"
        self._seen: dict[str, float] = {}  # dedup window
        self._running = False

    # ── attachment ────────────────────────────────────────

    def attach_metronome(self, metronome: Any) -> None:
        """Attach a DistributedMetronomeBridge instance."""
        self._metronome = metronome
        self._node_id = getattr(metronome, "node_id", "unknown")

    def attach_gossip(self, gossip: Any) -> None:
        """Attach a MeshVectorGossip instance."""
        self._gossip = gossip

    # ── lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        """Begin forwarding metronome events to gossip."""
        self._running = True
        logger.info("MetronomeGossipBridge started for node %s", self._node_id)

    def stop(self) -> None:
        self._running = False

    # ── forwarding ──────────────────────────────────────────

    def on_metronome_beat(
        self, bpm: float, beat_number: int, drift_ms: float = 0.0
    ) -> None:
        """Called by metronome on each beat. Forward to gossip."""
        if not self._running or not self.config.enable_beat_gossip:
            return

        payload = SyncPayload(
            msg_type=GossipMessageType.BEAT,
            node_id=self._node_id,
            bpm=bpm,
            beat_number=beat_number,
            timestamp=time.time(),
            drift_ms=drift_ms,
        )
        self._gossip_broadcast(payload)

    def on_drift_correction(self, target_bpm: float, correction: float) -> None:
        """Called by metronome when drift is corrected. Forward to gossip."""
        if not self._running or not self.config.enable_drift_gossip:
            return

        payload = SyncPayload(
            msg_type=GossipMessageType.DRIFT_CORRECTION,
            node_id=self._node_id,
            bpm=target_bpm,
            beat_number=-1,
            timestamp=time.time(),
            drift_ms=correction,
        )
        self._gossip_broadcast(payload)

    def on_vector_update(self, update: dict[str, Any]) -> None:
        """Called when local vector table changes. Forward to gossip."""
        if not self._running or not self.config.enable_vector_passthrough:
            return

        payload = SyncPayload(
            msg_type=GossipMessageType.VECTOR_UPDATE,
            node_id=self._node_id,
            bpm=0.0,
            beat_number=-1,
            timestamp=time.time(),
            vector_update=update,
        )
        self._gossip_broadcast(payload)

    def _gossip_broadcast(self, payload: SyncPayload) -> None:
        """Send payload through gossip mesh."""
        if self._gossip is None:
            return

        # Deduplication
        key = f"{payload.node_id}:{payload.msg_type.name}:{payload.beat_number}"
        now = time.time()
        if key in self._seen:
            if now - self._seen[key] < self.config.dedup_window_sec:
                return
        self._seen[key] = now

        # Clean old dedup entries
        cutoff = now - self.config.dedup_window_sec
        self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

        # Broadcast
        try:
            if hasattr(self._gossip, "broadcast"):
                self._gossip.broadcast(payload.to_json())
            elif hasattr(self._gossip, "send"):
                self._gossip.send(payload.to_json())
            else:
                logger.warning("Gossip has no broadcast/send method")
        except Exception as exc:
            logger.error("Gossip broadcast failed: %s", exc)

    # ── receiving ───────────────────────────────────────────

    def on_gossip_receive(self, raw_payload: str) -> SyncPayload | None:
        """Process an incoming gossip payload.

        Returns the parsed payload if it's a metronome message,
        None if it's a plain vector update (pass through).
        """
        try:
            payload = SyncPayload.from_json(raw_payload)
        except (json.JSONDecodeError, KeyError):
            # Not a SyncPayload — probably raw vector gossip
            return None

        # Age check
        age = time.time() - payload.timestamp
        if age > self.config.max_gossip_age_sec:
            logger.debug("Stale gossip from %s (age %.1fs)", payload.node_id, age)
            return None

        if payload.msg_type == GossipMessageType.BEAT:
            self._handle_remote_beat(payload)
        elif payload.msg_type == GossipMessageType.DRIFT_CORRECTION:
            self._handle_remote_drift(payload)
        elif payload.msg_type == GossipMessageType.NODE_ANNOUNCE:
            self._handle_node_announce(payload)
        elif payload.msg_type == GossipMessageType.VECTOR_UPDATE:
            # Pass through to vector table handler
            return payload

        return payload

    def _handle_remote_beat(self, payload: SyncPayload) -> None:
        """Update local metronome with remote beat info."""
        if self._metronome is None:
            return
        if hasattr(self._metronome, "on_remote_beat"):
            self._metronome.on_remote_beat(
                node_id=payload.node_id,
                bpm=payload.bpm,
                beat_number=payload.beat_number,
                timestamp=payload.timestamp,
            )
        elif hasattr(self._metronome, "sync"):
            self._metronome.sync(payload.node_id, payload.bpm, payload.timestamp)

    def _handle_remote_drift(self, payload: SyncPayload) -> None:
        """Apply remote drift correction if applicable."""
        if self._metronome is None:
            return
        if hasattr(self._metronome, "apply_drift_correction"):
            self._metronome.apply_drift_correction(
                node_id=payload.node_id,
                correction_ms=payload.drift_ms,
            )

    def _handle_node_announce(self, payload: SyncPayload) -> None:
        """New node discovered via gossip."""
        logger.info(
            "Node announced via gossip: %s @ %.1f BPM", payload.node_id, payload.bpm
        )

    # ── announcement ────────────────────────────────────────

    def announce_node(self, bpm: float) -> None:
        """Announce this node to the mesh."""
        payload = SyncPayload(
            msg_type=GossipMessageType.NODE_ANNOUNCE,
            node_id=self._node_id,
            bpm=bpm,
            beat_number=0,
            timestamp=time.time(),
        )
        self._gossip_broadcast(payload)

    # ── metrics ─────────────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        return {
            "node_id": self._node_id,
            "metronome_attached": self._metronome is not None,
            "gossip_attached": self._gossip is not None,
            "running": self._running,
            "dedup_window_size": len(self._seen),
            "config": asdict(self.config),
        }
