"""Tests for Metronome Mesh Gossip Bridge (nerve/metronome_mesh_bridge.py).

Covers:
    - SyncPayload serialization
    - Bridge attachment
    - Beat forwarding to gossip
    - Drift correction forwarding
    - Vector update passthrough
    - Deduplication
    - Stale message rejection
    - Remote beat handling
    - Remote drift handling
    - Node announcement
    - Metrics
"""

from __future__ import annotations

import json
import time

import pytest

from nerve.metronome_mesh_bridge import (
    BridgeConfig,
    GossipMessageType,
    MetronomeGossipBridge,
    SyncPayload,
)


# ── 1. SyncPayload ────────────────────────────────────────

class TestSyncPayload:
    def test_round_trip_json(self):
        p = SyncPayload(
            msg_type=GossipMessageType.BEAT,
            node_id="n1",
            bpm=120.0,
            beat_number=42,
            timestamp=1234.5,
            drift_ms=1.2,
            signature="sig",
            vector_update={"x": 1},
        )
        raw = p.to_json()
        p2 = SyncPayload.from_json(raw)
        assert p2.msg_type == GossipMessageType.BEAT
        assert p2.node_id == "n1"
        assert p2.bpm == 120.0
        assert p2.beat_number == 42
        assert p2.timestamp == 1234.5
        assert p2.drift_ms == 1.2
        assert p2.signature == "sig"
        assert p2.vector_update == {"x": 1}

    def test_from_json_with_defaults(self):
        raw = json.dumps(
            {
                "msg_type": "BEAT",
                "node_id": "n2",
                "bpm": 60.0,
                "beat_number": 0,
                "timestamp": 0.0,
            }
        )
        p = SyncPayload.from_json(raw)
        assert p.drift_ms == 0.0
        assert p.signature == ""
        assert p.vector_update == {}


# ── 2. Bridge basics ──────────────────────────────────────

class TestBridgeBasics:
    def test_attach_metronome(self):
        class FakeMetronome:
            node_id = "m1"

        bridge = MetronomeGossipBridge()
        bridge.attach_metronome(FakeMetronome())
        assert bridge._metronome is not None
        assert bridge._node_id == "m1"

    def test_attach_gossip(self):
        class FakeGossip:
            pass

        bridge = MetronomeGossipBridge()
        bridge.attach_gossip(FakeGossip())
        assert bridge._gossip is not None

    def test_start_stop(self):
        bridge = MetronomeGossipBridge()
        bridge.start()
        assert bridge._running is True
        bridge.stop()
        assert bridge._running is False


# ── 3. Forwarding ─────────────────────────────────────────

class TestForwarding:
    def test_beat_forwarded_to_gossip(self):
        class FakeGossip:
            def __init__(self):
                self.messages = []

            def broadcast(self, msg):
                self.messages.append(msg)

        bridge = MetronomeGossipBridge()
        gossip = FakeGossip()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=5, drift_ms=0.5)
        assert len(gossip.messages) == 1
        payload = SyncPayload.from_json(gossip.messages[0])
        assert payload.msg_type == GossipMessageType.BEAT
        assert payload.bpm == 120.0
        assert payload.beat_number == 5
        assert payload.drift_ms == 0.5

    def test_drift_forwarded_to_gossip(self):
        class FakeGossip:
            def __init__(self):
                self.messages = []

            def broadcast(self, msg):
                self.messages.append(msg)

        bridge = MetronomeGossipBridge()
        gossip = FakeGossip()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_drift_correction(target_bpm=130.0, correction=2.0)
        assert len(gossip.messages) == 1
        payload = SyncPayload.from_json(gossip.messages[0])
        assert payload.msg_type == GossipMessageType.DRIFT_CORRECTION
        assert payload.bpm == 130.0
        assert payload.drift_ms == 2.0

    def test_vector_update_forwarded_to_gossip(self):
        class FakeGossip:
            def __init__(self):
                self.messages = []

            def broadcast(self, msg):
                self.messages.append(msg)

        bridge = MetronomeGossipBridge()
        gossip = FakeGossip()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_vector_update({"agent_id": "a1", "vector": [1.0, 2.0]})
        assert len(gossip.messages) == 1
        payload = SyncPayload.from_json(gossip.messages[0])
        assert payload.msg_type == GossipMessageType.VECTOR_UPDATE
        assert payload.vector_update["agent_id"] == "a1"

    def test_no_gossip_does_not_crash(self):
        bridge = MetronomeGossipBridge()
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        # Should not raise

    def test_disabled_beat_gossip(self):
        class FakeGossip:
            def __init__(self):
                self.messages = []

            def broadcast(self, msg):
                self.messages.append(msg)

        cfg = BridgeConfig(enable_beat_gossip=False)
        bridge = MetronomeGossipBridge(cfg)
        gossip = FakeGossip()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        assert len(gossip.messages) == 0


# ── 4. Deduplication ──────────────────────────────────────

class TestDeduplication:
    def test_duplicate_beat_dropped(self):
        class FakeGossip:
            def __init__(self):
                self.messages = []

            def broadcast(self, msg):
                self.messages.append(msg)

        bridge = MetronomeGossipBridge()
        gossip = FakeGossip()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        assert len(gossip.messages) == 1

    def test_different_beat_allowed(self):
        class FakeGossip:
            def __init__(self):
                self.messages = []

            def broadcast(self, msg):
                self.messages.append(msg)

        bridge = MetronomeGossipBridge()
        gossip = FakeGossip()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        bridge.on_metronome_beat(bpm=120.0, beat_number=2)
        assert len(gossip.messages) == 2


# ── 5. Receiving ──────────────────────────────────────────

class TestReceiving:
    def test_remote_beat_handled(self):
        class FakeMetronome:
            def __init__(self):
                self.beats = []

            def on_remote_beat(self, node_id, bpm, beat_number, timestamp):
                self.beats.append((node_id, bpm, beat_number))

        bridge = MetronomeGossipBridge()
        metro = FakeMetronome()
        bridge.attach_metronome(metro)
        bridge.start()

        payload = SyncPayload(
            msg_type=GossipMessageType.BEAT,
            node_id="peer-1",
            bpm=120.0,
            beat_number=10,
            timestamp=time.time(),
        )
        bridge.on_gossip_receive(payload.to_json())
        assert len(metro.beats) == 1
        assert metro.beats[0] == ("peer-1", 120.0, 10)

    def test_remote_drift_handled(self):
        class FakeMetronome:
            def __init__(self):
                self.corrections = []

            def apply_drift_correction(self, node_id, correction_ms):
                self.corrections.append((node_id, correction_ms))

        bridge = MetronomeGossipBridge()
        metro = FakeMetronome()
        bridge.attach_metronome(metro)
        bridge.start()

        payload = SyncPayload(
            msg_type=GossipMessageType.DRIFT_CORRECTION,
            node_id="peer-1",
            bpm=120.0,
            beat_number=-1,
            timestamp=time.time(),
            drift_ms=2.5,
        )
        bridge.on_gossip_receive(payload.to_json())
        assert len(metro.corrections) == 1
        assert metro.corrections[0] == ("peer-1", 2.5)

    def test_stale_message_rejected(self):
        bridge = MetronomeGossipBridge()
        bridge.start()

        payload = SyncPayload(
            msg_type=GossipMessageType.BEAT,
            node_id="peer-1",
            bpm=120.0,
            beat_number=10,
            timestamp=time.time() - 100,  # very old
        )
        result = bridge.on_gossip_receive(payload.to_json())
        assert result is None

    def test_non_syncpayload_returns_none(self):
        bridge = MetronomeGossipBridge()
        result = bridge.on_gossip_receive("plain text not json")
        assert result is None

    def test_vector_update_passed_through(self):
        bridge = MetronomeGossipBridge()
        bridge.start()

        payload = SyncPayload(
            msg_type=GossipMessageType.VECTOR_UPDATE,
            node_id="peer-1",
            bpm=0.0,
            beat_number=-1,
            timestamp=time.time(),
            vector_update={"agent_id": "a1"},
        )
        result = bridge.on_gossip_receive(payload.to_json())
        assert result is not None
        assert result.msg_type == GossipMessageType.VECTOR_UPDATE


# ── 6. Node announcement ──────────────────────────────────

class TestNodeAnnouncement:
    def test_announce_node_broadcasts(self):
        class FakeGossip:
            def __init__(self):
                self.messages = []

            def broadcast(self, msg):
                self.messages.append(msg)

        bridge = MetronomeGossipBridge()
        gossip = FakeGossip()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.announce_node(bpm=120.0)
        assert len(gossip.messages) == 1
        payload = SyncPayload.from_json(gossip.messages[0])
        assert payload.msg_type == GossipMessageType.NODE_ANNOUNCE
        assert payload.bpm == 120.0


# ── 7. Metrics ────────────────────────────────────────────

class TestMetrics:
    def test_get_metrics(self):
        bridge = MetronomeGossipBridge()
        m = bridge.get_metrics()
        assert m["node_id"] == "unknown"
        assert m["metronome_attached"] is False
        assert m["gossip_attached"] is False
        assert m["running"] is False
        assert "dedup_window_size" in m
