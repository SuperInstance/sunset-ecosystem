"""Tests for MetronomeGossipBridge — metronome sync over mesh gossip.

Covers SyncPayload, BridgeConfig, MetronomeGossipBridge lifecycle,
forwarding, dedup, receiving, and metrics.
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from nerve.metronome_mesh_bridge import (
    BridgeConfig,
    GossipMessageType,
    MetronomeGossipBridge,
    SyncPayload,
)


# ---------------------------------------------------------------------------
# SyncPayload
# ---------------------------------------------------------------------------


class TestSyncPayload:
    def test_roundtrip(self):
        p = SyncPayload(
            msg_type=GossipMessageType.BEAT,
            node_id="n1",
            bpm=120.0,
            beat_number=7,
            timestamp=1234567890.0,
            drift_ms=5.0,
            signature="sig123",
            vector_update={"k": "v"},
        )
        raw = p.to_json()
        p2 = SyncPayload.from_json(raw)
        assert p2.msg_type == GossipMessageType.BEAT
        assert p2.node_id == "n1"
        assert p2.bpm == 120.0
        assert p2.beat_number == 7
        assert p2.drift_ms == 5.0
        assert p2.signature == "sig123"
        assert p2.vector_update == {"k": "v"}

    def test_defaults(self):
        p = SyncPayload(
            msg_type=GossipMessageType.NODE_ANNOUNCE,
            node_id="n2",
            bpm=0.0,
            beat_number=0,
            timestamp=0.0,
        )
        assert p.drift_ms == 0.0
        assert p.signature == ""
        assert p.vector_update == {}


# ---------------------------------------------------------------------------
# BridgeConfig
# ---------------------------------------------------------------------------


class TestBridgeConfig:
    def test_defaults(self):
        cfg = BridgeConfig()
        assert cfg.enable_drift_gossip is True
        assert cfg.enable_beat_gossip is True
        assert cfg.enable_vector_passthrough is True
        assert cfg.max_gossip_age_sec == 30.0
        assert cfg.dedup_window_sec == 60.0


# ---------------------------------------------------------------------------
# MetronomeGossipBridge init
# ---------------------------------------------------------------------------


class TestBridgeInit:
    def test_defaults(self):
        bridge = MetronomeGossipBridge()
        assert bridge._node_id == "unknown"
        assert not bridge._running
        assert bridge._metronome is None
        assert bridge._gossip is None

    def test_custom_config(self):
        cfg = BridgeConfig(enable_beat_gossip=False)
        bridge = MetronomeGossipBridge(cfg)
        assert bridge.config.enable_beat_gossip is False


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------


class TestAttachment:
    def test_attach_metronome(self):
        bridge = MetronomeGossipBridge()
        metro = MagicMock()
        metro.node_id = "metro-1"
        bridge.attach_metronome(metro)
        assert bridge._metronome is metro
        assert bridge._node_id == "metro-1"

    def test_attach_gossip(self):
        bridge = MetronomeGossipBridge()
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        assert bridge._gossip is gossip


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_stop(self):
        bridge = MetronomeGossipBridge()
        bridge.start()
        assert bridge._running is True
        bridge.stop()
        assert bridge._running is False


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------


class TestForwarding:
    def test_on_metronome_beat(self):
        bridge = MetronomeGossipBridge()
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=5, drift_ms=2.0)
        assert gossip.broadcast.called or gossip.send.called

    def test_on_metronome_beat_disabled(self):
        cfg = BridgeConfig(enable_beat_gossip=False)
        bridge = MetronomeGossipBridge(cfg)
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=5)
        assert not gossip.broadcast.called
        assert not gossip.send.called

    def test_on_drift_correction(self):
        bridge = MetronomeGossipBridge()
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_drift_correction(target_bpm=120.0, correction=1.5)
        assert gossip.broadcast.called or gossip.send.called

    def test_on_vector_update(self):
        bridge = MetronomeGossipBridge()
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_vector_update({"vec": [1.0, 2.0]})
        assert gossip.broadcast.called or gossip.send.called

    def test_no_gossip_no_crash(self):
        bridge = MetronomeGossipBridge()
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        # should not raise

    def test_not_running_no_forward(self):
        bridge = MetronomeGossipBridge()
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        # not started
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        assert not gossip.broadcast.called


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_dedup_same_key(self):
        bridge = MetronomeGossipBridge()
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        # second should be deduped
        assert gossip.broadcast.call_count == 1

    def test_dedup_expires(self):
        bridge = MetronomeGossipBridge()
        bridge.config.dedup_window_sec = 0.01
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        time.sleep(0.02)
        bridge.on_metronome_beat(bpm=120.0, beat_number=1)
        assert gossip.broadcast.call_count == 2


# ---------------------------------------------------------------------------
# Receiving
# ---------------------------------------------------------------------------


class TestReceiving:
    def test_beat_message(self):
        bridge = MetronomeGossipBridge()
        metro = MagicMock()
        bridge.attach_metronome(metro)
        payload = SyncPayload(
            msg_type=GossipMessageType.BEAT,
            node_id="n1",
            bpm=120.0,
            beat_number=3,
            timestamp=time.time(),
        )
        result = bridge.on_gossip_receive(payload.to_json())
        assert result is not None
        assert metro.on_remote_beat.called or metro.sync.called

    def test_drift_message(self):
        bridge = MetronomeGossipBridge()
        metro = MagicMock()
        metro.apply_drift_correction = MagicMock()
        bridge.attach_metronome(metro)
        payload = SyncPayload(
            msg_type=GossipMessageType.DRIFT_CORRECTION,
            node_id="n1",
            bpm=120.0,
            beat_number=-1,
            timestamp=time.time(),
            drift_ms=2.0,
        )
        bridge.on_gossip_receive(payload.to_json())
        assert metro.apply_drift_correction.called

    def test_stale_message(self):
        bridge = MetronomeGossipBridge()
        payload = SyncPayload(
            msg_type=GossipMessageType.BEAT,
            node_id="n1",
            bpm=120.0,
            beat_number=1,
            timestamp=time.time() - 100.0,
        )
        result = bridge.on_gossip_receive(payload.to_json())
        assert result is None

    def test_non_sync_payload(self):
        bridge = MetronomeGossipBridge()
        result = bridge.on_gossip_receive("not json")
        assert result is None

    def test_vector_update_passes_through(self):
        bridge = MetronomeGossipBridge()
        payload = SyncPayload(
            msg_type=GossipMessageType.VECTOR_UPDATE,
            node_id="n1",
            bpm=0.0,
            beat_number=-1,
            timestamp=time.time(),
            vector_update={"x": [1.0]},
        )
        result = bridge.on_gossip_receive(payload.to_json())
        assert result is not None
        assert result.vector_update == {"x": [1.0]}


# ---------------------------------------------------------------------------
# Announcement
# ---------------------------------------------------------------------------


class TestAnnouncement:
    def test_announce_node(self):
        bridge = MetronomeGossipBridge()
        gossip = MagicMock()
        bridge.attach_gossip(gossip)
        bridge.start()
        bridge.announce_node(bpm=100.0)
        assert gossip.broadcast.called or gossip.send.called


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_basic(self):
        bridge = MetronomeGossipBridge()
        m = bridge.get_metrics()
        assert m["node_id"] == "unknown"
        assert m["metronome_attached"] is False
        assert m["gossip_attached"] is False
        assert m["running"] is False

    def test_with_attachments(self):
        bridge = MetronomeGossipBridge()
        bridge.attach_metronome(MagicMock())
        bridge.attach_gossip(MagicMock())
        bridge.start()
        m = bridge.get_metrics()
        assert m["metronome_attached"] is True
        assert m["gossip_attached"] is True
        assert m["running"] is True
