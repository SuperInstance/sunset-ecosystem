"""
tests/test_distributed_metronome_bridge.py

Tests for nerve/distributed_metronome_bridge.py.
Covers:
* Tick increment with microsecond precision
* Sync message construction and peer dispatch
* Signature validation (accept / reject)
* Drift computation from peer timestamps
* BPM adjustment gates (threshold, clamp, history)
* PID integral / derivative accumulation
* Correction factor safe range (0.95–1.05)
* UnifiedBeat dataclass
* Thread-safe concurrent syncs
* Status dict accuracy
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from nerve.distributed_metronome_bridge import (
    DriftCorrection,
    MetronomeBridge,
    SyncMessage,
    UnifiedBeat,
    MetronomeBridgeError,
)


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def mock_identity():
    """AgentIdentity-like mock with deterministic sign/verify."""
    ident = MagicMock()
    ident.sign_task = MagicMock(return_value="sig_valid_12345")
    ident.verify_task = MagicMock(return_value=True)
    return ident


@pytest.fixture
def mock_identity_bad():
    """AgentIdentity-like mock that rejects every signature."""
    ident = MagicMock()
    ident.sign_task = MagicMock(return_value="sig_bad_99999")
    ident.verify_task = MagicMock(return_value=False)
    return ident


@pytest.fixture
def bridge_no_peers(mock_identity):
    return MetronomeBridge(
        local_bpm=120.0,
        node_id="alpha",
        peers=[],
        identity=mock_identity,
    )


@pytest.fixture
def bridge_three_peers(mock_identity):
    return MetronomeBridge(
        local_bpm=120.0,
        node_id="alpha",
        peers=["beta", "gamma", "delta"],
        identity=mock_identity,
    )


# ── UnifiedBeat ─────────────────────────────────────────────

class TestUnifiedBeat:
    def test_creation_and_fields(self):
        ub = UnifiedBeat(
            global_beat_count=42,
            local_beat_count=40,
            drift_ms=12.5,
            peer_count=3,
        )
        assert ub.global_beat_count == 42
        assert ub.local_beat_count == 40
        assert ub.drift_ms == 12.5
        assert ub.peer_count == 3

    def test_to_dict_roundtrip(self):
        ub = UnifiedBeat(global_beat_count=7, local_beat_count=5, drift_ms=3.0, peer_count=1)
        d = ub.to_dict()
        assert d["global_beat_count"] == 7
        assert d["drift_ms"] == 3.0
        assert "timestamp" in d


# ── DriftCorrection ─────────────────────────────────────────

class TestDriftCorrection:
    def test_should_correct_true_when_above_threshold(self):
        dc = DriftCorrection(threshold_ms=10.0)
        assert dc.should_correct(15.0) is True
        assert dc.should_correct(-15.0) is True

    def test_should_correct_false_when_below_threshold(self):
        dc = DriftCorrection(threshold_ms=10.0)
        assert dc.should_correct(5.0) is False
        assert dc.should_correct(0.0) is False

    def test_correction_factor_within_safe_range(self):
        dc = DriftCorrection()
        for drift in [-200.0, -50.0, -10.0, 10.0, 50.0, 200.0, 0.0]:
            f = dc.correction_factor(drift)
            assert 0.95 <= f <= 1.05, f"drift={drift} gave factor={f}"

    def test_correction_factor_resets_and_stable_at_zero(self):
        dc = DriftCorrection()
        f1 = dc.correction_factor(0.0)
        f2 = dc.correction_factor(0.0)
        assert f1 == pytest.approx(1.0, abs=0.001)
        assert f2 == pytest.approx(1.0, abs=0.001)

    def test_pid_integral_accumulates_error(self):
        dc = DriftCorrection(ki=0.1, kp=0.0, kd=0.0)
        dc.correction_factor(20.0)   # first call establishes dt
        time.sleep(0.01)
        f = dc.correction_factor(20.0)
        # With positive drift and non-zero ki, factor should be > 1.0
        assert f > 1.0
        assert f <= 1.05

    def test_integral_windup_clamped(self):
        dc = DriftCorrection(integral_window_ms=100.0, ki=1.0)
        # Bombard with huge drift to try to wind up
        for _ in range(50):
            dc.correction_factor(1000.0)
            time.sleep(0.001)
        f = dc.correction_factor(1000.0)
        assert f <= 1.05
        assert f >= 0.95

    def test_reset_clears_state(self):
        dc = DriftCorrection()
        dc.correction_factor(50.0)
        dc.reset()
        assert dc._integral == 0.0
        assert dc._last_drift == 0.0


# ── MetronomeBridge.tick ────────────────────────────────────

class TestTick:
    def test_tick_increments_correctly(self, bridge_no_peers):
        b = bridge_no_peers
        assert b.beat_count == 0
        assert b.tick() == 1
        assert b.tick() == 2
        assert b.beat_count == 2

    def test_tick_microsecond_precision(self, bridge_no_peers):
        b = bridge_no_peers
        before = time.time_ns()
        b.tick()
        after = time.time_ns()
        # The internal _last_tick_ns should land between before/after
        assert before <= b._last_tick_ns <= after


# ── MetronomeBridge.sync_with_peers ──────────────────────────

class TestSyncWithPeers:
    def test_sync_message_sent_to_all_peers(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        msgs = b.sync_with_peers()
        assert len(msgs) == 3
        assert {m.node_id for m in msgs} == {"alpha"}
        assert all(m.signature == "sig_valid_12345" for m in msgs)
        # sign_task called once (payload signed once, then copied)
        assert mock_identity.sign_task.call_count == 1

    def test_sync_uses_send_fn_when_provided(self, mock_identity):
        sent: list = []

        def capture(peer_id, payload):
            sent.append((peer_id, payload))

        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta", "gamma"],
            identity=mock_identity,
            send_fn=capture,
        )
        b.tick()
        msgs = b.sync_with_peers()
        assert len(sent) == 2
        assert sent[0][0] == "beta"
        assert sent[1][0] == "gamma"
        assert sent[0][1]["node_id"] == "alpha"

    def test_sync_with_no_identity_skips_signature(self):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta"],
            identity=None,
        )
        b.tick()
        msgs = b.sync_with_peers()
        assert msgs[0].signature == ""


# ── MetronomeBridge.receive_sync ────────────────────────────

class TestReceiveSync:
    def test_valid_signature_accepted(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        ok = b.receive_sync("beta", time.time_ns(), "sig_valid_12345")
        assert ok is True
        assert "beta" in b.peer_state

    def test_invalid_signature_rejected(self, bridge_three_peers, mock_identity_bad):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta"],
            identity=mock_identity_bad,
        )
        ok = b.receive_sync("beta", time.time_ns(), "sig_bad_99999")
        assert ok is False
        assert "beta" not in b.peer_state

    def test_receive_without_identity_and_no_signature_accepted(self):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta"],
            identity=None,
        )
        ok = b.receive_sync("beta", time.time_ns(), "")
        assert ok is True

    def test_receive_without_identity_but_with_signature_rejected(self):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta"],
            identity=None,
        )
        ok = b.receive_sync("beta", time.time_ns(), "some_sig")
        assert ok is False

    def test_peer_timestamp_recorded_correctly(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        ts = 1_700_000_000_000_000_000
        ok = b.receive_sync("beta", ts, "sig_valid_12345", beat_count=99, bpm=130.0)
        assert ok is True
        state = b.peer_state["beta"]
        assert state[0] == ts
        assert state[1] == 99
        assert state[2] == 130.0


# ── MetronomeBridge.compute_drift ───────────────────────────

class TestComputeDrift:
    def test_drift_zero_when_no_peers(self, bridge_no_peers):
        assert bridge_no_peers.compute_drift() == 0.0

    def test_drift_computed_from_peer_timestamps(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        local_ts = time.time_ns()
        b._last_tick_ns = local_ts
        # Peer is 50 ms behind
        peer_ts = local_ts - int(50_000_000)
        b.receive_sync("beta", peer_ts, "sig_valid_12345")
        drift = b.compute_drift()
        assert drift == pytest.approx(50.0, abs=0.1)

    def test_drift_max_across_multiple_peers(self, mock_identity):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta", "gamma"],
            identity=mock_identity,
        )
        local_ts = time.time_ns()
        b._last_tick_ns = local_ts
        b.receive_sync("beta", local_ts - int(30_000_000), "sig_valid_12345")
        b.receive_sync("gamma", local_ts - int(80_000_000), "sig_valid_12345")
        drift = b.compute_drift()
        assert drift == pytest.approx(80.0, abs=0.1)


# ── MetronomeBridge.adjust_bpm ─────────────────────────────

class TestAdjustBpm:
    def test_bpm_adjusted_when_drift_above_threshold(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        local_ts = time.time_ns()
        b._last_tick_ns = local_ts
        b.receive_sync("beta", local_ts - int(50_000_000), "sig_valid_12345")
        did_adjust, new_bpm = b.maybe_correct_drift()
        assert did_adjust is True
        assert new_bpm != 120.0

    def test_bpm_not_adjusted_when_drift_below_threshold(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        b._drift_correction.threshold_ms = 100.0
        local_ts = time.time_ns()
        b._last_tick_ns = local_ts
        b.receive_sync("beta", local_ts - int(5_000_000), "sig_valid_12345")
        did_adjust, new_bpm = b.maybe_correct_drift()
        assert did_adjust is False
        assert new_bpm == 120.0

    def test_correction_factor_within_safe_range(self, mock_identity):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta"],
            identity=mock_identity,
            max_bpm_delta_ratio=0.05,
        )
        # Force a huge drift to get maximum correction
        local_ts = time.time_ns()
        b._last_tick_ns = local_ts
        b.receive_sync("beta", local_ts - int(500_000_000), "sig_valid_12345")
        did_adjust, new_bpm = b.maybe_correct_drift()
        assert did_adjust is True
        assert 114.0 <= new_bpm <= 126.0  # ±5%

    def test_adjustment_history_tracked(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        old = b.bpm
        b.adjust_bpm(1.02)
        hist = b.get_status()["adjustment_history"]
        assert len(hist) == 1
        assert hist[0]["old_bpm"] == old
        assert hist[0]["new_bpm"] == pytest.approx(old * 1.02, abs=0.001)
        assert hist[0]["reason"] == "drift_correction"


# ── MetronomeBridge.status ──────────────────────────────────

class TestStatus:
    def test_status_dict_accurate(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        b.tick()
        b.receive_sync("beta", time.time_ns(), "sig_valid_12345", beat_count=200)
        s = b.get_status()
        assert s["node_id"] == "alpha"
        assert s["local_beat_count"] == 1
        assert s["unified_beat_count"] == 200
        assert s["peer_count"] == 1
        assert "bpm" in s
        assert "adjustment_history" in s

    def test_get_unified_beat_matches_status(self, bridge_three_peers, mock_identity):
        b = bridge_three_peers
        b.tick()
        b.receive_sync("beta", time.time_ns(), "sig_valid_12345", beat_count=50)
        ub = b.get_unified_beat()
        assert ub.global_beat_count == 50
        assert ub.local_beat_count == 1
        assert ub.peer_count == 1


# ── Thread safety ───────────────────────────────────────────

class TestThreadSafety:
    def test_thread_safe_concurrent_syncs(self, mock_identity):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=["beta", "gamma", "delta"],
            identity=mock_identity,
        )
        errors = []
        results = []

        def worker():
            try:
                b.tick()
                msgs = b.sync_with_peers()
                results.append(len(msgs))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 16
        assert all(r == 3 for r in results)
        # Beat count should be exactly 16 (every thread ticked once)
        assert b.beat_count == 16

    def test_thread_safe_concurrent_receives(self, mock_identity):
        b = MetronomeBridge(
            local_bpm=120.0,
            node_id="alpha",
            peers=[f"peer_{i}" for i in range(20)],
            identity=mock_identity,
        )
        errors = []

        def worker(i):
            try:
                ts = time.time_ns() - i * 1_000_000
                b.receive_sync(f"peer_{i}", ts, "sig_valid_12345")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(b.peer_state) == 20


# ── SyncMessage ─────────────────────────────────────────────

class TestSyncMessage:
    def test_payload_for_signing_stable(self):
        m = SyncMessage(node_id="a", beat_count=1, timestamp_ns=2, bpm=3.0, signature="x")
        p = m.payload_for_signing()
        assert "signature" not in p
        assert p == {"node_id": "a", "beat_count": 1, "timestamp_ns": 2, "bpm": 3.0}

    def test_roundtrip_dict(self):
        original = SyncMessage(node_id="n", beat_count=5, timestamp_ns=9, bpm=120.0, signature="s")
        d = original.to_dict()
        restored = SyncMessage.from_dict(d)
        assert restored == original
