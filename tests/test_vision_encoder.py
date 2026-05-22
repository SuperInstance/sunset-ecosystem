"""Tests for the Vision Tile Encoder pipeline.

These tests use the random_projection backend by default so they pass
without transformers, CLIP, torch, or heavy model downloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from perception import VisionTileEncoder, WebcamCapture, ScreenCapture, EncoderBackend


# ── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def encoder():
    """VisionTileEncoder with deterministic random projection."""
    return VisionTileEncoder(model="random_projection", device="cpu", target_dim=512)


@pytest.fixture
def synthetic_rgb():
    """A deterministic synthetic RGB frame."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


# ── VisionTileEncoder ──────────────────────────────────────

class TestEncodeFrame:
    """Test encode_frame returns a 512-dim normalised embedding."""

    def test_returns_512_dim(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        assert emb.shape == (512,)

    def test_returns_float32(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        assert emb.dtype == np.float32

    def test_l2_normalised(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_different_frames_different_embeddings(self, encoder):
        f1 = np.random.RandomState(1).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        f2 = np.random.RandomState(2).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        e1 = encoder.encode_frame(f1)
        e2 = encoder.encode_frame(f2)
        # Different inputs should not produce identical embeddings
        assert not np.allclose(e1, e2, atol=1e-4)

    def test_same_frame_same_embedding(self, encoder, synthetic_rgb):
        e1 = encoder.encode_frame(synthetic_rgb)
        e2 = encoder.encode_frame(synthetic_rgb)
        assert np.allclose(e1, e2, atol=1e-5)


class TestEncodeBatch:
    """Test batch encoding returns correct shape and dtype."""

    def test_batch_shape(self, encoder):
        frames = [
            np.random.RandomState(i).randint(0, 256, (240, 320, 3), dtype=np.uint8)
            for i in range(4)
        ]
        batch = encoder.encode_batch(frames)
        assert batch.shape == (4, 512)
        assert batch.dtype == np.float32

    def test_batch_empty(self, encoder):
        batch = encoder.encode_batch([])
        assert batch.shape == (0, 512)

    def test_batch_matches_individual(self, encoder, synthetic_rgb):
        frames = [synthetic_rgb for _ in range(3)]
        batch = encoder.encode_batch(frames)
        single = encoder.encode_frame(synthetic_rgb)
        for i in range(3):
            assert np.allclose(batch[i], single, atol=1e-5)


class TestToTile:
    """Test tile format matches NerveTopology contract."""

    def test_has_all_required_keys(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        tile = encoder.to_tile(emb, source="webcam_0")
        assert tile["type"] == "vision"
        assert tile["source"] == "webcam_0"
        assert isinstance(tile["embedding"], np.ndarray)
        assert "timestamp" in tile
        assert isinstance(tile["timestamp"], float)
        assert "metadata" in tile

    def test_metadata_keys(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        tile = encoder.to_tile(emb, source="screen_capture")
        meta = tile["metadata"]
        assert meta["model"] == encoder.model_name
        assert meta["backend"] == encoder.backend
        assert meta["device"] == encoder.device
        assert "fps" in meta
        assert "frame_count" in meta

    def test_extra_metadata_merged(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        tile = encoder.to_tile(emb, source="test", extra_metadata={"roi": "center"})
        assert tile["metadata"]["roi"] == "center"


class TestToSignal:
    """Test projection from 512-dim to NerveTopology signal_dim."""

    def test_signal_shape(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        signal = encoder.to_signal(emb, signal_dim=64)
        assert signal.shape == (64,)
        assert signal.dtype == np.float32

    def test_signal_normalised(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        signal = encoder.to_signal(emb, signal_dim=64)
        norm = np.linalg.norm(signal)
        assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_signal_deterministic(self, encoder, synthetic_rgb):
        emb = encoder.encode_frame(synthetic_rgb)
        s1 = encoder.to_signal(emb, signal_dim=64)
        s2 = encoder.to_signal(emb, signal_dim=64)
        assert np.allclose(s1, s2, atol=1e-5)

    def test_different_embeddings_different_signals(self, encoder):
        f1 = np.random.RandomState(3).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        f2 = np.random.RandomState(4).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        e1 = encoder.encode_frame(f1)
        e2 = encoder.encode_frame(f2)
        s1 = encoder.to_signal(e1, signal_dim=64)
        s2 = encoder.to_signal(e2, signal_dim=64)
        # Same image should map to same signal; different images to different
        assert not np.allclose(s1, s2, atol=1e-2)


class TestEncoderProperties:
    """Test encoder state properties."""

    def test_backend_is_string(self, encoder):
        assert isinstance(encoder.backend, str)
        assert encoder.backend == "RANDOM_PROJECTION"

    def test_fps_zero_before_encode(self, encoder):
        assert encoder.fps == 0.0

    def test_fps_nonzero_after_encode(self, encoder, synthetic_rgb):
        encoder.encode_frame(synthetic_rgb)
        assert encoder.fps > 0.0

    def test_frame_count_increments(self, encoder, synthetic_rgb):
        assert encoder._frame_count == 0
        encoder.encode_frame(synthetic_rgb)
        assert encoder._frame_count == 1
        encoder.encode_frame(synthetic_rgb)
        assert encoder._frame_count == 2


# ── Capture sources (mock, no real webcam needed) ──────────────

class TestWebcamCaptureMock:
    """Test WebcamCapture with mock frame (no hardware)."""

    def test_import_error_without_cv2(self, monkeypatch):
        monkeypatch.setattr("perception.capture._HAS_CV2", False)
        with pytest.raises(ImportError, match="opencv-python"):
            WebcamCapture(device_id=0)

    def test_stats_dict(self):
        cap = WebcamCapture(device_id=0)
        stats = cap.stats
        assert stats["device_id"] == 0
        assert stats["target_fps"] == 10.0
        assert stats["frames_captured"] == 0
        assert stats["is_open"] is False


class TestScreenCaptureMock:
    """Test ScreenCapture configuration (no actual screen capture)."""

    def test_backend_detection(self):
        cap = ScreenCapture()
        assert cap._backend in ("mss", "pil")

    def test_stats_dict(self):
        cap = ScreenCapture(monitor={"left": 0, "top": 0, "width": 640, "height": 480})
        stats = cap.stats
        assert stats["target_fps"] == 10.0
        assert stats["frames_captured"] == 0


# ── End-to-end: tile flows into topology signal ──────────────

class TestVisionTopologyIntegration:
    """Vision tile → topology signal round-trip."""

    def test_tile_to_signal_to_tick(self):
        """A full round-trip: frame → tile → signal → topology tick."""
        from nerve.topology import NerveTopology

        encoder = VisionTileEncoder(model="random_projection", target_dim=512)
        frame = np.random.RandomState(99).randint(0, 256, (240, 320, 3), dtype=np.uint8)

        emb = encoder.encode_frame(frame)
        tile = encoder.to_tile(emb, source="webcam_0")
        signal = encoder.to_signal(emb, signal_dim=64)

        topo = NerveTopology(n_fibers=2, n_rooms=20, signal_dim=64)
        result = topo.tick(signals={"fiber-0": signal, "fiber-1": np.zeros(64, dtype=np.float32)})

        assert result.fibers_perceived == 2
        assert result.tick == 1
        assert result.latency_ms > 0

    def test_batch_to_topology(self):
        """Batch-encoded frames produce distinct signals."""
        encoder = VisionTileEncoder(model="random_projection", target_dim=512)
        frames = [
            np.random.RandomState(i).randint(0, 256, (240, 320, 3), dtype=np.uint8)
            for i in range(4)
        ]
        batch = encoder.encode_batch(frames)
        signals = [encoder.to_signal(e, signal_dim=64) for e in batch]

        # Distinct frames should produce distinct signals
        for i in range(len(signals) - 1):
            assert not np.allclose(signals[i], signals[i + 1], atol=1e-2)
