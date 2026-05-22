"""Tests for the Audio Tile Encoder pipeline.

These tests use the random_projection backend by default so they pass
without transformers, whisper, or heavy model downloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from perception import (
    AudioTileEncoder,
    MicrophoneCapture,
    SystemAudioCapture,
    AudioEncoderBackend,
)


# ── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def encoder():
    """AudioTileEncoder with deterministic random projection."""
    return AudioTileEncoder(model="random_projection", device="cpu", target_dim=512)


@pytest.fixture
def synthetic_sine():
    """A deterministic synthetic sine wave (1 second at 16kHz)."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


@pytest.fixture
def synthetic_silence():
    """A silent audio segment (zeros)."""
    return np.zeros(16000, dtype=np.float32)


# ── AudioTileEncoder ──────────────────────────────────────

class TestEncodeSegment:
    """Test encode_segment returns a 512-dim normalised embedding."""

    def test_returns_512_dim(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        assert emb.shape == (512,)

    def test_returns_float32(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        assert emb.dtype == np.float32

    def test_l2_normalised(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_different_audio_different_embeddings(self, encoder):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        a1 = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        a2 = np.sin(2 * np.pi * 880.0 * t).astype(np.float32)
        e1 = encoder.encode_segment(a1, sample_rate=sr)
        e2 = encoder.encode_segment(a2, sample_rate=sr)
        # Different inputs should not produce identical embeddings
        assert not np.allclose(e1, e2, atol=1e-4)

    def test_same_audio_same_embedding(self, encoder, synthetic_sine):
        e1 = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        e2 = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        assert np.allclose(e1, e2, atol=1e-5)

    def test_encode_int16(self, encoder):
        """int16 input should be normalised and produce valid embedding."""
        audio_int16 = (np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, 16000)) * 32767).astype(np.int16)
        emb = encoder.encode_segment(audio_int16, sample_rate=16000)
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        assert not np.any(np.isnan(emb))


class TestEncodeSilence:
    """Silence must not produce NaN or crash."""

    def test_no_nan(self, encoder, synthetic_silence):
        emb = encoder.encode_segment(synthetic_silence, sample_rate=16000)
        assert not np.any(np.isnan(emb))
        assert not np.any(np.isinf(emb))

    def test_valid_shape_and_dtype(self, encoder, synthetic_silence):
        emb = encoder.encode_segment(synthetic_silence, sample_rate=16000)
        assert emb.shape == (512,)
        assert emb.dtype == np.float32

    def test_l2_normalised_even_for_silence(self, encoder, synthetic_silence):
        emb = encoder.encode_segment(synthetic_silence, sample_rate=16000)
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-4) == 1.0


class TestEncodeBatch:
    """Test batch encoding returns correct shape and dtype."""

    def test_batch_shape(self, encoder):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        segments = [
            np.sin(2 * np.pi * (220.0 + i * 50) * t).astype(np.float32)
            for i in range(4)
        ]
        batch = encoder.encode_batch(segments, sample_rate=sr)
        assert batch.shape == (4, 512)
        assert batch.dtype == np.float32

    def test_batch_empty(self, encoder):
        batch = encoder.encode_batch([], sample_rate=16000)
        assert batch.shape == (0, 512)

    def test_batch_matches_individual(self, encoder, synthetic_sine):
        segments = [synthetic_sine for _ in range(3)]
        batch = encoder.encode_batch(segments, sample_rate=16000)
        single = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        for i in range(3):
            assert np.allclose(batch[i], single, atol=1e-5)


class TestToTile:
    """Test tile format matches NerveTopology contract."""

    def test_has_all_required_keys(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        tile = encoder.to_tile(emb, source="microphone_0")
        assert tile["type"] == "audio"
        assert tile["source"] == "microphone_0"
        assert isinstance(tile["embedding"], np.ndarray)
        assert "timestamp" in tile
        assert isinstance(tile["timestamp"], float)
        assert "metadata" in tile

    def test_metadata_keys(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        tile = encoder.to_tile(emb, source="system_audio")
        meta = tile["metadata"]
        assert meta["model"] == encoder.model_name
        assert meta["backend"] == encoder.backend
        assert meta["device"] == encoder.device
        assert "latency_ms" in meta
        assert "segment_count" in meta
        assert "sample_rate" in meta

    def test_transcript_in_metadata(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        tile = encoder.to_tile(emb, source="mic", transcript="hello world")
        assert tile["metadata"]["transcript"] == "hello world"

    def test_extra_metadata_merged(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        tile = encoder.to_tile(emb, source="test", extra_metadata={"event": "door_slam"})
        assert tile["metadata"]["event"] == "door_slam"


class TestToSignal:
    """Test projection from 512-dim to NerveTopology signal_dim."""

    def test_signal_shape(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        signal = encoder.to_signal(emb, signal_dim=64)
        assert signal.shape == (64,)
        assert signal.dtype == np.float32

    def test_signal_normalised(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        signal = encoder.to_signal(emb, signal_dim=64)
        norm = np.linalg.norm(signal)
        assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_signal_deterministic(self, encoder, synthetic_sine):
        emb = encoder.encode_segment(synthetic_sine, sample_rate=16000)
        s1 = encoder.to_signal(emb, signal_dim=64)
        s2 = encoder.to_signal(emb, signal_dim=64)
        assert np.allclose(s1, s2, atol=1e-5)

    def test_different_embeddings_different_signals(self, encoder):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        a1 = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        a2 = np.sin(2 * np.pi * 880.0 * t).astype(np.float32)
        e1 = encoder.encode_segment(a1, sample_rate=sr)
        e2 = encoder.encode_segment(a2, sample_rate=sr)
        s1 = encoder.to_signal(e1, signal_dim=64)
        s2 = encoder.to_signal(e2, signal_dim=64)
        assert not np.allclose(s1, s2, atol=1e-2)


class TestEncoderProperties:
    """Test encoder state properties."""

    def test_backend_is_string(self, encoder):
        assert isinstance(encoder.backend, str)
        assert encoder.backend == "RANDOM_PROJECTION"

    def test_latency_zero_before_encode(self, encoder):
        assert encoder.latency_ms == 0.0

    def test_latency_nonzero_after_encode(self, encoder, synthetic_sine):
        encoder.encode_segment(synthetic_sine, sample_rate=16000)
        assert encoder.latency_ms > 0.0

    def test_segment_count_increments(self, encoder, synthetic_sine):
        assert encoder._segment_count == 0
        encoder.encode_segment(synthetic_sine, sample_rate=16000)
        assert encoder._segment_count == 1
        encoder.encode_segment(synthetic_sine, sample_rate=16000)
        assert encoder._segment_count == 2


# ── Capture sources (mock, no real microphone needed) ──────────────

class TestMicrophoneCaptureMock:
    """Test MicrophoneCapture with mock (no hardware)."""

    def test_import_error_without_deps(self, monkeypatch):
        monkeypatch.setattr("perception.audio_capture._HAS_SOUNDDEVICE", False)
        monkeypatch.setattr("perception.audio_capture._HAS_PYAUDIO", False)
        with pytest.raises(ImportError, match="sounddevice"):
            MicrophoneCapture(device_id=0)

    def test_stats_dict(self, monkeypatch):
        monkeypatch.setattr("perception.audio_capture._HAS_SOUNDDEVICE", True)
        cap = MicrophoneCapture(device_id=0)
        stats = cap.stats
        assert stats["device_id"] == 0
        assert stats["backend"] == "sounddevice"
        assert stats["sample_rate"] == 16000
        assert stats["chunks_captured"] == 0
        assert stats["is_open"] is False


class TestSystemAudioCaptureMock:
    """Test SystemAudioCapture configuration (no actual capture)."""

    def test_backend_detection(self, monkeypatch):
        monkeypatch.setattr("perception.audio_capture._HAS_FFMPEG", True)
        monkeypatch.setattr("perception.audio_capture._HAS_PYAUDIO", True)
        cap = SystemAudioCapture()
        assert cap._backend == "ffmpeg"

    def test_stats_dict(self, monkeypatch):
        monkeypatch.setattr("perception.audio_capture._HAS_FFMPEG", True)
        cap = SystemAudioCapture(monitor="default")
        stats = cap.stats
        assert stats["backend"] == "ffmpeg"
        assert stats["monitor"] == "default"
        assert stats["sample_rate"] == 16000
        assert stats["chunks_captured"] == 0


# ── End-to-end: tile flows into topology signal ──────────────

class TestAudioTopologyIntegration:
    """Audio tile → topology signal round-trip."""

    def test_tile_to_signal_to_tick(self):
        """A full round-trip: audio → tile → signal → topology tick."""
        from nerve.topology import NerveTopology

        encoder = AudioTileEncoder(model="random_projection", target_dim=512)
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        emb = encoder.encode_segment(audio, sample_rate=sr)
        tile = encoder.to_tile(emb, source="microphone_0")
        signal = encoder.to_signal(emb, signal_dim=64)

        topo = NerveTopology(n_fibers=2, n_rooms=20, signal_dim=64)
        result = topo.tick(signals={"fiber-0": signal, "fiber-1": np.zeros(64, dtype=np.float32)})

        assert result.fibers_perceived == 2
        assert result.tick == 1
        assert result.latency_ms > 0

    def test_batch_to_topology(self):
        """Batch-encoded segments produce distinct signals."""
        encoder = AudioTileEncoder(model="random_projection", target_dim=512)
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        segments = [
            np.sin(2 * np.pi * (220.0 + i * 100) * t).astype(np.float32)
            for i in range(4)
        ]
        batch = encoder.encode_batch(segments, sample_rate=sr)
        signals = [encoder.to_signal(e, signal_dim=64) for e in batch]

        # Distinct audio should produce distinct signals
        for i in range(len(signals) - 1):
            assert not np.allclose(signals[i], signals[i + 1], atol=1e-2)
