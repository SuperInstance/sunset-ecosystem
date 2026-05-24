"""Audio Tile Encoder — Compress audio streams into 512-dim embedding tiles.

Backends (in order of preference):
  1. Whisper via transformers (speech-to-text + embedding)
  2. Wav2Vec2 via transformers (audio-only embedding)
  3. CLAP via transformers (audio-text aligned)
  4. speechbrain / openai-whisper (fallback)
  5. ONNXRuntime (edge / Jetson)
  6. Random projection with FFT features (deterministic fallback, no deps)

All backends converge to a 512-dim float32 embedding that feeds into
NerveTopology as a first-class audio tile.
"""
from __future__ import annotations

__all__ = ["AudioTileEncoder", "AudioEncoderBackend"]

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Backend availability detection ───────────────────────────

_HAS_TRANSFORMERS = False
try:
    import transformers
    from transformers import AutoFeatureExtractor, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    pass

_HAS_SPEECHBRAIN = False
try:
    import speechbrain
    _HAS_SPEECHBRAIN = True
except Exception:
    pass

_HAS_OPENAI_WHISPER = False
try:
    import whisper as _whisper_module
    _HAS_OPENAI_WHISPER = True
except Exception:
    pass

_HAS_ONNX = False
try:
    import onnxruntime as ort
    _HAS_ONNX = True
except Exception:
    pass

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:
    pass

_HAS_SCIPY = False
try:
    import scipy.signal
    _HAS_SCIPY = True
except Exception:
    pass


class AudioEncoderBackend(Enum):
    """Available audio encoder backends, ordered by capability."""
    WHISPER = auto()         # transformers Whisper — STT + embedding
    WAV2VEC2 = auto()        # transformers Wav2Vec2 — audio-only
    CLAP = auto()            # transformers CLAP — audio-text aligned
    SPEECHBRAIN = auto()     # speechbrain wav2vec
    OPENAI_WHISPER = auto()  # openai-whisper
    ONNX = auto()            # onnxruntime edge inference
    RANDOM_PROJECTION = auto()  # deterministic fallback, no deps


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an audio model backend."""
    name: str
    embedding_dim: int
    sample_rate: int
    chunk_duration_sec: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "whisper": ModelSpec(
        name="openai/whisper-tiny",
        embedding_dim=384,      # whisper-tiny encoder output
        sample_rate=16000,
        chunk_duration_sec=30.0,
    ),
    "wav2vec2": ModelSpec(
        name="facebook/wav2vec2-base",
        embedding_dim=768,
        sample_rate=16000,
        chunk_duration_sec=10.0,
    ),
    "clap": ModelSpec(
        name="laion/clap-htsat-unfused",
        embedding_dim=512,
        sample_rate=48000,
        chunk_duration_sec=10.0,
    ),
    "speechbrain": ModelSpec(
        name="speechbrain/wav2vec2-base-superb-er",
        embedding_dim=768,
        sample_rate=16000,
        chunk_duration_sec=10.0,
    ),
    "openai_whisper": ModelSpec(
        name="tiny",
        embedding_dim=384,
        sample_rate=16000,
        chunk_duration_sec=30.0,
    ),
    "onnx": ModelSpec(
        name="audio_encoder.onnx",
        embedding_dim=512,
        sample_rate=16000,
        chunk_duration_sec=10.0,
    ),
    "random_projection": ModelSpec(
        name="random_projection",
        embedding_dim=512,
        sample_rate=16000,
        chunk_duration_sec=1.0,
    ),
}


class AudioTileEncoder:
    """Compress audio streams into 512-dim embedding tiles for NerveTopology.

    Args:
        model: Backend identifier — 'whisper', 'wav2vec2', 'clap',
               'speechbrain', 'openai_whisper', 'onnx',
               or 'random_projection' (deterministic fallback).
        device: Compute device — 'cpu', 'cuda', 'mps' (Apple Silicon).
        target_dim: Output embedding dimension (default 512).
    """

    def __init__(
        self,
        model: str = "whisper",
        device: str = "cpu",
        target_dim: int = 512,
    ):
        self.model_name = model
        self.device = device
        self.target_dim = target_dim
        self.spec = MODEL_SPECS.get(model, MODEL_SPECS["random_projection"])

        # Runtime state
        self._backend: AudioEncoderBackend | None = None
        self._processor: Any = None
        self._model: Any = None
        self._session: Any = None          # ONNX session
        self._projection: np.ndarray | None = None
        self._segment_count = 0
        self._latency_window: list[float] = []

        self._init_backend()

    # ── Backend initialisation ────────────────────────────────

    def _init_backend(self) -> None:
        """Auto-detect and initialise the best available backend."""
        requested = self.model_name.lower()

        preference_order = [
            ("whisper", AudioEncoderBackend.WHISPER, _HAS_TRANSFORMERS),
            ("wav2vec2", AudioEncoderBackend.WAV2VEC2, _HAS_TRANSFORMERS),
            ("clap", AudioEncoderBackend.CLAP, _HAS_TRANSFORMERS),
            ("speechbrain", AudioEncoderBackend.SPEECHBRAIN, _HAS_SPEECHBRAIN),
            ("openai_whisper", AudioEncoderBackend.OPENAI_WHISPER, _HAS_OPENAI_WHISPER),
            ("onnx", AudioEncoderBackend.ONNX, _HAS_ONNX),
            ("random_projection", AudioEncoderBackend.RANDOM_PROJECTION, True),
        ]

        ordered = sorted(
            preference_order,
            key=lambda t: 0 if t[0] == requested else 1,
        )

        for name, backend, available in ordered:
            if not available:
                continue
            try:
                self._setup_backend(backend)
                self._backend = backend
                logger.info(
                    "AudioTileEncoder initialised: backend=%s device=%s dim=%d",
                    backend.name, self.device, self.target_dim,
                )
                return
            except Exception as exc:
                logger.warning("Backend %s failed: %s", backend.name, exc)
                continue

        self._backend = AudioEncoderBackend.RANDOM_PROJECTION
        self._setup_random_projection()
        logger.info(
            "AudioTileEncoder fallback: backend=RANDOM_PROJECTION dim=%d",
            self.target_dim,
        )

    def _setup_backend(self, backend: AudioEncoderBackend) -> None:
        if backend == AudioEncoderBackend.WHISPER:
            self._setup_whisper()
        elif backend == AudioEncoderBackend.WAV2VEC2:
            self._setup_wav2vec2()
        elif backend == AudioEncoderBackend.CLAP:
            self._setup_clap()
        elif backend == AudioEncoderBackend.SPEECHBRAIN:
            self._setup_speechbrain()
        elif backend == AudioEncoderBackend.OPENAI_WHISPER:
            self._setup_openai_whisper()
        elif backend == AudioEncoderBackend.ONNX:
            self._setup_onnx()
        elif backend == AudioEncoderBackend.RANDOM_PROJECTION:
            self._setup_random_projection()

    def _setup_whisper(self) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        from transformers import WhisperModel, WhisperProcessor
        self._processor = WhisperProcessor.from_pretrained(self.spec.name)
        self._model = WhisperModel.from_pretrained(self.spec.name)
        self._model.eval()
        if _HAS_TORCH:
            self._model.to(self.device)
        self._maybe_build_projection(self.spec.embedding_dim)

    def _setup_wav2vec2(self) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        self._processor = Wav2Vec2Processor.from_pretrained(self.spec.name)
        self._model = Wav2Vec2Model.from_pretrained(self.spec.name)
        self._model.eval()
        if _HAS_TORCH:
            self._model.to(self.device)
        self._maybe_build_projection(self.spec.embedding_dim)

    def _setup_clap(self) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        from transformers import ClapModel, ClapProcessor
        self._processor = ClapProcessor.from_pretrained(self.spec.name)
        self._model = ClapModel.from_pretrained(self.spec.name)
        self._model.eval()
        if _HAS_TORCH:
            self._model.to(self.device)
        # CLAP already emits 512-dim for many variants
        self._maybe_build_projection(self.spec.embedding_dim)

    def _setup_speechbrain(self) -> None:
        if not _HAS_SPEECHBRAIN:
            raise ImportError("speechbrain not installed")
        from speechbrain.pretrained import EncoderClassifier
        self._model = EncoderClassifier.from_hparams(source=self.spec.name)
        self._maybe_build_projection(self.spec.embedding_dim)

    def _setup_openai_whisper(self) -> None:
        if not _HAS_OPENAI_WHISPER:
            raise ImportError("openai-whisper not installed")
        import whisper
        self._model = whisper.load_model(self.spec.name, device=self.device)
        self._maybe_build_projection(self.spec.embedding_dim)

    def _setup_onnx(self) -> None:
        if not _HAS_ONNX:
            raise ImportError("onnxruntime not installed")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.device == "cuda":
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            prov = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            self.spec.name, opts, providers=prov
        )
        self._input_name = self._session.get_inputs()[0].name

    def _setup_random_projection(self) -> None:
        """Deterministic FFT-feature random projection — no neural network needed."""
        rng = np.random.RandomState(42)
        # Feature vector = FFT magnitude bins (up to 8192) + basic stats
        feature_dim = 8192 + 8
        self._projection = rng.randn(feature_dim, self.target_dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)

    def _maybe_build_projection(self, in_dim: int) -> None:
        if in_dim != self.target_dim:
            rng = np.random.RandomState(in_dim)
            proj = rng.randn(in_dim, self.target_dim).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            self._projection = proj
            logger.debug("Built projection %d → %d", in_dim, self.target_dim)

    # ── Public API ────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend.name if self._backend else "none"

    @property
    def latency_ms(self) -> float:
        """Rolling average latency over the last 10 segments."""
        if not self._latency_window:
            return 0.0
        return float(np.mean(self._latency_window)) * 1000.0

    def encode_segment(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Encode a single audio segment (samples,) into a 512-dim float32 embedding.

        Args:
            audio: 1-D array of audio samples, float32 or int16.
            sample_rate: Sampling rate in Hz (default 16000).

        Returns:
            np.ndarray of shape (512,), dtype float32, L2-normalised.
        """
        t0 = time.perf_counter()
        embedding = self._encode_impl(audio, sample_rate)
        # L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        elapsed = time.perf_counter() - t0
        self._latency_window.append(elapsed)
        if len(self._latency_window) > 10:
            self._latency_window.pop(0)
        self._segment_count += 1

        return embedding.astype(np.float32)

    def encode_batch(
        self, segments: list[np.ndarray], sample_rate: int = 16000
    ) -> np.ndarray:
        """Batch encode for efficiency.

        Args:
            segments: List of 1-D audio sample arrays.
            sample_rate: Sampling rate in Hz (default 16000).

        Returns:
            np.ndarray of shape (N, 512), dtype float32.
        """
        if not segments:
            return np.zeros((0, self.target_dim), dtype=np.float32)

        if self._backend == AudioEncoderBackend.WHISPER and _HAS_TRANSFORMERS:
            return self._batch_whisper(segments, sample_rate)
        if self._backend == AudioEncoderBackend.WAV2VEC2 and _HAS_TRANSFORMERS:
            return self._batch_wav2vec2(segments, sample_rate)
        if self._backend == AudioEncoderBackend.CLAP and _HAS_TRANSFORMERS:
            return self._batch_clap(segments, sample_rate)

        # Fallback: loop
        return np.stack([self.encode_segment(s, sample_rate) for s in segments], axis=0)

    def to_tile(
        self,
        embedding: np.ndarray,
        source: str,
        transcript: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert embedding to a NerveTopology-compatible tile dict.

        Args:
            embedding: 512-dim np.ndarray from encode_segment / encode_batch.
            source: Identifier like 'microphone_0' or 'system_audio'.
            transcript: Optional transcribed text (if STT model).
            extra_metadata: Optional additional metadata merged in.

        Returns:
            Tile dict: {
                'type': 'audio',
                'source': str,
                'embedding': np.ndarray,
                'timestamp': float,
                'metadata': dict,
            }
        """
        meta: dict[str, Any] = {
            "model": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "latency_ms": self.latency_ms,
            "segment_count": self._segment_count,
            "sample_rate": self.spec.sample_rate,
        }
        if transcript is not None:
            meta["transcript"] = transcript
        if extra_metadata:
            meta.update(extra_metadata)

        return {
            "type": "audio",
            "source": source,
            "embedding": embedding,
            "timestamp": time.time(),
            "metadata": meta,
        }

    # ── Internal encoding implementations ────────────────────

    def _encode_impl(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1-D, got shape {audio.shape}")
        if audio.dtype != np.float32:
            # Normalise int16 to [-1, 1]
            if audio.dtype in (np.int16, np.int32):
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)

        if self._backend == AudioEncoderBackend.WHISPER:
            return self._encode_whisper(audio, sample_rate)
        if self._backend == AudioEncoderBackend.WAV2VEC2:
            return self._encode_wav2vec2(audio, sample_rate)
        if self._backend == AudioEncoderBackend.CLAP:
            return self._encode_clap(audio, sample_rate)
        if self._backend == AudioEncoderBackend.SPEECHBRAIN:
            return self._encode_speechbrain(audio, sample_rate)
        if self._backend == AudioEncoderBackend.OPENAI_WHISPER:
            return self._encode_openai_whisper(audio, sample_rate)
        if self._backend == AudioEncoderBackend.ONNX:
            return self._encode_onnx(audio, sample_rate)
        return self._encode_random_projection(audio, sample_rate)

    # ── Whisper (transformers) ──────────────────────────────

    def _encode_whisper(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        inputs = self._processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.encoder(inputs["input_features"])
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    def _batch_whisper(self, segments: list[np.ndarray], sample_rate: int) -> np.ndarray:
        import torch
        # Whisper expects mel spectrograms; batch by stacking features
        features = []
        for seg in segments:
            inputs = self._processor(seg, sampling_rate=sample_rate, return_tensors="pt")
            features.append(inputs["input_features"])
        batch_features = torch.cat(features, dim=0).to(self.device)
        with torch.no_grad():
            out = self._model.encoder(batch_features)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb.astype(np.float32)

    # ── Wav2Vec2 ────────────────────────────────────────────

    def _encode_wav2vec2(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        inputs = self._processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    def _batch_wav2vec2(self, segments: list[np.ndarray], sample_rate: int) -> np.ndarray:
        import torch
        # Pad to max length for batching
        max_len = max(len(s) for s in segments)
        padded = [np.pad(s, (0, max_len - len(s))) for s in segments]
        batch = np.stack(padded, axis=0)
        inputs = self._processor(batch, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb.astype(np.float32)

    # ── CLAP ────────────────────────────────────────────────

    def _encode_clap(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        inputs = self._processor(
            audios=audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_audio_features(**inputs)
        emb = out.cpu().numpy().flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    def _batch_clap(self, segments: list[np.ndarray], sample_rate: int) -> np.ndarray:
        import torch
        # CLAP processor handles batched audio natively
        inputs = self._processor(
            audios=segments, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_audio_features(**inputs)
        emb = out.cpu().numpy()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb.astype(np.float32)

    # ── SpeechBrain ─────────────────────────────────────────

    def _encode_speechbrain(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Resample if needed (speechbrain expects 16kHz)
        if sample_rate != 16000 and _HAS_SCIPY:
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sample_rate))
        audio_tensor = audio if _HAS_TORCH else audio
        if _HAS_TORCH:
            audio_tensor = torch.tensor(audio).unsqueeze(0).float()
        emb = self._model.encode_batch(audio_tensor).squeeze().cpu().numpy().flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    # ── OpenAI Whisper ──────────────────────────────────────

    def _encode_openai_whisper(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import whisper
        # Resample to 16kHz if needed
        if sample_rate != 16000 and _HAS_SCIPY:
            audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sample_rate))
        mel = whisper.log_mel_spectrogram(audio)
        with torch.no_grad() if _HAS_TORCH else contextlib.nullcontext():
            emb = self._model.encoder(mel.unsqueeze(0).to(self.device))
        emb = emb.mean(dim=-1).cpu().numpy().flatten() if _HAS_TORCH else emb
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    # ── ONNX ────────────────────────────────────────────────

    def _encode_onnx(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Simple preprocessing: pad/truncate to fixed length, normalise
        target_len = int(self.spec.sample_rate * self.spec.chunk_duration_sec)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        tensor = np.expand_dims(audio, axis=0).astype(np.float32)
        emb = self._session.run(None, {self._input_name: tensor})[0].flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    # ── Random projection (fallback) ──────────────────────────

    def _encode_random_projection(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Deterministic FFT-feature random projection."""
        # Resample to target sample rate if scipy available
        if sample_rate != self.spec.sample_rate and _HAS_SCIPY:
            audio = scipy.signal.resample(audio, int(len(audio) * self.spec.sample_rate / sample_rate))

        # Target 1 second of audio at spec sample rate
        target_len = self.spec.sample_rate
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
        elif len(audio) > target_len:
            audio = audio[:target_len]

        # FFT magnitude features
        n_fft = min(16384, len(audio))
        fft = np.fft.rfft(audio, n=n_fft)
        mag = np.abs(fft)

        # Take first 8192 bins (should cover most audible freq range)
        n_bins = 8192
        if len(mag) < n_bins:
            mag = np.pad(mag, (0, n_bins - len(mag)))
        else:
            mag = mag[:n_bins]

        # Basic stats
        stats = np.array([
            float(np.mean(audio)),
            float(np.std(audio)),
            float(np.max(audio)),
            float(np.min(audio)),
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.percentile(audio, 25)),
            float(np.percentile(audio, 75)),
        ], dtype=np.float32)

        features = np.concatenate([mag, stats])
        # features shape = (8192 + 8,) = 8200
        # projection shape = (8200, target_dim)
        emb = features @ self._projection
        return emb

    # ── Convenience: downsample to NerveTopology signal_dim ───

    def to_signal(
        self,
        embedding: np.ndarray,
        signal_dim: int = 64,
        seed: int | None = None,
    ) -> np.ndarray:
        """Project a 512-dim audio embedding down to a NerveTopology signal.

        Uses a deterministic random projection seeded by the embedding hash
        so the same audio always maps to the same signal.
        """
        if seed is None:
            seed = int(hashlib.md5(embedding.tobytes()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        proj = rng.randn(len(embedding), signal_dim).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        signal = embedding @ proj
        norm = np.linalg.norm(signal)
        if norm > 1e-8:
            signal /= norm
        return signal
