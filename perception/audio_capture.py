"""Capture sources for audio tiles.

MicrophoneCapture — sounddevice/pyaudio with frame-dropping for target latency.
SystemAudioCapture  — pyaudio loopback or ffmpeg for system audio.
"""
from __future__ import annotations

__all__ = ["MicrophoneCapture", "SystemAudioCapture", "AudioCaptureConfig"]

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Dependency detection ───────────────────────────────────

_HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except Exception:
    pass

_HAS_PYAUDIO = False
try:
    import pyaudio
    _HAS_PYAUDIO = True
except Exception:
    pass

_HAS_FFMPEG = False
try:
    import shutil
    if shutil.which("ffmpeg"):
        _HAS_FFMPEG = True
except Exception:
    pass


@dataclass
class AudioCaptureConfig:
    """Shared audio capture configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: float = 100.0   # target chunk size
    target_latency_ms: float = 100.0  # frame-dropping target
    dtype: str = "float32"

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000.0)


class MicrophoneCapture:
    """Microphone capture with frame dropping for target latency.

    Args:
        device_id: Input device index (default None = system default).
        config: AudioCaptureConfig with sample_rate, channels, chunk size.
    """

    def __init__(
        self,
        device_id: int | None = None,
        config: AudioCaptureConfig | None = None,
    ):
        self.device_id = device_id
        self.config = config or AudioCaptureConfig()
        self._stream: Any = None
        self._running = False
        self._last_chunk_time = 0.0
        self._chunk_interval = self.config.chunk_duration_ms / 1000.0
        self._chunk_count = 0
        self._dropped_count = 0
        self._backend: str | None = None

        if _HAS_SOUNDDEVICE:
            self._backend = "sounddevice"
        elif _HAS_PYAUDIO:
            self._backend = "pyaudio"
        else:
            raise ImportError(
                "MicrophoneCapture requires sounddevice or pyaudio. "
                "Install: pip install sounddevice"
            )

    # ── Lifecycle ───────────────────────────────────────────

    def open(self) -> bool:
        """Open the audio capture stream."""
        try:
            if self._backend == "sounddevice":
                self._open_sounddevice()
            else:
                self._open_pyaudio()
            self._running = True
            self._last_chunk_time = time.time()
            logger.info(
                "Microphone opened (backend=%s, sr=%d, channels=%d, chunk=%d samples)",
                self._backend,
                self.config.sample_rate,
                self.config.channels,
                self.config.chunk_samples,
            )
            return True
        except Exception as exc:
            logger.error("Failed to open microphone: %s", exc)
            return False

    def _open_sounddevice(self) -> None:
        self._stream = sd.InputStream(
            device=self.device_id,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            dtype=self.config.dtype,
            blocksize=self.config.chunk_samples,
        )
        self._stream.start()

    def _open_pyaudio(self) -> None:
        self._pa = pyaudio.PyAudio()
        fmt = (
            pyaudio.paFloat32
            if self.config.dtype == "float32"
            else pyaudio.paInt16
        )
        self._stream = self._pa.open(
            format=fmt,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_samples,
            input_device_index=self.device_id,
        )

    def close(self) -> None:
        """Release the audio capture stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._backend == "pyaudio" and hasattr(self, "_pa"):
            try:
                self._pa.terminate()
            except Exception:
                pass
        logger.info(
            "Microphone closed (%d chunks, %d dropped)",
            self._chunk_count,
            self._dropped_count,
        )

    def __enter__(self) -> "MicrophoneCapture":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Capture ──────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Read an audio chunk, dropping frames to maintain target latency.

        Returns:
            1-D np.ndarray of audio samples, or None on failure.
        """
        if not self._running or self._stream is None:
            return None

        now = time.time()
        elapsed = now - self._last_chunk_time

        # Frame dropping: consume buffered chunks until interval elapsed
        while elapsed < self._chunk_interval:
            self._drop_chunk()
            elapsed = time.time() - self._last_chunk_time
            self._dropped_count += 1
            if not self._running:
                return None

        chunk = self._read_chunk()
        if chunk is None:
            return None

        self._last_chunk_time = time.time()
        self._chunk_count += 1
        return chunk

    def _drop_chunk(self) -> None:
        """Discard one chunk without processing."""
        try:
            if self._backend == "sounddevice":
                self._stream.read(self.config.chunk_samples)
            else:
                self._stream.read(self.config.chunk_samples)
        except Exception:
            pass

    def _read_chunk(self) -> np.ndarray | None:
        try:
            if self._backend == "sounddevice":
                data, _ = self._stream.read(self.config.chunk_samples)
                # sounddevice returns (samples, channels)
                if data.ndim == 2:
                    data = data[:, 0] if self.config.channels == 1 else data.flatten()
                return np.asarray(data, dtype=np.float32)
            else:
                # pyaudio
                raw = self._stream.read(self.config.chunk_samples, exception_on_overflow=False)
                if self.config.dtype == "float32":
                    arr = np.frombuffer(raw, dtype=np.float32)
                else:
                    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if self.config.channels > 1:
                    arr = arr.reshape(-1, self.config.channels)[:, 0]
                return arr
        except Exception as exc:
            logger.warning("Audio read failed: %s", exc)
            return None

    def read_batch(self, n: int) -> list[np.ndarray]:
        """Read N chunks at target spacing.

        Returns:
            List of audio chunk arrays. May return fewer than N on failure.
        """
        chunks: list[np.ndarray] = []
        for _ in range(n):
            chunk = self.read()
            if chunk is not None:
                chunks.append(chunk)
            else:
                break
        return chunks

    @property
    def is_open(self) -> bool:
        return self._running and self._stream is not None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "backend": self._backend,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "chunk_duration_ms": self.config.chunk_duration_ms,
            "chunks_captured": self._chunk_count,
            "chunks_dropped": self._dropped_count,
            "is_open": self.is_open,
        }


class SystemAudioCapture:
    """System audio (loopback) capture using ffmpeg or pyaudio.

    Args:
        monitor: Audio device name or index for loopback.
        config: AudioCaptureConfig with sample_rate, channels, chunk size.
    """

    def __init__(
        self,
        monitor: str | int | None = None,
        config: AudioCaptureConfig | None = None,
    ):
        self.monitor = monitor
        self.config = config or AudioCaptureConfig()
        self._process: Any = None
        self._last_chunk_time = 0.0
        self._chunk_interval = self.config.chunk_duration_ms / 1000.0
        self._chunk_count = 0
        self._dropped_count = 0
        self._buffer = b""

        # Prefer ffmpeg (works on Linux via pulse/alsa loopback), fallback pyaudio
        if _HAS_FFMPEG:
            self._backend = "ffmpeg"
        elif _HAS_PYAUDIO:
            self._backend = "pyaudio"
        else:
            raise ImportError(
                "SystemAudioCapture requires ffmpeg or pyaudio. "
                "Install: apt-get install ffmpeg || pip install pyaudio"
            )

    # ── Lifecycle ───────────────────────────────────────────

    def open(self) -> bool:
        try:
            if self._backend == "ffmpeg":
                self._open_ffmpeg()
            else:
                self._open_pyaudio_loopback()
            self._last_chunk_time = time.time()
            logger.info(
                "System audio opened (backend=%s, sr=%d, channels=%d)",
                self._backend,
                self.config.sample_rate,
                self.config.channels,
            )
            return True
        except Exception as exc:
            logger.error("Failed to open system audio: %s", exc)
            return False

    def _open_ffmpeg(self) -> None:
        fmt = "f32le" if self.config.dtype == "float32" else "s16le"
        cmd = [
            "ffmpeg",
            "-f", "pulse",  # or alsa
            "-i", "default",
            "-ac", str(self.config.channels),
            "-ar", str(self.config.sample_rate),
            "-f", fmt,
            "-acodec", "pcm_" + fmt,
            "-",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _open_pyaudio_loopback(self) -> None:
        self._pa = pyaudio.PyAudio()
        fmt = (
            pyaudio.paFloat32
            if self.config.dtype == "float32"
            else pyaudio.paInt16
        )
        device_idx = self.monitor if isinstance(self.monitor, int) else None
        self._stream = self._pa.open(
            format=fmt,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_samples,
            input_device_index=device_idx,
        )

    def close(self) -> None:
        if self._backend == "ffmpeg" and self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                pass
            self._process = None
        elif self._backend == "pyaudio" and hasattr(self, "_stream"):
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            if hasattr(self, "_pa"):
                try:
                    self._pa.terminate()
                except Exception:
                    pass
        logger.info(
            "System audio closed (%d chunks, %d dropped)",
            self._chunk_count,
            self._dropped_count,
        )

    def __enter__(self) -> "SystemAudioCapture":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Capture ──────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Capture a system audio chunk, dropping to maintain target latency.

        Returns:
            1-D np.ndarray of audio samples, or None on failure.
        """
        now = time.time()
        elapsed = now - self._last_chunk_time

        if elapsed < self._chunk_interval:
            time.sleep(self._chunk_interval - elapsed)

        chunk = self._read_chunk_impl()
        if chunk is not None:
            self._last_chunk_time = time.time()
            self._chunk_count += 1
        return chunk

    def _read_chunk_impl(self) -> np.ndarray | None:
        try:
            if self._backend == "ffmpeg":
                return self._read_ffmpeg()
            return self._read_pyaudio()
        except Exception as exc:
            logger.warning("System audio read failed: %s", exc)
            return None

    def _read_ffmpeg(self) -> np.ndarray | None:
        if self._process is None or self._process.poll() is not None:
            return None
        bytes_per_sample = 4 if self.config.dtype == "float32" else 2
        frame_size = self.config.chunk_samples * self.config.channels * bytes_per_sample
        raw = self._process.stdout.read(frame_size)
        if len(raw) < frame_size:
            return None
        if self.config.dtype == "float32":
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self.config.channels > 1:
            arr = arr.reshape(-1, self.config.channels)[:, 0]
        return arr

    def _read_pyaudio(self) -> np.ndarray | None:
        raw = self._stream.read(self.config.chunk_samples, exception_on_overflow=False)
        if self.config.dtype == "float32":
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self.config.channels > 1:
            arr = arr.reshape(-1, self.config.channels)[:, 0]
        return arr

    def read_batch(self, n: int) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        for _ in range(n):
            chunk = self.read()
            if chunk is not None:
                chunks.append(chunk)
            else:
                break
        return chunks

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "monitor": self.monitor,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "chunk_duration_ms": self.config.chunk_duration_ms,
            "chunks_captured": self._chunk_count,
            "chunks_dropped": self._dropped_count,
        }
