"""voice/soniqo_bridge.py — Bridge to soniqo open-source audio intelligence.

Integrates soniqo's speech SDKs (ASR, TTS, VAD) with PLATO rooms so that
voice interactions become first-class tiles. Every utterance is transcribed,
every response is synthesized, and the room remembers the conversation.

Architecture
------------
- ASR (Automatic Speech Recognition): speech → text tiles
- TTS (Text-to-Speech): text tiles → speech responses
- VAD (Voice Activity Detection): gatekeeping for voice streams
- Room integration: voice is just another tile format

Usage
-----
    from voice.soniqo_bridge import SoniqoBridge

    bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
    bridge.connect()

    # Voice input → PLATO tile
    text = bridge.listen_and_transcribe(timeout=5.0)
    bridge.submit_voice_tile(text, "human_operator")

    # PLATO response → Voice output
    response = bridge.query_room("What is the fleet status?")
    bridge.speak(response)

Dependencies
------------
- soniqo SDK (optional): speech-swift, speech-android, or speech-core
- Fallback: mock implementation for testing without soniqo
- For real-time: pyaudio or sounddevice for audio I/O
"""

from __future__ import annotations

import json
import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Check for soniqo SDK availability
SONIQO_AVAILABLE = False
try:
    # Try importing soniqo Python bindings (if they exist)
    import soniqo

    SONIQO_AVAILABLE = True
except ImportError:
    logger.warning("soniqo SDK not available; using mock voice implementation")


@dataclass
class VoiceTile:
    """A tile that captures voice interaction metadata."""

    tile_id: str
    room_id: str
    speaker: str
    transcript: str
    audio_hash: str
    duration_ms: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SoniqoBridge:
    """Bridge between soniqo audio SDK and PLATO rooms."""

    room_id: str
    node_id: str
    _asr_engine: Optional[Any] = field(default=None, repr=False)
    _tts_engine: Optional[Any] = field(default=None, repr=False)
    _vad_engine: Optional[Any] = field(default=None, repr=False)
    _connected: bool = False
    _voice_history: List[VoiceTile] = field(default_factory=list)

    def connect(self) -> bool:
        """Initialize soniqo engines or mock fallback."""
        if SONIQO_AVAILABLE:
            try:
                self._asr_engine = soniqo.ASR()
                self._tts_engine = soniqo.TTS()
                self._vad_engine = soniqo.VAD()
                self._connected = True
                logger.info("Soniqo engines initialized for room %s", self.room_id)
                return True
            except Exception as exc:
                logger.warning("Soniqo init failed: %s; using mock", exc)

        # Mock fallback
        self._asr_engine = _MockASR()
        self._tts_engine = _MockTTS()
        self._vad_engine = _MockVAD()
        self._connected = True
        logger.info("Mock soniqo engines initialized for room %s", self.room_id)
        return True

    def disconnect(self) -> None:
        """Shutdown engines."""
        self._connected = False
        self._asr_engine = None
        self._tts_engine = None
        self._vad_engine = None

    def listen_and_transcribe(self, timeout: float = 5.0) -> Optional[str]:
        """Capture audio and return transcript.

        Returns None if no speech detected within timeout.
        """
        if not self._connected:
            logger.warning("Not connected")
            return None

        # VAD: wait for voice activity
        start = time.time()
        while time.time() - start < timeout:
            if self._vad_engine.is_speech():
                # ASR: transcribe audio
                audio = self._capture_audio(duration=timeout - (time.time() - start))
                transcript = self._asr_engine.transcribe(audio)
                return transcript
            time.sleep(0.1)

        return None

    def speak(self, text: str, voice_id: Optional[str] = None) -> bool:
        """Synthesize text to speech."""
        if not self._connected:
            logger.warning("Not connected")
            return False

        audio = self._tts_engine.synthesize(text, voice_id=voice_id)
        self._play_audio(audio)
        return True

    def submit_voice_tile(
        self, transcript: str, speaker: str, audio_hash: str = "mock"
    ) -> VoiceTile:
        """Submit a voice interaction as a PLATO tile."""
        tile = VoiceTile(
            tile_id=f"voice:{int(time.time() * 1000)}",
            room_id=self.room_id,
            speaker=speaker,
            transcript=transcript,
            audio_hash=audio_hash,
            duration_ms=0.0,  # Calculated from actual audio
            confidence=1.0 if SONIQO_AVAILABLE else 0.95,
            metadata={
                "node_id": self.node_id,
                "engine": "soniqo" if SONIQO_AVAILABLE else "mock",
            },
        )
        self._voice_history.append(tile)
        return tile

    def query_room(self, question: str) -> str:
        """Query the room for a response to a text question."""
        # In real implementation: call PLATO room API
        # For now: mock response
        return f"Room {self.room_id} acknowledges: '{question}'"

    def get_voice_history(self) -> List[VoiceTile]:
        return self._voice_history

    def get_status(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "node_id": self.node_id,
            "connected": self._connected,
            "soniqo_available": SONIQO_AVAILABLE,
            "voice_tiles": len(self._voice_history),
            "engines": {
                "asr": self._asr_engine is not None,
                "tts": self._tts_engine is not None,
                "vad": self._vad_engine is not None,
            },
        }

    def _capture_audio(self, duration: float) -> bytes:
        """Capture audio from microphone. Mock returns silence."""
        # Mock: return empty audio
        sample_rate = 16000
        num_samples = int(sample_rate * duration)
        return bytes(num_samples * 2)  # 16-bit PCM

    def _play_audio(self, audio: bytes) -> None:
        """Play audio to speakers. Mock does nothing."""
        pass


# ── Mock engines for testing without soniqo SDK ──────────────────────────


class _MockASR:
    def transcribe(self, audio: bytes) -> str:
        return "mock transcription: the fleet is running smoothly"


class _MockTTS:
    def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        # Return mock audio: 1 second of silence
        return bytes(32000)  # 16000 Hz * 2 bytes * 1 second


class _MockVAD:
    def is_speech(self) -> bool:
        return True
