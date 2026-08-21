"""Perception layer — Vision, audio, and capture for NerveTopology.

Provides:
  - VisionTileEncoder: 512-dim embedding encoder (multiple backends)
  - AudioTileEncoder: 512-dim audio embedding encoder (multiple backends)
  - WebcamCapture: OpenCV-based frame capture with FPS throttling
  - ScreenCapture: MSS/PIL-based screen region capture
  - MicrophoneCapture: sounddevice/pyaudio capture with frame dropping
  - SystemAudioCapture: ffmpeg/pyaudio loopback capture
  - CognitionLoop: observe → reason → act cycle for RoomGrid agents
  - AgentConfig: configuration for autonomous agent behavior
"""

from __future__ import annotations

__all__ = [
    "VisionTileEncoder",
    "EncoderBackend",
    "AudioTileEncoder",
    "AudioEncoderBackend",
    "WebcamCapture",
    "ScreenCapture",
    "MicrophoneCapture",
    "SystemAudioCapture",
    "CognitionLoop",
    "AgentConfig",
    "CognitionState",
]

from .vision_encoder import VisionTileEncoder, EncoderBackend
from .audio_encoder import AudioTileEncoder, AudioEncoderBackend
from .capture import WebcamCapture, ScreenCapture
from .audio_capture import MicrophoneCapture, SystemAudioCapture
from .cognition_loop import CognitionLoop, AgentConfig, CognitionState
