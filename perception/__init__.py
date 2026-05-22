"""Perception layer — Vision and screen capture for NerveTopology.

Provides:
  - VisionTileEncoder: 512-dim embedding encoder (multiple backends)
  - WebcamCapture: OpenCV-based frame capture with FPS throttling
  - ScreenCapture: MSS/PIL-based screen region capture
"""
from __future__ import annotations

__all__ = [
    "VisionTileEncoder",
    "EncoderBackend",
    "WebcamCapture",
    "ScreenCapture",
]

from .vision_encoder import VisionTileEncoder, EncoderBackend
from .capture import WebcamCapture, ScreenCapture
