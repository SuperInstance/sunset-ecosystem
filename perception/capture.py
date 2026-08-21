"""Capture sources for vision tiles.

WebcamCapture — OpenCV VideoCapture with frame-dropping for target FPS.
ScreenCapture  — MSS or PIL.ImageGrab for screen regions.
"""

from __future__ import annotations

__all__ = ["WebcamCapture", "ScreenCapture"]

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Dependency detection ───────────────────────────────────

_HAS_CV2 = False
try:
    import cv2

    _HAS_CV2 = True
except Exception:
    pass

_HAS_MSS = False
try:
    import mss

    _HAS_MSS = True
except Exception:
    pass

_HAS_PIL = False
try:
    from PIL import ImageGrab

    _HAS_PIL = True
except Exception:
    pass


@dataclass
class CaptureConfig:
    """Shared capture configuration."""

    target_fps: float = 10.0
    width: int | None = None
    height: int | None = None
    fourcc: str | None = None


class WebcamCapture:
    """OpenCV webcam capture with frame dropping for target FPS.

    Args:
        device_id: Camera index (default 0).
        config: CaptureConfig with target_fps, optional resolution.
    """

    def __init__(self, device_id: int = 0, config: CaptureConfig | None = None):
        self.device_id = device_id
        self.config = config or CaptureConfig()
        self._cap: Any = None
        self._running = False
        self._last_frame_time = 0.0
        self._frame_interval = 1.0 / self.config.target_fps
        self._frame_count = 0
        self._dropped_count = 0

        if not _HAS_CV2:
            raise ImportError(
                "WebcamCapture requires opencv-python. "
                "Install: pip install opencv-python-headless"
            )

    # ── Lifecycle ───────────────────────────────────────────

    def open(self) -> bool:
        """Open the video capture device."""
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            logger.error("Failed to open webcam device %d", self.device_id)
            return False

        if self.config.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fourcc:
            self._cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.config.fourcc),
            )

        self._running = True
        self._last_frame_time = time.time()
        logger.info(
            "Webcam %d opened at ~%.1f FPS", self.device_id, self.config.target_fps
        )
        return True

    def close(self) -> None:
        """Release the video capture device."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(
            "Webcam %d closed (%d frames, %d dropped)",
            self.device_id,
            self._frame_count,
            self._dropped_count,
        )

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Capture ──────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Read a frame, dropping frames to maintain target FPS.

        Returns:
            RGB np.ndarray (H, W, 3) uint8, or None on failure.
        """
        if not self._running or self._cap is None:
            return None

        now = time.time()
        elapsed = now - self._last_frame_time

        # Frame dropping: consume buffered frames until interval elapsed
        while elapsed < self._frame_interval:
            ret = self._cap.grab()  # discard
            if not ret:
                return None
            elapsed = time.time() - self._last_frame_time
            self._dropped_count += 1

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        # Convert BGR → RGB
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._last_frame_time = time.time()
        self._frame_count += 1
        return frame

    def read_batch(self, n: int) -> list[np.ndarray]:
        """Read N frames at target FPS spacing (useful for batch encoding).

        Returns:
            List of RGB frames. May return fewer than N if capture fails.
        """
        frames: list[np.ndarray] = []
        for _ in range(n):
            frame = self.read()
            if frame is not None:
                frames.append(frame)
            else:
                break
        return frames

    @property
    def is_open(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "target_fps": self.config.target_fps,
            "frames_captured": self._frame_count,
            "frames_dropped": self._dropped_count,
            "is_open": self.is_open,
        }


class ScreenCapture:
    """Screen region capture using MSS (preferred) or PIL fallback.

    Args:
        monitor: Screen region as dict {left, top, width, height},
                 or integer monitor index for full monitor.
                 Default None captures the primary monitor.
        config: CaptureConfig with target_fps.
    """

    def __init__(
        self,
        monitor: dict[str, int] | int | None = None,
        config: CaptureConfig | None = None,
    ):
        self.monitor = monitor
        self.config = config or CaptureConfig()
        self._sct: Any = None
        self._last_frame_time = 0.0
        self._frame_interval = 1.0 / self.config.target_fps
        self._frame_count = 0
        self._dropped_count = 0

        # Prefer MSS, fallback to PIL
        if _HAS_MSS:
            self._backend = "mss"
        elif _HAS_PIL:
            self._backend = "pil"
        else:
            raise ImportError(
                "ScreenCapture requires mss or PIL. Install: pip install mss"
            )

    # ── Lifecycle ───────────────────────────────────────────

    def open(self) -> bool:
        if self._backend == "mss":
            self._sct = mss.mss()
        self._last_frame_time = time.time()
        logger.info("ScreenCapture opened (backend=%s)", self._backend)
        return True

    def close(self) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None
        logger.info(
            "ScreenCapture closed (%d frames, %d dropped)",
            self._frame_count,
            self._dropped_count,
        )

    def __enter__(self) -> "ScreenCapture":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Capture ──────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Capture a screen region, dropping to maintain target FPS.

        Returns:
            RGB np.ndarray (H, W, 3) uint8, or None on failure.
        """
        now = time.time()
        elapsed = now - self._last_frame_time

        # Simple sleep-based throttling (no frame buffer to drain like webcam)
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        if self._backend == "mss":
            return self._read_mss()
        return self._read_pil()

    def _read_mss(self) -> np.ndarray | None:
        if self._sct is None:
            return None
        try:
            if isinstance(self.monitor, int):
                mon = self._sct.monitors[self.monitor]
            elif isinstance(self.monitor, dict):
                mon = self.monitor
            else:
                mon = self._sct.monitors[1]  # primary
            sct_img = self._sct.grab(mon)
            # BGRA → RGB
            frame = np.array(sct_img)
            frame = (
                cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB) if _HAS_CV2 else frame[:, :, :3]
            )
        except Exception as exc:
            logger.warning("MSS grab failed: %s", exc)
            return None

        self._last_frame_time = time.time()
        self._frame_count += 1
        return frame

    def _read_pil(self) -> np.ndarray | None:
        try:
            if isinstance(self.monitor, dict):
                bbox = (
                    self.monitor["left"],
                    self.monitor["top"],
                    self.monitor["left"] + self.monitor["width"],
                    self.monitor["top"] + self.monitor["height"],
                )
            elif isinstance(self.monitor, int):
                # PIL doesn't support monitor index; use full screen
                bbox = None
            else:
                bbox = None
            img = ImageGrab.grab(bbox=bbox)
            frame = np.array(img.convert("RGB"))
        except Exception as exc:
            logger.warning("PIL grab failed: %s", exc)
            return None

        self._last_frame_time = time.time()
        self._frame_count += 1
        return frame

    def read_batch(self, n: int) -> list[np.ndarray]:
        """Read N frames at target FPS spacing."""
        frames: list[np.ndarray] = []
        for _ in range(n):
            frame = self.read()
            if frame is not None:
                frames.append(frame)
            else:
                break
        return frames

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "monitor": self.monitor,
            "target_fps": self.config.target_fps,
            "frames_captured": self._frame_count,
            "frames_dropped": self._dropped_count,
        }
