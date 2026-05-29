"""Log file rotation with size and time-based policies.

Rotates log files when they exceed size limits or time windows.
Supports retention policies and compression hooks. Used for fleet
log management and disk space control.

Usage:
    rotator = LogRotator(max_size=1024*1024, max_files=5)
    rotator.write("app.log", "log line\n")
    rotator.rotate("app.log")
"""
from __future__ import annotations

import gzip
import os
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class RotationPolicy:
    """Rotation policy configuration."""

    max_size: int = 10 * 1024 * 1024  # 10MB
    max_files: int = 5
    max_age_days: Optional[int] = None


class LogRotator:
    """
    Size and time-based log rotator.

    :param policy: RotationPolicy or defaults.
    :param compress: Whether to gzip rotated files.
    """

    def __init__(
        self,
        policy: Optional[RotationPolicy] = None,
        compress: bool = True,
    ):
        self._policy = policy or RotationPolicy()
        self._compress = compress
        self._stats: Dict[str, int] = {"rotations": 0, "bytes_written": 0}

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, path: str, data: str) -> None:
        """Append data to a log file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(data)
        self._stats["bytes_written"] += len(data.encode("utf-8"))

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def should_rotate(self, path: str) -> bool:
        """Check if a file should be rotated."""
        if not os.path.exists(path):
            return False
        size = os.path.getsize(path)
        if size >= self._policy.max_size:
            return True
        if self._policy.max_age_days:
            mtime = os.path.getmtime(path)
            age = (time.time() - mtime) / 86400
            if age >= self._policy.max_age_days:
                return True
        return False

    def rotate(self, path: str) -> str:
        """
        Rotate a log file.

        :returns: Path to the rotated file.
        """
        if not os.path.exists(path):
            return ""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        rotated = f"{path}.{timestamp}"
        shutil.move(path, rotated)
        if self._compress:
            compressed = f"{rotated}.gz"
            with open(rotated, "rb") as f_in:
                with gzip.open(compressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(rotated)
            rotated = compressed
        self._stats["rotations"] += 1
        self._cleanup_old(path)
        return rotated

    def _cleanup_old(self, base_path: str) -> None:
        """Remove excess rotated files."""
        pattern = os.path.basename(base_path)
        dirname = os.path.dirname(base_path) or "."
        files = [
            f
            for f in os.listdir(dirname)
            if f.startswith(pattern + ".") and f != pattern
        ]
        files.sort()
        while len(files) > self._policy.max_files:
            oldest = files.pop(0)
            os.remove(os.path.join(dirname, oldest))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_rotated(self, base_path: str) -> List[str]:
        """List rotated files for a base path."""
        pattern = os.path.basename(base_path)
        dirname = os.path.dirname(base_path) or "."
        return sorted(
            f
            for f in os.listdir(dirname)
            if f.startswith(pattern + ".") and f != pattern
        )

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"<LogRotator max_size={self._policy.max_size} max_files={self._policy.max_files}>"
