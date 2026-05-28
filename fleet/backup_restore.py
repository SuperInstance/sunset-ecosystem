"""backup_restore.py — Fleet state snapshot and restore.

Provides:
1. Snapshot fleet state to dict/JSON
2. Incremental snapshots (delta from last)
3. Compression of state data
4. Restore from snapshot
5. Snapshot validation (hash verification)

Usage:
    br = BackupRestore()
    snapshot = br.snapshot({"agents": agents, "config": config})
    br.validate(snapshot)
    restored = br.restore(snapshot)
"""
from __future__ import annotations

__all__ = [
    "BackupRestore",
    "Snapshot",
]

import hashlib
import json
import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """A fleet state snapshot."""
    version: str
    timestamp: float
    checksum: str
    compressed: bool
    data: bytes
    metadata: dict[str, Any]


class BackupRestore:
    """Snapshot and restore fleet state."""

    def __init__(self, compress: bool = True, compression_level: int = 6) -> None:
        self._compress = compress
        self._compression_level = compression_level
        self._last_snapshot: dict[str, Any] | None = None

    def snapshot(
        self,
        state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Create a snapshot of fleet state."""
        payload = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()

        compressed = False
        data = payload
        if self._compress:
            data = zlib.compress(payload, level=self._compression_level)
            compressed = True

        snap = Snapshot(
            version="1.0",
            timestamp=time.time(),
            checksum=checksum,
            compressed=compressed,
            data=data,
            metadata=metadata or {},
        )
        self._last_snapshot = state
        return snap

    def restore(self, snapshot: Snapshot) -> dict[str, Any]:
        """Restore state from a snapshot."""
        data = snapshot.data
        if snapshot.compressed:
            try:
                data = zlib.decompress(data)
            except zlib.error as e:
                raise ValueError(f"Decompression failed: {e}")

        state = json.loads(data.decode("utf-8"))

        # Verify checksum
        payload = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
        actual_checksum = hashlib.sha256(payload).hexdigest()
        if actual_checksum != snapshot.checksum:
            raise ValueError(f"Checksum mismatch: expected {snapshot.checksum}, got {actual_checksum}")

        return state

    def validate(self, snapshot: Snapshot) -> bool:
        """Validate a snapshot without full restore."""
        try:
            data = snapshot.data
            if snapshot.compressed:
                data = zlib.decompress(data)
            actual_checksum = hashlib.sha256(data).hexdigest()
            return actual_checksum == snapshot.checksum
        except Exception as e:
            logger.warning(f"Snapshot validation failed: {e}")
            return False

    def delta(
        self,
        current_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Compute delta from last snapshot. Placeholder — full diff logic is complex."""
        if self._last_snapshot is None:
            return None
        # Simple shallow key diff
        delta: dict[str, Any] = {}
        for key, value in current_state.items():
            if key not in self._last_snapshot or self._last_snapshot[key] != value:
                delta[key] = value
        for key in self._last_snapshot:
            if key not in current_state:
                delta[key] = None  # Marked as removed
        return delta if delta else None

    def size_bytes(self, snapshot: Snapshot) -> int:
        """Get snapshot size in bytes."""
        return len(snapshot.data)

    def info(self, snapshot: Snapshot) -> dict[str, Any]:
        """Get snapshot metadata."""
        return {
            "version": snapshot.version,
            "timestamp": snapshot.timestamp,
            "checksum": snapshot.checksum,
            "compressed": snapshot.compressed,
            "size_bytes": len(snapshot.data),
            "metadata": snapshot.metadata,
        }

    def __repr__(self) -> str:
        return f"BackupRestore(compress={self._compress})"
