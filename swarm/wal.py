"""WAL (Write-Ahead Log) append-only persistence for fleet knowledge.

Provides crash-safe, append-only storage for all fleet memory layers:
    - KnowledgePipeline document ingestion
    - FluxVectorTable agent DNA updates
    - JepaGridMemory temporal state
    - HardwareProfileIndex device telemetry

Design:
    - Append-only log: every write is a new entry, never overwrites
    - Checkpoints: periodic snapshots for fast recovery
    - Replay: rebuild state from WAL on startup
    - Segments: rotate files when size exceeds threshold

Usage::

    from swarm.wal import FleetWAL

    wal = FleetWAL(base_path="/data/fleet-wal")
    wal.append("knowledge", room="forge", doc_id="abc", text="...")
    wal.append("agent", agent_id=123, vector=[...], fitness=0.9)

    # On restart:
    wal.replay()  # Rebuilds all indices from log
"""

from __future__ import annotations

__all__ = ["FleetWAL", "WALEntry", "WALCheckpoint"]

import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

WAL_MAGIC = b"FLWAL\x00"  # Fleet WAL magic bytes
WAL_VERSION = 1
WAL_EXT = ".wal"
CHECKPOINT_EXT = ".chk"

DEFAULT_SEGMENT_SIZE = 64 * 1024 * 1024  # 64MB
DEFAULT_CHECKPOINT_INTERVAL = 1000  # entries


# ── Data Structures ───────────────────────────────────────────────

@dataclass(frozen=True)
class WALEntry:
    """Single WAL record."""

    layer: str  # "knowledge", "agent", "temporal", "hardware"
    timestamp: float
    sequence: int
    payload: dict[str, Any]  # layer-specific data
    checksum: str  # blake2b of (layer + seq + payload_json)


@dataclass
class WALCheckpoint:
    """Snapshot of state at a point in time."""

    sequence: int  # Up to which entry this checkpoint covers
    timestamp: float
    layer_states: dict[str, dict[str, Any]]  # layer → serialized state
    wal_files: list[str]  # Which WAL files are included


# ── Core WAL ────────────────────────────────────────────────────────

class FleetWAL:
    """Append-only write-ahead log for fleet memory.

    Args:
        base_path: Directory for WAL files.
        segment_size: Rotate segment when file exceeds this size.
        checkpoint_interval: Create checkpoint every N entries.
    """

    def __init__(
        self,
        base_path: str | Path,
        segment_size: int = DEFAULT_SEGMENT_SIZE,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.segment_size = segment_size
        self.checkpoint_interval = checkpoint_interval

        self._sequence: int = self._load_latest_sequence()
        self._current_segment: Path = self._current_segment_path()
        self._entries_since_checkpoint: int = 0
        self._writers: dict[str, Any] = {}  # layer → callback for replay

    # ── segment management ────────────────────────────────────

    def _segment_files(self) -> list[Path]:
        """List all WAL segment files, sorted."""
        return sorted(self.base_path.glob(f"*{WAL_EXT}"))

    def _current_segment_path(self) -> Path:
        """Get or create the current segment file."""
        files = self._segment_files()
        if not files:
            return self.base_path / f"00000001{WAL_EXT}"
        latest = files[-1]
        if latest.stat().st_size >= self.segment_size:
            # Rotate
            num = int(latest.stem) + 1
            return self.base_path / f"{num:08d}{WAL_EXT}"
        return latest

    def _load_latest_sequence(self) -> int:
        """Recover highest sequence number from existing WAL."""
        max_seq = 0
        for seg_file in self._segment_files():
            for entry in self._read_segment(seg_file):
                if entry.sequence > max_seq:
                    max_seq = entry.sequence
        return max_seq

    # ── entry I/O ─────────────────────────────────────────────

    def _compute_checksum(self, layer: str, sequence: int, payload: dict) -> str:
        data = f"{layer}:{sequence}:{json.dumps(payload, sort_keys=True)}"
        return hashlib.blake2b(data.encode(), digest_size=8).hexdigest()

    def _read_segment(self, path: Path) -> list[WALEntry]:
        """Read all entries from a segment file."""
        entries: list[WALEntry] = []
        if not path.exists():
            return entries

        with open(path, "rb") as f:
            # Check magic
            magic = f.read(len(WAL_MAGIC))
            if magic != WAL_MAGIC:
                logger.warning("Invalid WAL magic in %s", path)
                return entries

            # Read version
            version = struct.unpack("<H", f.read(2))[0]
            if version != WAL_VERSION:
                logger.warning("WAL version mismatch: %d vs %d", version, WAL_VERSION)
                return entries

            # Read entries
            while True:
                # Entry header: 4-byte length
                header = f.read(4)
                if len(header) < 4:
                    break
                length = struct.unpack("<I", header)[0]
                data = f.read(length)
                if len(data) < length:
                    logger.warning("Truncated entry in %s", path)
                    break

                try:
                    raw = json.loads(data.decode("utf-8"))
                    entry = WALEntry(
                        layer=raw["layer"],
                        timestamp=raw["timestamp"],
                        sequence=raw["sequence"],
                        payload=raw["payload"],
                        checksum=raw["checksum"],
                    )
                    # Verify checksum
                    expected = self._compute_checksum(entry.layer, entry.sequence, entry.payload)
                    if entry.checksum != expected:
                        logger.warning("Checksum mismatch at sequence %d", entry.sequence)
                        continue
                    entries.append(entry)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Corrupt entry in %s: %s", path, exc)
                    continue

        return entries

    def _write_entry(self, entry: WALEntry) -> None:
        """Append an entry to the current segment."""
        # Rotate if needed
        if self._current_segment.exists() and self._current_segment.stat().st_size >= self.segment_size:
            num = int(self._current_segment.stem) + 1
            self._current_segment = self.base_path / f"{num:08d}{WAL_EXT}"

        # Write magic + version on new file
        if not self._current_segment.exists():
            with open(self._current_segment, "wb") as f:
                f.write(WAL_MAGIC)
                f.write(struct.pack("<H", WAL_VERSION))

        # Serialize entry
        data = json.dumps(
            {
                "layer": entry.layer,
                "timestamp": entry.timestamp,
                "sequence": entry.sequence,
                "payload": entry.payload,
                "checksum": entry.checksum,
            },
            default=str,
        ).encode("utf-8")

        with open(self._current_segment, "ab") as f:
            f.write(struct.pack("<I", len(data)))
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    # ── public API ────────────────────────────────────────────

    def append(self, layer: str, **payload: Any) -> WALEntry:
        """Append a new entry to the WAL.

        Args:
            layer: Target memory layer ("knowledge", "agent", "temporal", "hardware").
            **payload: Layer-specific data.

        Returns:
            The created WALEntry.
        """
        self._sequence += 1
        entry = WALEntry(
            layer=layer,
            timestamp=time.time(),
            sequence=self._sequence,
            payload=payload,
            checksum=self._compute_checksum(layer, self._sequence, payload),
        )
        self._write_entry(entry)
        self._entries_since_checkpoint += 1

        logger.debug("WAL append #%d: %s", entry.sequence, layer)

        # Trigger checkpoint if needed
        if self._entries_since_checkpoint >= self.checkpoint_interval:
            self.checkpoint()

        return entry

    def register_layer(self, layer: str, callback: callable) -> None:
        """Register a callback for replaying entries to a layer.

        Args:
            layer: Layer name.
            callback: Function(entry: WALEntry) -> None.
        """
        self._writers[layer] = callback
        logger.info("Registered WAL replay handler for layer '%s'", layer)

    def replay(self, since_sequence: int = 0) -> int:
        """Replay all entries from a given sequence number.

        Args:
            since_sequence: Start from this sequence (default: 0 = all).

        Returns:
            Number of entries replayed.
        """
        count = 0
        for seg_file in self._segment_files():
            for entry in self._read_segment(seg_file):
                if entry.sequence > since_sequence:
                    handler = self._writers.get(entry.layer)
                    if handler:
                        handler(entry)
                    count += 1

        logger.info("WAL replay: %d entries from sequence %d", count, since_sequence)
        return count

    # ── checkpoints ───────────────────────────────────────────

    def checkpoint(self) -> WALCheckpoint:
        """Create a checkpoint snapshot.

        Captures current state from all registered layers and records
        the checkpoint. After this, older WAL segments can be archived.

        Returns:
            The created checkpoint.
        """
        # Gather state from all registered layers
        layer_states: dict[str, dict[str, Any]] = {}
        for layer, callback in self._writers.items():
            # Expect callback to have a .checkpoint() method
            if hasattr(callback, "checkpoint"):
                layer_states[layer] = callback.checkpoint()

        chk = WALCheckpoint(
            sequence=self._sequence,
            timestamp=time.time(),
            layer_states=layer_states,
            wal_files=[f.name for f in self._segment_files()],
        )

        # Serialize checkpoint
        chk_path = self.base_path / f"checkpoint_{self._sequence:012d}{CHECKPOINT_EXT}"
        with open(chk_path, "w") as f:
            json.dump(
                {
                    "sequence": chk.sequence,
                    "timestamp": chk.timestamp,
                    "layer_states": chk.layer_states,
                    "wal_files": chk.wal_files,
                },
                f,
                indent=2,
                default=str,
            )

        self._entries_since_checkpoint = 0
        logger.info("Checkpoint created at sequence %d → %s", chk.sequence, chk_path.name)
        return chk

    def load_checkpoint(self, sequence: Optional[int] = None) -> Optional[WALCheckpoint]:
        """Load the latest (or specified) checkpoint.

        Returns:
            Checkpoint if found, None otherwise.
        """
        chk_files = sorted(self.base_path.glob(f"*{CHECKPOINT_EXT}"))
        if not chk_files:
            return None

        target = chk_files[-1]  # Latest
        if sequence is not None:
            matches = [f for f in chk_files if f"{sequence:012d}" in f.name]
            if matches:
                target = matches[-1]

        with open(target) as f:
            raw = json.load(f)

        return WALCheckpoint(
            sequence=raw["sequence"],
            timestamp=raw["timestamp"],
            layer_states=raw["layer_states"],
            wal_files=raw["wal_files"],
        )

    # ── stats ─────────────────────────────────────────────────

    @property
    def segment_count(self) -> int:
        return len(self._segment_files())

    @property
    def total_entries(self) -> int:
        return self._sequence

    def __repr__(self) -> str:
        return (
            f"FleetWAL(segments={self.segment_count}, "
            f"entries={self.total_entries}, "
            f"base={self.base_path})"
        )
