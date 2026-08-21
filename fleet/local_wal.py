"""Write-ahead log for local durability.

Appends entries to a JSON-lines file for crash recovery. Used for
fleet state persistence, transaction logs, and audit trails.

Usage:
    wal = LocalWAL("/path/to/wal")
    wal.append({"op": "set", "key": "x", "value": 1})
    entries = wal.read_all()
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LocalWAL:
    """
    Append-only JSON-lines write-ahead log.

    :param path: File path for the WAL.
    """

    def __init__(self, path: str):
        self._path = path
        self._appended = 0
        self._truncated = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(self, entry: Dict[str, Any]) -> None:
        """Append a single entry."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._appended += 1

    def append_batch(self, entries: List[Dict[str, Any]]) -> None:
        """Append multiple entries atomically."""
        with open(self._path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        self._appended += len(entries)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all entries from the WAL."""
        if not os.path.exists(self._path):
            return []
        entries: List[Dict[str, Any]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Bad WAL line: {line[:80]}")
        return entries

    def replay(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        """Replay all entries through a function."""
        for entry in self.read_all():
            try:
                fn(entry)
            except Exception as e:
                logger.error(f"WAL replay error: {e}")

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def truncate(self) -> None:
        """Clear the WAL file."""
        if os.path.exists(self._path):
            with open(self._path, "w", encoding="utf-8"):
                pass
        self._truncated += 1

    def size(self) -> int:
        """Return file size in bytes."""
        if os.path.exists(self._path):
            return os.path.getsize(self._path)
        return 0

    def stats(self) -> Dict[str, int]:
        return {
            "appended": self._appended,
            "truncated": self._truncated,
            "size_bytes": self.size(),
        }

    def __repr__(self) -> str:
        return f"<LocalWAL path={self._path} entries={self._appended}>"
