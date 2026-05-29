"""File system watcher with debounce for hot-reloading config.

Watches files or directories for changes and triggers callbacks with
debounce to avoid storming on batch changes (e.g., git pull).

Usage:
    watcher = FileWatcher(debounce_sec=0.5)
    watcher.watch("config.yaml", lambda path: reload_config(path))
    watcher.check()  # call periodically
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FileWatcher:
    """
    Debounced file watcher.

    :param debounce_sec: Minimum seconds between change events for same path.
    """

    def __init__(self, debounce_sec: float = 0.5):
        self._debounce = debounce_sec
        self._watches: Dict[str, Callable[[str], None]] = {}
        self._state: Dict[str, float] = {}  # path -> last mtime
        self._last_trigger: Dict[str, float] = {}  # path -> last callback time
        self._pending: Set[str] = set()

    # ------------------------------------------------------------------
    # Watch management
    # ------------------------------------------------------------------

    def watch(self, path: str, callback: Callable[[str], None]) -> None:
        """Start watching a file or directory."""
        self._watches[path] = callback
        self._state[path] = self._mtime(path)

    def unwatch(self, path: str) -> None:
        """Stop watching a path."""
        self._watches.pop(path, None)
        self._state.pop(path, None)
        self._last_trigger.pop(path, None)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def check(self) -> List[str]:
        """
        Poll all watched paths. Returns list of paths that changed.

        Call this periodically (e.g., in a heartbeat or loop).
        """
        changed: List[str] = []
        now = time.time()
        for path, callback in self._watches.items():
            current_mtime = self._mtime(path)
            last_mtime = self._state.get(path, 0.0)
            if current_mtime != last_mtime:
                self._state[path] = current_mtime
                # Debounce: skip if we just triggered this path
                last_trigger = self._last_trigger.get(path, 0.0)
                if now - last_trigger < self._debounce:
                    self._pending.add(path)
                    continue
                self._last_trigger[path] = now
                try:
                    callback(path)
                    changed.append(path)
                except Exception:
                    logger.exception("File watcher callback failed for %s", path)
        return changed

    def flush_pending(self) -> List[str]:
        """Process any pending debounced callbacks."""
        now = time.time()
        triggered: List[str] = []
        still_pending: Set[str] = set()
        for path in self._pending:
            last_trigger = self._last_trigger.get(path, 0.0)
            if now - last_trigger >= self._debounce:
                callback = self._watches.get(path)
                if callback:
                    try:
                        callback(path)
                        triggered.append(path)
                        self._last_trigger[path] = now
                    except Exception:
                        logger.exception("Pending callback failed for %s", path)
            else:
                still_pending.add(path)
        self._pending = still_pending
        return triggered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mtime(self, path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def watched_paths(self) -> List[str]:
        return list(self._watches.keys())

    def __repr__(self) -> str:
        return f"<FileWatcher paths={len(self._watches)}>"
