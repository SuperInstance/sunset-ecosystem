"""Backup scheduling and rotation manager.

Manages backup schedules, retention policies, and rotation. Used
for fleet data protection, point-in-time recovery, and compliance.

Usage:
    mgr = BackupManager()
    mgr.add_schedule("daily", interval_sec=86400, retention=7)
    mgr.add_schedule("weekly", interval_sec=604800, retention=4)
    due = mgr.due_schedules()
    mgr.record_backup("daily", success=True)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class BackupManager:
    """
    Backup schedule and rotation manager.

    :param clock: Optional clock function for testing.
    """

    def __init__(self, clock: Optional[callable] = None):
        self._schedules: Dict[str, Dict[str, Any]] = {}
        self._backups: Dict[str, List[Dict[str, Any]]] = {}
        self._clock = clock or time.time

    # ------------------------------------------------------------------
    # Schedule management
    # ------------------------------------------------------------------

    def add_schedule(
        self,
        name: str,
        interval_sec: float,
        retention: int,
    ) -> None:
        """Register a backup schedule."""
        self._schedules[name] = {
            "interval_sec": interval_sec,
            "retention": retention,
            "last_run": 0,
        }
        if name not in self._backups:
            self._backups[name] = []

    def remove_schedule(self, name: str) -> bool:
        """Remove a schedule."""
        if name not in self._schedules:
            return False
        del self._schedules[name]
        del self._backups[name]
        return True

    # ------------------------------------------------------------------
    # Backup tracking
    # ------------------------------------------------------------------

    def record_backup(self, name: str, success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a completed backup."""
        if name not in self._schedules:
            raise ValueError(f"Unknown schedule: {name}")
        self._backups[name].append({
            "timestamp": self._clock(),
            "success": success,
            "metadata": metadata or {},
        })
        self._schedules[name]["last_run"] = self._clock()
        self._rotate(name)

    def _rotate(self, name: str) -> None:
        """Enforce retention policy."""
        retention = self._schedules[name]["retention"]
        backups = self._backups[name]
        if len(backups) > retention:
            self._backups[name] = backups[-retention:]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def due_schedules(self) -> List[str]:
        """Get schedules that are due for backup."""
        now = self._clock()
        due: List[str] = []
        for name, schedule in self._schedules.items():
            if now - schedule["last_run"] >= schedule["interval_sec"]:
                due.append(name)
        return due

    def last_backup(self, name: str) -> Optional[Dict[str, Any]]:
        """Get most recent backup for a schedule."""
        backups = self._backups.get(name, [])
        return backups[-1] if backups else None

    def backup_history(self, name: str) -> List[Dict[str, Any]]:
        """Get all retained backups for a schedule."""
        return list(self._backups.get(name, []))

    def success_rate(self, name: str) -> float:
        """Calculate success rate for a schedule."""
        backups = self._backups.get(name, [])
        if not backups:
            return 0.0
        successes = sum(1 for b in backups if b["success"])
        return successes / len(backups)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "schedules": len(self._schedules),
            "total_backups": sum(len(b) for b in self._backups.values()),
        }

    def __repr__(self) -> str:
        return f"<BackupManager schedules={len(self._schedules)}>"
