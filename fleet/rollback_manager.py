"""Rollback and versioning manager for fleet deployments.

Manages deployment versions, rollback checkpoints, and version
comparisons. Supports snapshot-based rollback, health-gated promotions,
and deployment history. Used for fleet deployment safety, canary
rollbacks, and disaster recovery.

Usage:
    rm = RollbackManager()
    rm.deploy("v1.0", {"status": "ok"})
    rm.deploy("v1.1", {"status": "fail"})
    assert rm.rollback() == "v1.0"
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class RollbackManager:
    """
    Deployment rollback manager with versioning.
    """

    def __init__(self, max_history: int = 10):
        self._max_history = max_history
        self._versions: List[Dict[str, Any]] = []
        self._current: Optional[str] = None

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def deploy(
        self,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        health_check: Optional[callable] = None,
    ) -> bool:
        """
        Record a new deployment.

        :param version: Version identifier.
        :param metadata: Deployment metadata.
        :param health_check: Optional callable for health validation.
        :returns: True if deployment succeeded.
        """
        if health_check and not health_check():
            return False

        self._versions.append(
            {
                "version": version,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
        )

        # Trim history
        if len(self._versions) > self._max_history:
            self._versions = self._versions[-self._max_history :]

        self._current = version
        return True

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, steps: int = 1) -> Optional[str]:
        """
        Rollback to a previous version.

        :param steps: Number of versions to roll back.
        :returns: Version rolled back to, or None if not possible.
        """
        if len(self._versions) <= steps:
            return None

        target = self._versions[-(steps + 1)]
        self._current = target["version"]
        return self._current

    def rollback_to(self, version: str) -> bool:
        """Rollback to a specific version."""
        for i, v in enumerate(self._versions):
            if v["version"] == version:
                # Truncate history after this version
                self._versions = self._versions[: i + 1]
                self._current = version
                return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def current(self) -> Optional[str]:
        """Get current deployed version."""
        return self._current

    def history(self) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return list(self._versions)

    def previous(self) -> Optional[str]:
        """Get previous version."""
        if len(self._versions) >= 2:
            return self._versions[-2]["version"]
        return None

    def versions(self) -> List[str]:
        """List all deployed versions."""
        return [v["version"] for v in self._versions]

    def is_rollback_available(self) -> bool:
        """Check if rollback is possible."""
        return len(self._versions) > 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "current": self._current,
            "history_size": len(self._versions),
            "max_history": self._max_history,
            "rollback_available": self.is_rollback_available(),
        }

    def __repr__(self) -> str:
        return (
            f"<RollbackManager current={self._current} history={len(self._versions)}>"
        )
