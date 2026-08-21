"""Alert routing, deduplication, and suppression manager.

Routes alerts to appropriate channels, deduplicates by fingerprint, and
supports suppression windows. Used for fleet monitoring, incident
management, and on-call rotation.

Usage:
    alerts = AlertManager()
    alerts.register_channel("slack", lambda msg: print(msg))
    alerts.send("CPU high", severity="warning", channel="slack")
    assert alerts.stats()["sent"] == 1
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional


class AlertManager:
    """
    Alert manager with deduplication and suppression.

    :param suppression_sec: Default suppression window for duplicates.
    """

    def __init__(self, suppression_sec: float = 300.0):
        self._suppression = suppression_sec
        self._channels: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._recent: Dict[str, float] = {}  # fingerprint -> last sent time
        self._sent = 0
        self._suppressed = 0

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def register_channel(
        self, name: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register an alert channel."""
        self._channels[name] = handler

    def unregister_channel(self, name: str) -> bool:
        """Unregister an alert channel."""
        if name in self._channels:
            del self._channels[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Alert sending
    # ------------------------------------------------------------------

    def send(
        self,
        message: str,
        severity: str = "info",
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        suppression_sec: Optional[float] = None,
    ) -> bool:
        """
        Send an alert.

        :param message: Alert message.
        :param severity: Alert severity (info, warning, critical).
        :param channel: Target channel (defaults to all registered).
        :param metadata: Additional alert metadata.
        :param suppression_sec: Override suppression window.
        :returns: True if alert was sent (not suppressed).
        """
        fingerprint = self._fingerprint(message, severity)
        window = suppression_sec or self._suppression
        now = time.time()

        # Check suppression
        if fingerprint in self._recent:
            elapsed = now - self._recent[fingerprint]
            if elapsed < window:
                self._suppressed += 1
                return False

        payload = {
            "message": message,
            "severity": severity,
            "timestamp": now,
            "metadata": metadata or {},
        }

        if channel:
            handler = self._channels.get(channel)
            if handler:
                handler(payload)
                self._sent += 1
        else:
            for handler in self._channels.values():
                handler(payload)
            self._sent += 1

        self._recent[fingerprint] = now
        return True

    def _fingerprint(self, message: str, severity: str) -> str:
        """Generate alert fingerprint for deduplication."""
        return hashlib.md5(f"{severity}:{message}".encode()).hexdigest()

    # ------------------------------------------------------------------
    # Suppression management
    # ------------------------------------------------------------------

    def clear_suppression(self, message: str, severity: str = "info") -> bool:
        """Clear suppression for a specific alert."""
        fingerprint = self._fingerprint(message, severity)
        if fingerprint in self._recent:
            del self._recent[fingerprint]
            return True
        return False

    def clear_all_suppression(self) -> None:
        """Clear all suppression state."""
        self._recent.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def channels(self) -> List[str]:
        return list(self._channels.keys())

    def is_suppressed(self, message: str, severity: str = "info") -> bool:
        """Check if an alert is currently suppressed."""
        fingerprint = self._fingerprint(message, severity)
        if fingerprint not in self._recent:
            return False
        elapsed = time.time() - self._recent[fingerprint]
        return elapsed < self._suppression

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "channels": len(self._channels),
            "sent": self._sent,
            "suppressed": self._suppressed,
            "active_suppressions": len(self._recent),
        }

    def __repr__(self) -> str:
        return f"<AlertManager channels={len(self._channels)} sent={self._sent}>"
