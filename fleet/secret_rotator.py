"""Secret rotation with dual-key support and grace period.

Manages cryptographic secrets with rotation support, dual-key grace
periods, and expiration tracking. Used for fleet API key management,
token rotation, and credential lifecycle.

Usage:
    rotator = SecretRotator(default_ttl_sec=3600, grace_period_sec=300)
    rotator.set_secret("api-key", "secret-123")
    assert rotator.get_secret("api-key") == "secret-123"
    rotator.rotate("api-key", "new-secret-456")
    assert rotator.get_secret("api-key") == "new-secret-456"
    # Old secret still valid during grace period
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class SecretRotator:
    """
    Secret rotator with dual-key support and grace period.

    :param default_ttl_sec: Default time-to-live for secrets.
    :param grace_period_sec: Grace period after rotation where old secret is valid.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        default_ttl_sec: float = 3600.0,
        grace_period_sec: float = 300.0,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._default_ttl = default_ttl_sec
        self._grace_period = grace_period_sec
        self._clock = clock or time.time
        self._secrets: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Secret management
    # ------------------------------------------------------------------

    def set_secret(
        self,
        name: str,
        value: str,
        ttl_sec: Optional[float] = None,
    ) -> None:
        """
        Set a secret.

        :param name: Secret name.
        :param value: Secret value.
        :param ttl_sec: Optional TTL (uses default if None).
        """
        self._secrets[name] = {
            "current": value,
            "previous": None,
            "expires": self._clock() + (ttl_sec or self._default_ttl),
            "rotated_at": None,
        }

    def get_secret(self, name: str) -> Optional[str]:
        """
        Get current secret value.

        :param name: Secret name.
        :returns: Current secret or None if expired.
        """
        entry = self._secrets.get(name)
        if not entry:
            return None
        if entry["expires"] <= self._clock():
            return None
        return entry["current"]

    def get_previous(self, name: str) -> Optional[str]:
        """
        Get previous secret value (during grace period).

        :param name: Secret name.
        :returns: Previous secret or None.
        """
        entry = self._secrets.get(name)
        if not entry or not entry["previous"]:
            return None
        if entry["rotated_at"] is not None and (self._clock() - entry["rotated_at"]) > self._grace_period:
            return None
        return entry["previous"]

    def verify(self, name: str, value: str) -> bool:
        """
        Verify a secret value (current or during grace period).

        :param name: Secret name.
        :param value: Secret value to verify.
        :returns: True if valid.
        """
        if self.get_secret(name) == value:
            return True
        if self.get_previous(name) == value:
            return True
        return False

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate(self, name: str, new_value: str) -> bool:
        """
        Rotate a secret to a new value.

        :param name: Secret name.
        :param new_value: New secret value.
        :returns: True if rotated.
        """
        entry = self._secrets.get(name)
        if not entry:
            return False
        entry["previous"] = entry["current"]
        entry["current"] = new_value
        entry["rotated_at"] = self._clock()
        entry["expires"] = self._clock() + self._default_ttl
        return True

    def remove(self, name: str) -> bool:
        """Remove a secret."""
        if name in self._secrets:
            del self._secrets[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        return list(self._secrets.keys())

    def is_expired(self, name: str) -> bool:
        """Check if secret is expired."""
        entry = self._secrets.get(name)
        if not entry:
            return True
        return entry["expires"] <= self._clock()

    def time_to_expiry(self, name: str) -> Optional[float]:
        """Get time until secret expires."""
        entry = self._secrets.get(name)
        if not entry:
            return None
        remaining = entry["expires"] - self._clock()
        return max(0.0, remaining) if remaining > 0 else 0.0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        expired = sum(1 for name in self._secrets if self.is_expired(name))
        return {
            "secrets": len(self._secrets),
            "expired": expired,
            "active": len(self._secrets) - expired,
            "default_ttl": self._default_ttl,
            "grace_period": self._grace_period,
        }

    def __repr__(self) -> str:
        return f"<SecretRotator secrets={len(self._secrets)}>"
