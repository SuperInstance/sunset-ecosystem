"""Secrets management with encryption stubs.

Manages secrets (API keys, passwords, tokens) with basic encryption
stubs. In production, integrates with a real key management system.
Used for fleet credential management and secure configuration.

Usage:
    mgr = SecretsManager()
    mgr.set_secret("api_key", "secret123")
    assert mgr.get_secret("api_key") == "secret123"
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, Optional


class SecretsManager:
    """
    In-memory secrets manager with integrity verification.

    :param master_key: Key for HMAC integrity (not real encryption).
    """

    def __init__(self, master_key: str = "default-key"):
        self._master = master_key.encode("utf-8")
        self._secrets: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def set_secret(self, name: str, value: str) -> None:
        """Store a secret."""
        self._secrets[name] = {
            "value": value,
            "hmac": self._hmac(value),
        }

    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret."""
        entry = self._secrets.get(name)
        if not entry:
            return None
        if not hmac.compare_digest(entry["hmac"], self._hmac(entry["value"])):
            raise ValueError(f"Secret tampered: {name}")
        return entry["value"]

    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        if name in self._secrets:
            del self._secrets[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def _hmac(self, value: str) -> str:
        return hmac.new(self._master, value.encode("utf-8"), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_secrets(self) -> list:
        return list(self._secrets.keys())

    def has_secret(self, name: str) -> bool:
        return name in self._secrets

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {"secrets": len(self._secrets)}

    def __repr__(self) -> str:
        return f"<SecretsManager secrets={len(self._secrets)}>"
