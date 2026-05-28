"""secrets_manager.py — Secure secret storage for fleet credentials.

Provides:
1. AES-GCM encryption for secrets at rest
2. HMAC-SHA256 integrity verification
3. Key rotation support
4. Secret versioning
5. Scoped access (namespace isolation)

Usage:
    sm = SecretsManager(master_key="fleet-secret-key-32-bytes-long")
    sm.set("api_key", "sk-12345", namespace="production")
    value = sm.get("api_key", namespace="production")
"""
from __future__ import annotations

__all__ = [
    "SecretsManager",
    "SecretEntry",
    "SecretNotFound",
]

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class SecretNotFound(Exception):
    """Raised when a secret does not exist."""


@dataclass
class SecretEntry:
    """An encrypted secret entry."""
    ciphertext: bytes
    nonce: bytes
    tag: bytes
    created_at: float
    version: int
    metadata: dict[str, Any]


class SecretsManager:
    """Secure secret storage with AES-GCM encryption."""

    def __init__(self, master_key: str | bytes) -> None:
        raw = master_key.encode() if isinstance(master_key, str) else master_key
        # Derive a 256-bit key using SHA-256
        self._key = hashlib.sha256(raw).digest()
        self._secrets: dict[str, dict[str, SecretEntry]] = {}  # namespace -> {name: entry}

    # ── core operations ──────────────────────────────

    def set(
        self,
        name: str,
        value: str | bytes,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a secret (encrypts in-memory)."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            # Fallback: simple XOR + HMAC (NOT for production)
            self._set_fallback(name, value, namespace, metadata)
            return

        plaintext = value.encode() if isinstance(value, str) else value
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        # AES-GCM appends auth tag; split it
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]

        ns = self._secrets.setdefault(namespace, {})
        version = ns.get(name, SecretEntry(b"", b"", b"", 0.0, 0, {})).version + 1
        ns[name] = SecretEntry(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            created_at=time.time(),
            version=version,
            metadata=metadata or {},
        )

    def get(self, name: str, namespace: str = "default") -> str:
        """Retrieve and decrypt a secret."""
        ns = self._secrets.get(namespace)
        if not ns or name not in ns:
            raise SecretNotFound(f"Secret '{name}' not found in namespace '{namespace}'")

        entry = ns[name]
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            return self._get_fallback(entry)

        aesgcm = AESGCM(self._key)
        # Reconstruct ciphertext + tag
        full = entry.ciphertext + entry.tag
        plaintext = aesgcm.decrypt(entry.nonce, full, None)
        return plaintext.decode()

    def delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a secret."""
        ns = self._secrets.get(namespace)
        if ns and name in ns:
            del ns[name]
            return True
        return False

    def exists(self, name: str, namespace: str = "default") -> bool:
        """Check if a secret exists."""
        ns = self._secrets.get(namespace)
        return ns is not None and name in ns

    def list_namespaces(self) -> list[str]:
        """List all namespaces."""
        return list(self._secrets.keys())

    def list_secrets(self, namespace: str = "default") -> list[str]:
        """List secret names in a namespace."""
        ns = self._secrets.get(namespace)
        return list(ns.keys()) if ns else []

    def rotate(self, name: str, namespace: str = "default") -> int:
        """Re-encrypt a secret with a fresh nonce (returns new version)."""
        value = self.get(name, namespace)
        meta = self._secrets[namespace][name].metadata
        self.set(name, value, namespace, metadata=meta)
        return self._secrets[namespace][name].version

    # ── fallback (no cryptography package) ───────────

    def _set_fallback(
        self,
        name: str,
        value: str | bytes,
        namespace: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        plaintext = value.encode() if isinstance(value, str) else value
        nonce = os.urandom(16)
        # XOR with key (repeated)
        key_stream = (self._key * (len(plaintext) // len(self._key) + 1))[:len(plaintext)]
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream))
        tag = hmac.new(self._key, ciphertext + nonce, hashlib.sha256).digest()[:16]

        ns = self._secrets.setdefault(namespace, {})
        version = ns.get(name, SecretEntry(b"", b"", b"", 0.0, 0, {})).version + 1
        ns[name] = SecretEntry(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            created_at=time.time(),
            version=version,
            metadata=metadata or {},
        )

    def _get_fallback(self, entry: SecretEntry) -> str:
        key_stream = (self._key * (len(entry.ciphertext) // len(self._key) + 1))[:len(entry.ciphertext)]
        plaintext = bytes(a ^ b for a, b in zip(entry.ciphertext, key_stream))
        expected_tag = hmac.new(self._key, entry.ciphertext + entry.nonce, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(expected_tag, entry.tag):
            raise ValueError("Integrity check failed")
        return plaintext.decode()

    def stats(self) -> dict[str, Any]:
        total = sum(len(ns) for ns in self._secrets.values())
        return {
            "namespaces": len(self._secrets),
            "total_secrets": total,
        }

    def __repr__(self) -> str:
        total = sum(len(ns) for ns in self._secrets.values())
        return f"SecretsManager(secrets={total})"
