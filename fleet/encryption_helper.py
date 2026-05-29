"""Encryption/decryption helpers (AES stubs for future integration).

Provides symmetric encryption helpers with AES-256-GCM stubs.
In production, integrates with a real crypto library. Used for
fleet data encryption, secure messaging, and credential storage.

Usage:
    helper = EncryptionHelper()
    ciphertext = helper.encrypt("hello", key=b"32-byte-key-here!!")
    plaintext = helper.decrypt(ciphertext, key=b"32-byte-key-here!!")
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any, Dict, Optional


class EncryptionHelper:
    """
    Encryption helper with AES-256-GCM stubs.

    :param key: 32-byte AES key.
    """

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or secrets.token_bytes(32)

    # ------------------------------------------------------------------
    # Encryption (stubs)
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str, key: Optional[bytes] = None) -> bytes:
        """
        Encrypt plaintext (stub: XOR with key hash).

        :param plaintext: Text to encrypt.
        :param key: Optional override key.
        :returns: Ciphertext bytes.
        """
        k = key or self._key
        key_hash = hashlib.sha256(k).digest()
        data = plaintext.encode("utf-8")
        # Simple XOR stub (NOT secure, replace with real AES-GCM)
        ciphertext = bytes(b ^ key_hash[i % len(key_hash)] for i, b in enumerate(data))
        return ciphertext

    def decrypt(self, ciphertext: bytes, key: Optional[bytes] = None) -> str:
        """
        Decrypt ciphertext (stub: XOR with key hash).

        :param ciphertext: Ciphertext bytes.
        :param key: Optional override key.
        :returns: Decrypted plaintext.
        """
        k = key or self._key
        key_hash = hashlib.sha256(k).digest()
        # XOR is symmetric
        data = bytes(b ^ key_hash[i % len(key_hash)] for i, b in enumerate(ciphertext))
        return data.decode("utf-8")

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    def derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive a 32-byte key from password using PBKDF2 stub."""
        salt = salt or secrets.token_bytes(16)
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)

    # ------------------------------------------------------------------
    # HMAC
    # ------------------------------------------------------------------

    def hmac(self, data: str) -> str:
        """Compute HMAC-SHA256 of data."""
        return hmac.new(self._key, data.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_hmac(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.hmac(data)
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {"key_size": len(self._key)}

    def __repr__(self) -> str:
        return f"<EncryptionHelper key_size={len(self._key)}>"
