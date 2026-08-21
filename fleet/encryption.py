"""Symmetric encryption and HMAC signing for fleet secrets.

Provides AES-256-GCM authenticated encryption and SHA-256 HMAC for
fleet-wide secret exchange and tamper detection.

Usage:
    vault = EncryptionVault(key=os.urandom(32))
    ciphertext, nonce = vault.encrypt(b"secret payload")
    plaintext = vault.decrypt(ciphertext, nonce)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTOGRAPHY = False

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    pass


class DecryptionError(EncryptionError):
    pass


class IntegrityError(EncryptionError):
    pass


@dataclass
class EncryptedPayload:
    """Container for encrypted data + nonce + HMAC."""

    ciphertext: bytes
    nonce: bytes
    tag: bytes


class EncryptionVault:
    """
    AES-256-GCM encryption vault with HMAC integrity checks.

    :param key: 32-byte master key.
    :param hmac_key: Optional separate key for HMAC (defaults to key[:16] + key[16:]).
    """

    def __init__(
        self,
        key: bytes,
        hmac_key: Optional[bytes] = None,
    ):
        if len(key) != 32:
            raise EncryptionError("Key must be 32 bytes")
        if not _HAS_CRYPTOGRAPHY:
            raise EncryptionError(
                "cryptography library required. pip install cryptography"
            )
        self._key = key
        self._hmac_key = hmac_key or hashlib.sha256(key).digest()
        self._stats: Dict[str, int] = {"encrypted": 0, "decrypted": 0}

    # ------------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------------

    def encrypt(
        self, plaintext: bytes, associated_data: Optional[bytes] = None
    ) -> EncryptedPayload:
        """Encrypt plaintext and return payload with HMAC tag."""
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        tag = self._hmac(ciphertext, nonce, associated_data)
        self._stats["encrypted"] += 1
        return EncryptedPayload(ciphertext=ciphertext, nonce=nonce, tag=tag)

    def decrypt(
        self,
        payload: EncryptedPayload,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """Decrypt and verify HMAC integrity."""
        expected_tag = self._hmac(payload.ciphertext, payload.nonce, associated_data)
        if not hmac.compare_digest(payload.tag, expected_tag):
            raise IntegrityError("HMAC verification failed")

        aesgcm = AESGCM(self._key)
        try:
            plaintext = aesgcm.decrypt(
                payload.nonce, payload.ciphertext, associated_data
            )
        except Exception as exc:
            raise DecryptionError(f"Decryption failed: {exc}") from exc

        self._stats["decrypted"] += 1
        return plaintext

    # ------------------------------------------------------------------
    # HMAC helpers
    # ------------------------------------------------------------------

    def _hmac(
        self,
        ciphertext: bytes,
        nonce: bytes,
        associated_data: Optional[bytes],
    ) -> bytes:
        mac = hmac.new(self._hmac_key, digestmod=hashlib.sha256)
        mac.update(nonce)
        mac.update(ciphertext)
        if associated_data:
            mac.update(associated_data)
        return mac.digest()

    def sign(self, data: bytes) -> bytes:
        """Standalone HMAC signature."""
        return hmac.new(self._hmac_key, data, hashlib.sha256).digest()

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify standalone HMAC signature."""
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def rekey(self, new_key: bytes, new_hmac_key: Optional[bytes] = None) -> None:
        """Rotate encryption key."""
        if len(new_key) != 32:
            raise EncryptionError("New key must be 32 bytes")
        self._key = new_key
        self._hmac_key = new_hmac_key or hashlib.sha256(new_key).digest()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        return "<EncryptionVault>"


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def generate_key() -> bytes:
    """Generate a new 32-byte key from os.urandom."""
    return os.urandom(32)
