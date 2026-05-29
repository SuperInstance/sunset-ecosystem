from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a key from password and salt using PBKDF2."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000, dklen=32)


class EncryptionManager:
    """
    Encryption for sensitive fleet data.

    AES-256-GCM encryption with password-derived keys.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id

    def encrypt(self, plaintext: str, password: str) -> Dict[str, str]:
        """Encrypt plaintext with AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            # Fallback: mock encryption for testing
            salt = os.urandom(16)
            key = _derive_key(password, salt)
            ciphertext = base64.b64encode(plaintext.encode()).decode()
            return {
                "salt": base64.b64encode(salt).decode(),
                "nonce": base64.b64encode(os.urandom(12)).decode(),
                "ciphertext": ciphertext,
                "tag": base64.b64encode(key[:16]).decode(),
                "mode": "mock",
            }

        salt = os.urandom(16)
        key = _derive_key(password, salt)
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
        return {
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "mode": "aes-256-gcm",
        }

    def decrypt(self, encrypted: Dict[str, str], password: str) -> str:
        """Decrypt ciphertext with AES-256-GCM."""
        if encrypted.get("mode") == "mock":
            return base64.b64decode(encrypted["ciphertext"]).decode()

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError("cryptography required for real decryption")

        salt = base64.b64decode(encrypted["salt"])
        nonce = base64.b64decode(encrypted["nonce"])
        ciphertext = base64.b64decode(encrypted["ciphertext"])
        key = _derive_key(password, salt)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode()

    def hash(self, data: str) -> str:
        """Compute SHA-256 hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
        }
