"""Tests for encryption.py — Symmetric encryption and HMAC.

Run: python3 -m pytest tests/test_encryption.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.encryption import (
    EncryptionVault,
    EncryptionError,
    IntegrityError,
    generate_key,
)


cryptography = pytest.importorskip("cryptography")


class TestEncryptionVault:
    def test_create(self):
        key = generate_key()
        vault = EncryptionVault(key)
        assert repr(vault) == "<EncryptionVault>"

    def test_create_bad_key_length(self):
        with pytest.raises(EncryptionError):
            EncryptionVault(b"short")

    def test_encrypt_decrypt_roundtrip(self):
        key = generate_key()
        vault = EncryptionVault(key)
        payload = b"breeding-agent-secrets"
        enc = vault.encrypt(payload)
        plain = vault.decrypt(enc)
        assert plain == payload

    def test_encrypt_with_aad(self):
        key = generate_key()
        vault = EncryptionVault(key)
        enc = vault.encrypt(b"data", associated_data=b"context")
        plain = vault.decrypt(enc, associated_data=b"context")
        assert plain == b"data"

    def test_decrypt_wrong_aad(self):
        key = generate_key()
        vault = EncryptionVault(key)
        enc = vault.encrypt(b"data", associated_data=b"correct")
        with pytest.raises(IntegrityError):
            vault.decrypt(enc, associated_data=b"wrong")

    def test_decrypt_tampered_ciphertext(self):
        key = generate_key()
        vault = EncryptionVault(key)
        enc = vault.encrypt(b"data")
        enc = type(enc)(
            ciphertext=enc.ciphertext[:-1] + b"\x00",
            nonce=enc.nonce,
            tag=enc.tag,
        )
        with pytest.raises(IntegrityError):
            vault.decrypt(enc)

    def test_sign_verify(self):
        key = generate_key()
        vault = EncryptionVault(key)
        sig = vault.sign(b"important")
        assert vault.verify(b"important", sig) is True
        assert vault.verify(b"tampered", sig) is False

    def test_rekey(self):
        key1 = generate_key()
        key2 = generate_key()
        vault = EncryptionVault(key1)
        enc = vault.encrypt(b"data")
        vault.rekey(key2)
        with pytest.raises(IntegrityError):
            vault.decrypt(enc)

    def test_stats(self):
        key = generate_key()
        vault = EncryptionVault(key)
        vault.encrypt(b"a")
        vault.encrypt(b"b")
        vault.decrypt(vault.encrypt(b"c"))
        stats = vault.stats()
        assert stats["encrypted"] == 3
        assert stats["decrypted"] == 1
