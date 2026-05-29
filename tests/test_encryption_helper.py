"""Tests for encryption_helper.py — Encryption/decryption helpers.

Run: python3 -m pytest tests/test_encryption_helper.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.encryption_helper import EncryptionHelper


class TestEncryptionHelper:
    def test_create(self):
        helper = EncryptionHelper()
        assert helper.stats()["key_size"] == 32

    def test_encrypt_decrypt(self):
        helper = EncryptionHelper()
        plaintext = "hello world"
        key = b"32-byte-key-here!!not-really-32"
        ciphertext = helper.encrypt(plaintext, key=key)
        decrypted = helper.decrypt(ciphertext, key=key)
        assert decrypted == plaintext

    def test_default_key(self):
        helper = EncryptionHelper()
        plaintext = "hello"
        ciphertext = helper.encrypt(plaintext)
        decrypted = helper.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_derive_key(self):
        helper = EncryptionHelper()
        key = helper.derive_key("password", salt=b"salt" * 4)
        assert len(key) == 32

    def test_hmac(self):
        helper = EncryptionHelper(key=b"test-key" * 4)
        sig = helper.hmac("hello")
        assert helper.verify_hmac("hello", sig) is True
        assert helper.verify_hmac("tampered", sig) is False

    def test_repr(self):
        helper = EncryptionHelper()
        assert "EncryptionHelper" in repr(helper)
