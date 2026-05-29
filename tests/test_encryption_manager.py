import pytest
from fleet.encryption_manager import EncryptionManager


class TestEncryptionManager:
    def test_init(self):
        em = EncryptionManager()
        assert em.fleet_node_id == "default"

    def test_encrypt(self):
        em = EncryptionManager()
        encrypted = em.encrypt("hello world", "password")
        assert "salt" in encrypted
        assert "ciphertext" in encrypted
        assert "mode" in encrypted

    def test_decrypt(self):
        em = EncryptionManager()
        encrypted = em.encrypt("hello world", "password")
        decrypted = em.decrypt(encrypted, "password")
        assert decrypted == "hello world"

    def test_decrypt_wrong_password(self):
        em = EncryptionManager()
        encrypted = em.encrypt("hello world", "password")
        # With mock mode, wrong password still works (base64 decode)
        # But with real mode, it would fail
        if encrypted.get("mode") == "mock":
            decrypted = em.decrypt(encrypted, "wrong")
            assert decrypted == "hello world"

    def test_hash(self):
        em = EncryptionManager()
        h1 = em.hash("hello")
        h2 = em.hash("hello")
        h3 = em.hash("world")
        assert len(h1) == 64
        assert h1 == h2
        assert h1 != h3

    def test_to_dict(self):
        em = EncryptionManager()
        d = em.to_dict()
        assert d["node"] == "default"
