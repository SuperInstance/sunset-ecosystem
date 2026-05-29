import time
import pytest
from fleet.secret_manager import Secret, SecretManager


class TestSecret:
    def test_is_expired(self):
        s = Secret("name", "value", 0.0, expires_at=time.time() - 1)
        assert s.is_expired() is True

    def test_is_not_expired(self):
        s = Secret("name", "value", 0.0, expires_at=time.time() + 100)
        assert s.is_expired() is False

    def test_to_dict(self):
        s = Secret("name", "value", 0.0, {"meta": 1})
        d = s.to_dict()
        assert d["name"] == "name"
        assert "value" not in d


class TestSecretManager:
    def test_init(self):
        sm = SecretManager()
        assert sm.fleet_node_id == "default"
        assert sm.get_stats()["total_secrets"] == 0

    def test_set_and_get(self):
        sm = SecretManager()
        sm.set("key", "secret-value")
        assert sm.get("key") == "secret-value"

    def test_get_missing(self):
        sm = SecretManager()
        assert sm.get("missing") is None

    def test_get_expired(self):
        sm = SecretManager()
        sm.set("key", "value", ttl_seconds=0.001)
        time.sleep(0.01)
        assert sm.get("key") is None

    def test_delete(self):
        sm = SecretManager()
        sm.set("key", "value")
        assert sm.delete("key") is True
        assert sm.get("key") is None
        assert sm.delete("key") is False

    def test_list_names(self):
        sm = SecretManager()
        sm.set("a", "1")
        sm.set("b", "2")
        names = sm.list_names()
        assert sorted(names) == ["a", "b"]

    def test_get_metadata(self):
        sm = SecretManager()
        sm.set("key", "value", metadata={"env": "prod"})
        meta = sm.get_metadata("key")
        assert meta["env"] == "prod"

    def test_get_metadata_missing(self):
        sm = SecretManager()
        assert sm.get_metadata("missing") is None

    def test_get_stats(self):
        sm = SecretManager()
        sm.set("a", "1")
        sm.set("b", "2")
        sm.get("a")
        stats = sm.get_stats()
        assert stats["total_secrets"] == 2
        assert stats["access_count"] == 1

    def test_to_dict(self):
        sm = SecretManager()
        sm.set("key", "value")
        d = sm.to_dict()
        assert d["stats"]["total_secrets"] == 1
