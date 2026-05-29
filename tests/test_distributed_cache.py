import time
import pytest
from fleet.distributed_cache import CacheEntry, DistributedCache


class TestCacheEntry:
    def test_is_expired(self):
        e = CacheEntry(key="k", value="v", timestamp=0.0, ttl=1.0)
        assert e.is_expired() is True

    def test_is_not_expired(self):
        e = CacheEntry(key="k", value="v", timestamp=time.time(), ttl=60.0)
        assert e.is_expired() is False

    def test_to_dict(self):
        e = CacheEntry(key="k", value="v", timestamp=0.0, ttl=1.0)
        d = e.to_dict()
        assert d["key"] == "k"


class TestDistributedCache:
    def test_init(self):
        cache = DistributedCache()
        assert cache.fleet_node_id == "default"
        assert cache.keys() == []

    def test_set_and_get(self):
        cache = DistributedCache()
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_get_missing(self):
        cache = DistributedCache()
        assert cache.get("missing") is None

    def test_get_expired(self):
        cache = DistributedCache(default_ttl=0.001)
        cache.set("k", "v")
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_delete(self):
        cache = DistributedCache()
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.get("k") is None

    def test_delete_missing(self):
        cache = DistributedCache()
        assert cache.delete("missing") is False

    def test_clear(self):
        cache = DistributedCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.keys() == []

    def test_keys(self):
        cache = DistributedCache()
        cache.set("a", 1)
        cache.set("b", 2)
        keys = cache.keys()
        assert sorted(keys) == ["a", "b"]

    def test_keys_cleanup_expired(self):
        cache = DistributedCache(default_ttl=0.001)
        cache.set("a", 1)
        cache.set("b", 2)
        time.sleep(0.01)
        assert cache.keys() == []

    def test_get_stats(self):
        cache = DistributedCache()
        cache.set("k", "v")
        cache.get("k")
        cache.get("k")
        cache.get("missing")
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_export_json(self):
        cache = DistributedCache()
        cache.set("k", "v")
        j = cache.export_json()
        assert "k" in j
        assert "stats" in j

    def test_to_dict(self):
        cache = DistributedCache()
        cache.set("k", "v")
        d = cache.to_dict()
        assert "stats" in d
