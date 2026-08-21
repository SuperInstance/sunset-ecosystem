import time
import pytest
from fleet.distributed_lock import LockToken, DistributedLock


class TestLockToken:
    def test_is_expired(self):
        t = LockToken(resource="r", holder="h", timestamp=0.0, ttl=1.0, token_id="x")
        assert t.is_expired() is True

    def test_is_not_expired(self):
        t = LockToken(
            resource="r", holder="h", timestamp=time.time(), ttl=60.0, token_id="x"
        )
        assert t.is_expired() is False

    def test_to_dict(self):
        t = LockToken(resource="r", holder="h", timestamp=0.0, ttl=1.0, token_id="x")
        d = t.to_dict()
        assert d["resource"] == "r"


class TestDistributedLock:
    def test_init(self):
        dl = DistributedLock()
        assert dl.fleet_node_id == "default"
        assert dl.list_locks() == []

    def test_acquire(self):
        dl = DistributedLock()
        token = dl.acquire("resource_1")
        assert token is not None
        assert token.resource == "resource_1"
        assert dl.is_locked("resource_1") is True

    def test_acquire_already_locked(self):
        dl = DistributedLock()
        dl.acquire("resource_1")
        token2 = dl.acquire("resource_1")
        assert token2 is None

    def test_release(self):
        dl = DistributedLock()
        token = dl.acquire("resource_1")
        assert dl.release("resource_1", token.token_id) is True
        assert dl.is_locked("resource_1") is False

    def test_release_wrong_token(self):
        dl = DistributedLock()
        dl.acquire("resource_1")
        assert dl.release("resource_1", "wrong_token") is False

    def test_release_not_found(self):
        dl = DistributedLock()
        assert dl.release("missing", "token") is False

    def test_renew(self):
        dl = DistributedLock()
        token = dl.acquire("resource_1", ttl=1.0)
        assert dl.renew("resource_1", token.token_id, extension=10.0) is True

    def test_renew_wrong_token(self):
        dl = DistributedLock()
        dl.acquire("resource_1")
        assert dl.renew("resource_1", "wrong") is False

    def test_renew_not_found(self):
        dl = DistributedLock()
        assert dl.renew("missing", "token") is False

    def test_get_holder(self):
        dl = DistributedLock()
        dl.acquire("resource_1", holder="node_42")
        assert dl.get_holder("resource_1") == "node_42"

    def test_get_holder_expired(self):
        dl = DistributedLock()
        dl.acquire("resource_1", ttl=0.001)
        time.sleep(0.01)
        assert dl.get_holder("resource_1") is None

    def test_list_locks(self):
        dl = DistributedLock()
        dl.acquire("r1")
        dl.acquire("r2")
        locks = dl.list_locks()
        assert len(locks) == 2

    def test_cleanup_expired(self):
        dl = DistributedLock()
        dl.acquire("r1", ttl=0.001)
        time.sleep(0.01)
        removed = dl.cleanup_expired()
        assert removed == 1
        assert dl.is_locked("r1") is False

    def test_get_stats(self):
        dl = DistributedLock()
        dl.acquire("r1")
        dl.acquire("r2")
        dl.release(
            "r1", dl._locks.get("r1", None).token_id if "r1" in dl._locks else ""
        )
        stats = dl.get_stats()
        assert stats["acquired"] == 2
        assert stats["released"] == 1

    def test_to_dict(self):
        dl = DistributedLock()
        dl.acquire("r1")
        d = dl.to_dict()
        assert d["node"] == "default"
        assert "stats" in d
