"""Tests for the Commit-Caster I2I Router."""

import hashlib
import hmac
import json
import time
import pytest

from fleet.commit_caster import CommitCaster, CommitEvent


class TestCommitEvent:
    def test_from_dict(self):
        d = {
            "repo": "SuperInstance/sunset-ecosystem",
            "commit": "abc123",
            "author": "Casey",
            "message": "Fix bug",
            "branch": "main",
            "timestamp": "2024-01-01T00:00:00Z",
            "files": ["a.py"],
        }
        ev = CommitEvent.from_dict(d)
        assert ev.repo == "SuperInstance/sunset-ecosystem"
        assert ev.commit == "abc123"

    def test_fingerprint(self):
        ev = CommitEvent("r", "c", "a", "m", "b", "t")
        assert ev.fingerprint() == "r:c"

    def test_to_dict_roundtrip(self):
        ev = CommitEvent("r", "c", "a", "m", "b", "t", ["f"])
        d = ev.to_dict()
        ev2 = CommitEvent.from_dict(d)
        assert ev2.fingerprint() == ev.fingerprint()


class TestCommitCaster:
    def test_init(self):
        cc = CommitCaster("secret")
        assert cc.get_stats()["received"] == 0

    def test_validate_good(self):
        cc = CommitCaster("secret")
        payload = b'{"repo":"r"}'
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        assert cc.validate(payload, sig) is True

    def test_validate_bad_secret(self):
        cc = CommitCaster("secret")
        payload = b'{"repo":"r"}'
        sig = "sha256=" + hmac.new(b"wrong", payload, hashlib.sha256).hexdigest()
        assert cc.validate(payload, sig) is False

    def test_validate_bad_prefix(self):
        cc = CommitCaster("secret")
        assert cc.validate(b"x", "badprefix") is False

    def test_receive_good(self):
        cc = CommitCaster("secret")
        payload = json.dumps(
            {
                "repo": "r",
                "commit": "c",
                "author": "a",
                "message": "m",
                "branch": "b",
                "timestamp": "t",
                "files": [],
            }
        ).encode()
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        ev = cc.receive(payload, sig)
        assert ev is not None
        assert cc.get_stats()["accepted"] == 1

    def test_receive_bad_signature(self):
        cc = CommitCaster("secret")
        payload = b'{"repo":"r"}'
        ev = cc.receive(payload, "sha256=bad")
        assert ev is None
        assert cc.get_stats()["rejected"] == 1

    def test_receive_bad_json(self):
        cc = CommitCaster("secret")
        payload = b"not json"
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        ev = cc.receive(payload, sig)
        assert ev is None

    def test_deduplication(self):
        cc = CommitCaster("secret")
        payload = json.dumps(
            {
                "repo": "r",
                "commit": "c",
                "author": "a",
                "message": "m",
                "branch": "b",
                "timestamp": "t",
                "files": [],
            }
        ).encode()
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        cc.receive(payload, sig)
        ev2 = cc.receive(payload, sig)
        assert ev2 is None
        assert cc.get_stats()["accepted"] == 1
        assert cc.get_stats()["rejected"] == 1

    def test_deduplication_window_expires(self):
        cc = CommitCaster("secret", window_sec=0.01)
        payload = json.dumps(
            {
                "repo": "r",
                "commit": "c",
                "author": "a",
                "message": "m",
                "branch": "b",
                "timestamp": "t",
                "files": [],
            }
        ).encode()
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        cc.receive(payload, sig)
        time.sleep(0.02)
        ev2 = cc.receive(payload, sig)
        assert ev2 is not None

    def test_broadcast(self):
        calls = []
        cc = CommitCaster("secret", mesh_broadcast=lambda d: calls.append(d))
        payload = json.dumps(
            {
                "repo": "r",
                "commit": "c",
                "author": "a",
                "message": "m",
                "branch": "b",
                "timestamp": "t",
                "files": [],
            }
        ).encode()
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        cc.receive(payload, sig)
        assert len(calls) == 1

    def test_broadcast_failure_queues(self):
        cc = CommitCaster(
            "secret",
            mesh_broadcast=lambda d: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        payload = json.dumps(
            {
                "repo": "r",
                "commit": "c",
                "author": "a",
                "message": "m",
                "branch": "b",
                "timestamp": "t",
                "files": [],
            }
        ).encode()
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        cc.receive(payload, sig)
        assert cc.get_stats()["queued"] == 1
        assert len(cc.get_queue()) == 1

    def test_flush_queue(self):
        calls = []
        cc = CommitCaster("secret")
        payload = json.dumps(
            {
                "repo": "r",
                "commit": "c",
                "author": "a",
                "message": "m",
                "branch": "b",
                "timestamp": "t",
                "files": [],
            }
        ).encode()
        sig = "sha256=" + hmac.new(b"secret", payload, hashlib.sha256).hexdigest()
        cc.receive(payload, sig)
        assert len(cc.get_queue()) == 1
        cc.mesh_broadcast = lambda d: calls.append(d)
        sent = cc.flush_queue()
        assert sent == 1
        assert len(cc.get_queue()) == 0

    def test_to_dict(self):
        cc = CommitCaster("secret")
        d = cc.to_dict()
        assert d["secret_set"] is True
        assert d["broadcast_set"] is False

    def test_stats(self):
        cc = CommitCaster("secret")
        s = cc.get_stats()
        assert "received" in s
        assert "accepted" in s
        assert "rejected" in s
        assert "queued" in s
        assert "queue_size" in s
