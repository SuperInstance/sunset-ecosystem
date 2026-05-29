import time
import pytest
from fleet.audit_trail import AuditEntry, AuditTrail


class TestAuditEntry:
    def test_init(self):
        e = AuditEntry(timestamp=0.0, action="test", actor="a", target="t")
        assert e.action == "test"
        assert e.prev_hash == ""

    def test_compute_hash(self):
        e = AuditEntry(timestamp=0.0, action="test", actor="a", target="t")
        h = e.compute_hash()
        assert len(h) == 16
        assert h != ""

    def test_seal(self):
        e = AuditEntry(timestamp=0.0, action="test", actor="a", target="t")
        e.seal()
        assert e.hash == e.compute_hash()

    def test_to_dict(self):
        e = AuditEntry(timestamp=0.0, action="test", actor="a", target="t")
        d = e.to_dict()
        assert d["action"] == "test"
        assert "entry_id" in d


class TestAuditTrail:
    def test_init(self):
        at = AuditTrail()
        assert at.entries == []
        assert at.fleet_node_id == "default"

    def test_log(self):
        at = AuditTrail()
        e = at.log("test", "actor1", "target1", {"x": 1})
        assert len(at.entries) == 1
        assert e.action == "test"
        assert e.hash != ""

    def test_log_breeding(self):
        at = AuditTrail()
        e = at.log_breeding(1, ["p1", "p2"], "child1")
        assert e.action == "breeding"
        assert e.details["generation"] == 1

    def test_log_deployment(self):
        at = AuditTrail()
        e = at.log_deployment("model_v1", "node_42")
        assert e.action == "deployment"
        assert e.details["model_id"] == "model_v1"

    def test_log_consensus(self):
        at = AuditTrail()
        e = at.log_consensus("prop_1", 7)
        assert e.action == "consensus"
        assert e.details["votes"] == 7

    def test_verify_chain(self):
        at = AuditTrail()
        at.log("a", "x", "y")
        at.log("b", "x", "z")
        assert at.verify_chain() is True

    def test_verify_chain_tampered(self):
        at = AuditTrail()
        at.log("a", "x", "y")
        at.log("b", "x", "z")
        # Tamper with an entry
        at.entries[0].action = "tampered"
        assert at.verify_chain() is False

    def test_get_entries_by_action(self):
        at = AuditTrail()
        at.log("breeding", "x", "y")
        at.log("deployment", "x", "z")
        at.log("breeding", "x", "w")
        entries = at.get_entries_by_action("breeding")
        assert len(entries) == 2

    def test_get_entries_by_actor(self):
        at = AuditTrail()
        at.log("test", "alice", "t1")
        at.log("test", "bob", "t2")
        entries = at.get_entries_by_actor("alice")
        assert len(entries) == 1

    def test_get_entries_by_target(self):
        at = AuditTrail()
        at.log("test", "x", "target_1")
        at.log("test", "x", "target_2")
        entries = at.get_entries_by_target("target_1")
        assert len(entries) == 1

    def test_get_time_range(self):
        at = AuditTrail()
        t0 = time.time()
        at.log("test", "x", "y")
        at.log("test", "x", "z")
        t1 = time.time()
        entries = at.get_time_range(t0, t1)
        assert len(entries) >= 2

    def test_get_stats(self):
        at = AuditTrail()
        at.log("test", "x", "y")
        at.log("test", "x", "z")
        s = at.get_stats()
        assert s["total_entries"] == 2
        assert s["chain_integrity"] is True

    def test_export_json(self):
        at = AuditTrail()
        at.log("test", "x", "y")
        j = at.export_json()
        assert "test" in j
        assert "chain_integrity" in j

    def test_to_dict(self):
        at = AuditTrail()
        at.log("test", "x", "y")
        d = at.to_dict()
        assert d["node"] == "default"
        assert "stats" in d
