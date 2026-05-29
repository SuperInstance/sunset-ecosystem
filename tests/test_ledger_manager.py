import pytest
from fleet.ledger_manager import LedgerEntry, LedgerManager


class TestLedgerEntry:
    def test_compute_hash(self):
        e = LedgerEntry("e1", 0.0, "action", "actor", {}, "0000000000000000")
        h = e.compute_hash()
        assert len(h) == 16
        assert h != "0000000000000000"

    def test_to_dict(self):
        e = LedgerEntry("e1", 0.0, "action", "actor", {}, "prev")
        d = e.to_dict()
        assert d["entry_id"] == "e1"
        assert d["previous_hash"] == "prev"


class TestLedgerManager:
    def test_init(self):
        lm = LedgerManager()
        assert lm.fleet_node_id == "default"
        assert lm.get_stats()["total_entries"] == 0

    def test_append(self):
        lm = LedgerManager()
        e = lm.append("breed", "agent1", {"parents": ["a", "b"]})
        assert e.entry_id == "entry_0"
        assert e.action == "breed"
        assert e.hash is not None
        assert lm.get_stats()["total_entries"] == 1

    def test_chain(self):
        lm = LedgerManager()
        e1 = lm.append("action1", "actor", {})
        e2 = lm.append("action2", "actor", {})
        assert e2.previous_hash == e1.hash

    def test_get(self):
        lm = LedgerManager()
        e = lm.append("action", "actor", {})
        assert lm.get(e.entry_id) == e

    def test_get_missing(self):
        lm = LedgerManager()
        assert lm.get("missing") is None

    def test_get_range(self):
        lm = LedgerManager()
        for i in range(5):
            lm.append("action", "actor", {"i": i})
        entries = lm.get_range(1, 3)
        assert len(entries) == 2
        assert entries[0].data["i"] == 1

    def test_verify(self):
        lm = LedgerManager()
        for i in range(3):
            lm.append("action", "actor", {"i": i})
        assert lm.verify() is True

    def test_verify_tampered(self):
        lm = LedgerManager()
        lm.append("action", "actor", {})
        lm.append("action", "actor", {})
        # Tamper with an entry
        lm._entries[0].data = {"tampered": True}
        assert lm.verify() is False

    def test_get_stats(self):
        lm = LedgerManager()
        lm.append("breed", "a", {})
        lm.append("eval", "a", {})
        lm.append("breed", "b", {})
        stats = lm.get_stats()
        assert stats["total_entries"] == 3
        assert stats["actions"]["breed"] == 2

    def test_export_json(self):
        lm = LedgerManager()
        lm.append("action", "actor", {})
        j = lm.export_json()
        assert "entries" in j
        assert "action" in j

    def test_to_dict(self):
        lm = LedgerManager()
        lm.append("action", "actor", {})
        d = lm.to_dict()
        assert d["stats"]["total_entries"] == 1
