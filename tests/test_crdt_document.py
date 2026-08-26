"""Tests for crdt_document.py — LWW-Element-Dict CRDT.

Run: python3 -m pytest tests/test_crdt_document.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.crdt_document import CRDTDocument


class TestCRDTDocument:
    def test_create(self):
        doc = CRDTDocument("node-1")
        assert doc.node_id == "node-1"
        assert len(doc.keys()) == 0

    def test_set_and_get(self):
        doc = CRDTDocument("node-1")
        doc.set("x", 42)
        assert doc.get("x") == 42

    def test_has(self):
        doc = CRDTDocument("node-1")
        doc.set("x", 42)
        assert doc.has("x") is True
        assert doc.has("missing") is False

    def test_delete(self):
        doc = CRDTDocument("node-1")
        doc.set("x", 42)
        doc.delete("x")
        assert doc.has("x") is False
        assert doc.get("x") is None

    def test_merge(self):
        doc1 = CRDTDocument("node-1")
        doc1.set("x", 1)
        doc2 = CRDTDocument("node-2")
        doc2.set("y", 2)
        doc1.merge(doc2)
        assert doc1.get("x") == 1
        assert doc1.get("y") == 2

    def test_merge_conflict_newer_wins(self):
        doc1 = CRDTDocument("node-1")
        doc1.set("x", "old", timestamp=1.0)
        doc2 = CRDTDocument("node-2")
        doc2.set("x", "new", timestamp=2.0)
        doc1.merge(doc2)
        assert doc1.get("x") == "new"

    def test_merge_conflict_older_loses(self):
        doc1 = CRDTDocument("node-1")
        doc1.set("x", "new", timestamp=2.0)
        doc2 = CRDTDocument("node-2")
        doc2.set("x", "old", timestamp=1.0)
        doc1.merge(doc2)
        assert doc1.get("x") == "new"

    def test_items(self):
        doc = CRDTDocument("node-1")
        doc.set("a", 1)
        doc.set("b", 2)
        assert doc.items() == {"a": 1, "b": 2}

    def test_serialization(self):
        doc = CRDTDocument("node-1")
        doc.set("x", 42)
        data = doc.to_dict()
        restored = CRDTDocument.from_dict("node-2", data)
        assert restored.get("x") == 42

    def test_repr(self):
        doc = CRDTDocument("node-1")
        doc.set("x", 1)
        assert "CRDTDocument" in repr(doc)
