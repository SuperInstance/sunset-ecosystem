"""Tests for memory_index.py — In-memory inverted index.

Run: python3 -m pytest tests/test_memory_index.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.memory_index import MemoryIndex


class TestMemoryIndex:
    def test_create(self):
        idx = MemoryIndex()
        assert idx.doc_count() == 0

    def test_add_and_search(self):
        idx = MemoryIndex()
        idx.add("doc-1", "hello world")
        idx.add("doc-2", "hello fleet")
        results = idx.search("hello")
        assert sorted(results) == ["doc-1", "doc-2"]

    def test_search_specific(self):
        idx = MemoryIndex()
        idx.add("doc-1", "hello world")
        idx.add("doc-2", "hello fleet")
        results = idx.search("fleet")
        assert results == ["doc-2"]

    def test_search_any(self):
        idx = MemoryIndex()
        idx.add("doc-1", "alpha beta")
        idx.add("doc-2", "beta gamma")
        results = idx.search_any("alpha gamma")
        assert sorted(results) == ["doc-1", "doc-2"]

    def test_remove(self):
        idx = MemoryIndex()
        idx.add("doc-1", "hello world")
        assert idx.remove("doc-1") is True
        assert idx.search("hello") == []
        assert idx.remove("missing") is False

    def test_stopwords_ignored(self):
        idx = MemoryIndex()
        idx.add("doc-1", "the quick brown fox")
        results = idx.search("the")
        assert results == []

    def test_doc_count(self):
        idx = MemoryIndex()
        idx.add("a", "hello")
        idx.add("b", "world")
        assert idx.doc_count() == 2

    def test_terms(self):
        idx = MemoryIndex()
        idx.add("a", "hello world")
        assert "hello" in idx.terms()
        assert "world" in idx.terms()

    def test_empty_search(self):
        idx = MemoryIndex()
        assert idx.search("anything") == []

    def test_repr(self):
        idx = MemoryIndex()
        idx.add("a", "hello")
        assert "MemoryIndex" in repr(idx)
