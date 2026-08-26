"""Tests for semantic_search.py — TF-IDF semantic index.

Run: python3 -m pytest tests/test_semantic_search.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.semantic_search import SemanticIndex


class TestSemanticIndex:
    def test_create(self):
        index = SemanticIndex()
        assert index.doc_count() == 0

    def test_add_and_search(self):
        index = SemanticIndex()
        index.add_doc("doc-1", "The fleet sails at midnight")
        index.add_doc("doc-2", "Breeders work through the night")
        results = index.search("midnight fleet")
        assert len(results) > 0
        assert results[0][0] == "doc-1"

    def test_search_no_match(self):
        index = SemanticIndex()
        index.add_doc("doc-1", "hello world")
        results = index.search("completely unrelated terms")
        assert len(results) == 0

    def test_update_doc(self):
        index = SemanticIndex()
        index.add_doc("doc-1", "hello world")
        index.add_doc("doc-1", "goodbye world")
        results = index.search("goodbye")
        assert results[0][0] == "doc-1"

    def test_remove_doc(self):
        index = SemanticIndex()
        index.add_doc("doc-1", "hello world")
        index.add_doc("doc-2", "hello fleet")
        assert index.remove_doc("doc-1") is True
        results = index.search("world")
        assert len(results) == 0

    def test_remove_missing(self):
        index = SemanticIndex()
        assert index.remove_doc("missing") is False

    def test_vocab_size(self):
        index = SemanticIndex()
        index.add_doc("a", "hello world")
        index.add_doc("b", "hello fleet")
        assert index.vocabulary_size() == 3

    def test_top_k(self):
        index = SemanticIndex()
        index.add_doc("a", "python programming")
        index.add_doc("b", "python scripting")
        index.add_doc("c", "java programming")
        results = index.search("python", top_k=2)
        assert len(results) == 2

    def test_cosine_similarity_range(self):
        index = SemanticIndex()
        index.add_doc("a", "hello world")
        results = index.search("hello")
        assert 0.0 <= results[0][1] <= 1.0

    def test_repr(self):
        index = SemanticIndex()
        assert "SemanticIndex" in repr(index)
