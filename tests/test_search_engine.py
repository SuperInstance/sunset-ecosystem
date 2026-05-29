import pytest
from fleet.search_engine import SearchEngine


class TestSearchEngine:
    def test_init(self):
        se = SearchEngine()
        assert se.fleet_node_id == "default"
        assert se.get_stats()["total_docs"] == 0

    def test_index(self):
        se = SearchEngine()
        se.index("doc1", "hello world")
        assert se.get_stats()["total_docs"] == 1

    def test_search(self):
        se = SearchEngine()
        se.index("doc1", "hello world")
        se.index("doc2", "goodbye world")
        results = se.search("hello")
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

    def test_search_multi_term(self):
        se = SearchEngine()
        se.index("doc1", "hello world test")
        se.index("doc2", "hello world other")
        se.index("doc3", "hello different")
        results = se.search("hello world")
        assert len(results) == 2

    def test_search_no_match(self):
        se = SearchEngine()
        se.index("doc1", "hello world")
        results = se.search("missing")
        assert len(results) == 0

    def test_get_doc(self):
        se = SearchEngine()
        se.index("doc1", "content", {"meta": 1})
        doc = se.get_doc("doc1")
        assert doc["content"] == "content"
        assert doc["metadata"]["meta"] == 1

    def test_delete(self):
        se = SearchEngine()
        se.index("doc1", "hello world")
        assert se.delete("doc1") is True
        assert se.get_stats()["total_docs"] == 0
        assert se.delete("doc1") is False

    def test_delete_removes_from_index(self):
        se = SearchEngine()
        se.index("doc1", "hello world")
        se.delete("doc1")
        results = se.search("hello")
        assert len(results) == 0

    def test_get_stats(self):
        se = SearchEngine()
        se.index("doc1", "hello world")
        se.index("doc2", "another document")
        stats = se.get_stats()
        assert stats["total_docs"] == 2
        assert stats["indexed"] == 2

    def test_to_dict(self):
        se = SearchEngine()
        se.index("doc1", "hello")
        d = se.to_dict()
        assert d["stats"]["total_docs"] == 1
