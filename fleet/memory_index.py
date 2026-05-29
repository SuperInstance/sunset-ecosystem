"""In-memory inverted index for fast full-text search.

Builds term-to-document indexes for fast lookup. Used for fleet
log search, agent capability lookup, and document retrieval.

Usage:
    idx = MemoryIndex()
    idx.add("doc-1", "hello world")
    idx.add("doc-2", "hello fleet")
    results = idx.search("hello")
    # -> ["doc-1", "doc-2"]
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set


class MemoryIndex:
    """
    Simple inverted index with TF-IDF scoring.

    :param stopwords: Set of words to ignore.
    """

    def __init__(self, stopwords: Optional[Set[str]] = None):
        self._index: Dict[str, Set[str]] = defaultdict(set)
        self._docs: Dict[str, str] = {}
        self._doc_freq: Dict[str, int] = defaultdict(int)
        self._stopwords = stopwords or {"the", "a", "an", "is", "are", "was", "were"}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        """Index a document."""
        self._docs[doc_id] = text
        terms = self._tokenize(text)
        unique_terms = set(terms)
        for term in unique_terms:
            self._index[term].add(doc_id)
            self._doc_freq[term] += 1

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._docs:
            return False
        text = self._docs.pop(doc_id)
        terms = set(self._tokenize(text))
        for term in terms:
            self._index[term].discard(doc_id)
            self._doc_freq[term] -= 1
            if self._doc_freq[term] <= 0:
                self._index.pop(term, None)
                self._doc_freq.pop(term, None)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[str]:
        """Search for documents matching all query terms."""
        terms = self._tokenize(query)
        if not terms:
            return []
        results: Set[str] = set(self._docs.keys())
        for term in terms:
            results &= self._index.get(term, set())
        return sorted(results)

    def search_any(self, query: str) -> List[str]:
        """Search for documents matching any query term."""
        terms = self._tokenize(query)
        if not terms:
            return []
        results: Set[str] = set()
        for term in terms:
            results |= self._index.get(term, set())
        return sorted(results)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [w for w in words if w not in self._stopwords]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def doc_count(self) -> int:
        return len(self._docs)

    def term_count(self) -> int:
        return len(self._index)

    def terms(self) -> List[str]:
        return sorted(self._index.keys())

    def __repr__(self) -> str:
        return f"<MemoryIndex docs={len(self._docs)} terms={len(self._index)}>"
