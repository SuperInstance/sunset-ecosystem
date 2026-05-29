"""Semantic text search with TF-IDF and cosine similarity.

Lightweight vector search for fleet knowledge bases, documentation,
and agent output indexing. No external ML dependencies.

Usage:
    index = SemanticIndex()
    index.add_doc("doc-1", "The fleet sails at midnight")
    index.add_doc("doc-2", "Breeders work through the night")
    results = index.search("midnight fleet", top_k=2)
    # [("doc-1", 0.85), ...]
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class SemanticIndex:
    """
    TF-IDF based semantic index.

    :param min_df: Minimum document frequency for a term to be indexed.
    :param max_df_ratio: Maximum document frequency ratio (0.0-1.0).
    """

    def __init__(self, min_df: int = 1, max_df_ratio: float = 0.95):
        self._min_df = min_df
        self._max_df_ratio = max_df_ratio
        self._docs: Dict[str, str] = {}
        self._tokens: Dict[str, List[str]] = {}  # doc_id -> tokens
        self._df: Counter = Counter()  # term -> document frequency
        self._vectors: Dict[str, Dict[str, float]] = {}  # doc_id -> tf-idf vector
        self._n_docs = 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_doc(self, doc_id: str, text: str) -> None:
        """Add or update a document in the index."""
        # Remove old doc if exists
        if doc_id in self._docs:
            self.remove_doc(doc_id)

        tokens = self._tokenize(text)
        self._docs[doc_id] = text
        self._tokens[doc_id] = tokens
        self._n_docs += 1

        # Update DF
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self._df[token] += 1

        # Rebuild all vectors (simplified — in production, incrementally update)
        self._rebuild_vectors()

    def remove_doc(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._docs:
            return False
        tokens = set(self._tokens[doc_id])
        for token in tokens:
            self._df[token] -= 1
            if self._df[token] <= 0:
                del self._df[token]
        del self._docs[doc_id]
        del self._tokens[doc_id]
        del self._vectors[doc_id]
        self._n_docs -= 1
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching query. Returns (doc_id, score) pairs."""
        query_tokens = self._tokenize(query)
        query_vec = self._tfidf_vector(query_tokens)
        if not query_vec:
            return []

        scores: List[Tuple[str, float]] = []
        for doc_id, doc_vec in self._vectors.items():
            score = self._cosine_similarity(query_vec, doc_vec)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, alphanumeric only."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Compute TF-IDF vector for a token list."""
        tf = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        vec: Dict[str, float] = {}
        max_df = int(self._max_df_ratio * self._n_docs)
        for token, count in tf.items():
            df = self._df.get(token, 0)
            if df < self._min_df or (max_df > 0 and df > max_df):
                continue
            idf = math.log((self._n_docs + 1) / (df + 1)) + 1
            vec[token] = (count / total) * idf
        return vec

    def _rebuild_vectors(self) -> None:
        """Rebuild all TF-IDF vectors."""
        self._vectors = {}
        for doc_id, tokens in self._tokens.items():
            self._vectors[doc_id] = self._tfidf_vector(tokens)

    def _cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float],
    ) -> float:
        """Cosine similarity between two sparse vectors."""
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for term, w1 in vec1.items():
            norm1 += w1 * w1
            w2 = vec2.get(term)
            if w2 is not None:
                dot += w1 * w2
        for w2 in vec2.values():
            norm2 += w2 * w2
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def doc_count(self) -> int:
        return self._n_docs

    def vocabulary_size(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"<SemanticIndex docs={self._n_docs} vocab={len(self._df)}>"
