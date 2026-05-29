from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class SearchEngine:
    """
    Full-text search for fleet data.

    Simple inverted index for searching documents.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._index: Dict[str, set] = {}  # term -> set(doc_ids)
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, int] = {"indexed": 0, "searched": 0}

    def index(self, doc_id: str, content: str,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Index a document."""
        self._docs[doc_id] = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        terms = self._tokenize(content)
        for term in terms:
            if term not in self._index:
                self._index[term] = set()
            self._index[term].add(doc_id)
        self._stats["indexed"] += 1

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms."""
        return [t.lower() for t in text.split() if len(t) > 2]

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for documents matching query."""
        self._stats["searched"] += 1
        terms = self._tokenize(query)
        if not terms:
            return []

        # Find intersection of all term postings
        doc_ids = None
        for term in terms:
            if term in self._index:
                if doc_ids is None:
                    doc_ids = self._index[term].copy()
                else:
                    doc_ids &= self._index[term]
            else:
                return []

        if not doc_ids:
            return []

        return [
            {
                "doc_id": doc_id,
                "content": self._docs[doc_id]["content"],
                "metadata": self._docs[doc_id]["metadata"],
            }
            for doc_id in doc_ids
        ]

    def get_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return self._docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Delete a document from the index."""
        if doc_id not in self._docs:
            return False
        doc = self._docs[doc_id]
        terms = self._tokenize(doc["content"])
        for term in terms:
            if term in self._index and doc_id in self._index[term]:
                self._index[term].remove(doc_id)
        del self._docs[doc_id]
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            **self._stats,
            "total_docs": len(self._docs),
            "total_terms": len(self._index),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
