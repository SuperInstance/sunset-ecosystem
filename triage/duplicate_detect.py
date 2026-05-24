"""Duplicate Issue Detection — SPEC-REPO-METRIC §3.

Uses simple TF-IDF + cosine similarity to flag potentially
duplicate GitHub issues before they fragment discussion.
"""
from __future__ import annotations

__all__ = ["DuplicateDetector", "find_duplicates"]

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class DuplicatePair:
    """A pair of issues flagged as potential duplicates."""

    issue_a: int
    issue_b: int
    similarity: float
    shared_terms: List[str]


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, return word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_idf(docs: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency for each term."""
    n = len(docs)
    idf: Dict[str, float] = {}
    for doc in docs:
        unique = set(doc)
        for term in unique:
            idf[term] = idf.get(term, 0) + 1
    for term in idf:
        idf[term] = n / idf[term]
    return idf


def _tfidf_vector(doc: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute TF-IDF vector for a document."""
    tf = Counter(doc)
    total = len(doc)
    return {term: (count / total) * idf.get(term, 1.0) for term, count in tf.items()}


def _cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    shared = set(a.keys()) & set(b.keys())
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DuplicateDetector:
    """Find duplicate issue candidates from a list of GitHub issues."""

    def __init__(self, threshold: float = 0.65, min_shared_terms: int = 3) -> None:
        self.threshold = threshold
        self.min_shared_terms = min_shared_terms

    def detect(
        self, issues: List[dict]
    ) -> List[DuplicatePair]:
        """Analyze issues and return flagged duplicate pairs.

        Args:
            issues: List of dicts with keys 'number', 'title', 'body'.

        Returns:
            Sorted list of DuplicatePair (highest similarity first).
        """
        # Build corpus
        docs: List[List[str]] = []
        for issue in issues:
            text = f"{issue.get('title', '')} {issue.get('body', '')}"
            docs.append(_tokenize(text))

        if len(docs) < 2:
            return []

        idf = _compute_idf(docs)
        vectors = [_tfidf_vector(d, idf) for d in docs]

        pairs: List[DuplicatePair] = []
        for i in range(len(issues)):
            for j in range(i + 1, len(issues)):
                sim = _cosine_sim(vectors[i], vectors[j])
                if sim >= self.threshold:
                    shared = sorted(
                        set(vectors[i].keys()) & set(vectors[j].keys()),
                        key=lambda t: vectors[i][t] * vectors[j][t],
                        reverse=True,
                    )
                    if len(shared) >= self.min_shared_terms:
                        pairs.append(
                            DuplicatePair(
                                issue_a=issues[i]["number"],
                                issue_b=issues[j]["number"],
                                similarity=round(sim, 3),
                                shared_terms=shared[:5],
                            )
                        )

        pairs.sort(key=lambda p: p.similarity, reverse=True)
        return pairs


def find_duplicates(
    issues: List[dict],
    threshold: float = 0.65,
    min_shared_terms: int = 3,
) -> List[DuplicatePair]:
    """One-liner entrypoint."""
    return DuplicateDetector(threshold, min_shared_terms).detect(issues)
