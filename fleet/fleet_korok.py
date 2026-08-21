"""FleetKorok — Hybrid text search adapter for Pringled/korok.

Wraps **korok** (dense + BM25 sparse + cross-encoder reranking) to provide
text-based tile and document retrieval for the fleet.

Use cases:
- Search past agent tiles by semantic meaning + keyword
- Retrieve relevant documentation for breeding context
- Rerank search results with a cross-encoder for higher precision

Reference: https://github.com/Pringled/korok
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FleetKorokEntry:
    """A single document / tile in the hybrid search index."""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None  # pre-computed dense vector


@dataclass
class FleetKorokResult:
    """One search result."""

    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


@dataclass
class FleetKorokConfig:
    """Configuration for FleetKorok hybrid search."""

    alpha: float = 0.5  # dense/sparse balance (0=sparse only, 1=dense only)
    k_reranker: int = 30  # candidates for cross-encoder reranking
    use_bm25: bool = True
    use_dense: bool = True
    use_reranker: bool = False
    encoder: Optional[Any] = None  # model2vec StaticModel or similar
    reranker: Optional[Any] = None  # sentence-transformers CrossEncoder
    distance_metric: str = "cosine"
    stopwords: str = "en"


class FleetKorokIndex:
    """Hybrid search index for fleet documents and tiles.

    Wraps korok.Pipeline with fleet-specific conveniences:
    - doc_id mapping (korok uses raw text strings)
    - metadata passthrough
    - pre-computed vector ingest (skip re-encoding)
    - fallback to pure-Python when korok is unavailable
    """

    def __init__(self, config: Optional[FleetKorokConfig] = None):
        self.config = config or FleetKorokConfig()
        self._entries: Dict[str, FleetKorokEntry] = {}
        self._pipeline: Optional[Any] = None
        self._ready = False
        self._try_init()

    # ── Initialization ───────────────────────────────────────────────

    def _try_init(self) -> None:
        """Attempt to import korok. If unavailable, mark fallback mode."""
        try:
            import korok

            self._korok_module = korok
            log.info("FleetKorok: korok backend available")
        except ImportError:
            self._korok_module = None
            log.warning("FleetKorok: korok not installed; using fallback")

    # ── Ingest ─────────────────────────────────────────────────────────

    def add_entries(self, entries: List[FleetKorokEntry]) -> None:
        """Add documents to the index.

        If the korok backend is available and the index is not yet built,
        this stores entries. Call ``build()`` to create the searchable index.
        """
        for e in entries:
            self._entries[e.doc_id] = e
        self._ready = False  # needs rebuild

    def build(self) -> None:
        """Build the hybrid search index from stored entries."""
        if not self._entries:
            self._pipeline = None
            self._ready = True
            return

        texts = [e.text for e in self._entries.values()]

        if self._korok_module is not None:
            try:
                self._pipeline = self._korok_module.Pipeline.fit(
                    texts=texts,
                    encoder=self.config.encoder,
                    use_bm25=self.config.use_bm25,
                    reranker=self.config.reranker,
                    alpha=self.config.alpha,
                    stopwords=self.config.stopwords,
                )
                self._ready = True
                log.info("FleetKorok: index built with %d docs", len(texts))
                return
            except Exception as exc:
                log.warning("korok build failed: %s; falling back", exc)
                self._pipeline = None

        # Fallback: store for brute-force search
        self._ready = True

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> List[FleetKorokResult]:
        """Hybrid search for ``query``.

        Returns top-``k`` results ordered by relevance score.
        """
        if not self._ready:
            self.build()
        if not self._entries:
            return []

        if self._pipeline is not None:
            return self._search_korok(query, k)
        return self._search_fallback(query, k)

    def _search_korok(self, query: str, k: int) -> List[FleetKorokResult]:
        results = self._pipeline.query([query], k=k, k_reranker=self.config.k_reranker)
        # results is list of (text, score) tuples
        out: List[FleetKorokResult] = []
        for text, score in results[0]:
            # map text back to doc_id
            doc_id = self._text_to_doc_id(text)
            entry = self._entries.get(doc_id)
            out.append(
                FleetKorokResult(
                    doc_id=doc_id,
                    text=text,
                    score=float(score),
                    metadata=entry.metadata if entry else {},
                )
            )
        return out

    def _search_fallback(self, query: str, k: int) -> List[FleetKorokResult]:
        """Brute-force fallback: simple keyword matching."""
        query_lower = query.lower()
        scored: List[Tuple[str, float]] = []
        for doc_id, entry in self._entries.items():
            text_lower = entry.text.lower()
            # Simple score: count query word occurrences
            words = query_lower.split()
            score = sum(text_lower.count(w) for w in words) / max(len(words), 1)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        out: List[FleetKorokResult] = []
        for doc_id, score in scored[:k]:
            entry = self._entries[doc_id]
            out.append(
                FleetKorokResult(
                    doc_id=doc_id,
                    text=entry.text,
                    score=score,
                    metadata=entry.metadata,
                )
            )
        return out

    def _text_to_doc_id(self, text: str) -> str:
        """Reverse-map text back to doc_id."""
        for doc_id, entry in self._entries.items():
            if entry.text == text:
                return doc_id
        return text  # fallback: use text itself as id

    # ── Utilities ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def get_entry(self, doc_id: str) -> Optional[FleetKorokEntry]:
        return self._entries.get(doc_id)

    def remove(self, doc_id: str) -> bool:
        if doc_id in self._entries:
            del self._entries[doc_id]
            self._ready = False
            return True
        return False

    def clear(self) -> None:
        self._entries.clear()
        self._pipeline = None
        self._ready = False

    # ── Integration helpers ──────────────────────────────────────────────

    @classmethod
    def from_tile_list(
        cls,
        tiles: List[Dict[str, Any]],
        text_extractor: Optional[Callable[[Dict[str, Any]], str]] = None,
        config: Optional[FleetKorokConfig] = None,
    ) -> "FleetKorokIndex":
        """Build index from a list of tile dicts.

        :param tiles: List of tile dicts with at least ``tile_id`` and ``text``.
        :param text_extractor: Callable that extracts searchable text from a tile dict.
        :param config: FleetKorokConfig instance.
        """
        if text_extractor is None:

            def text_extractor(tile):
                return tile.get("text", tile.get("description", ""))

        inst = cls(config=config)
        entries = [
            FleetKorokEntry(
                doc_id=tile.get("tile_id", str(i)),
                text=text_extractor(tile),
                metadata=tile,
            )
            for i, tile in enumerate(tiles)
        ]
        inst.add_entries(entries)
        inst.build()
        return inst

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata (not the index itself)."""
        return {
            "n_entries": len(self._entries),
            "config": {
                "alpha": self.config.alpha,
                "use_bm25": self.config.use_bm25,
                "use_dense": self.config.use_dense,
                "use_reranker": self.config.use_reranker,
            },
            "ready": self._ready,
            "backend": "korok" if self._pipeline is not None else "fallback",
        }
