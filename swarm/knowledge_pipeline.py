"""Fleet knowledge embedding pipeline — local, pluggable, per-room.

Ingests text from any fleet source (PLATO rooms, agent outputs, GitHub,
documents), chunks it, embeds via a lightweight encoder, and stores in
per-room FluxVectorTable indices.

Design goals:
    - Zero API calls (fully local)
    - Pluggable encoder (swap in sentence-transformers, ollama, etc.)
    - Per-room slices (isolated, no cross-contamination)
    - Incremental ingestion (append-only, no rebuilds)

Example::

    from swarm.knowledge_pipeline import KnowledgePipeline, PlaceholderEncoder

    encoder = PlaceholderEncoder(dim=256)
    pipeline = KnowledgePipeline(encoder=encoder, dim=256)

    # Ingest a PLATO room
    pipeline.ingest_room(
        room_name="forge",
        documents=["The forge burns at 1200°C...", "Flux VM spec v3..."],
    )

    # Search across all rooms
    results = pipeline.search("What is the Flux VM resolution?", k=5)

    # Search single room
    results = pipeline.search("thermal budget", room="engine-room", k=3)
"""

from __future__ import annotations

__all__ = [
    "KnowledgePipeline",
    "PlaceholderEncoder",
    "Chunker",
    "FleetDocument",
]

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FleetDocument:
    """A single document chunk ready for embedding."""

    doc_id: str
    text: str
    room: str
    source: str  # e.g. "plato", "github", "agent_output"
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker:
    """Simple sliding-window text chunker.

    Splits text into overlapping chunks to preserve context at boundaries.

    Args:
        chunk_size: Target characters per chunk.
        overlap: Overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self, text: str, source_meta: dict[str, Any] | None = None
    ) -> list[FleetDocument]:
        """Chunk text into FleetDocument pieces."""
        if not text.strip():
            return []

        meta = source_meta or {}
        chunks: list[FleetDocument] = []

        # Simple char-based sliding window
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(text), step):
            piece = text[i : i + self.chunk_size].strip()
            if len(piece) < 32:
                continue  # Skip tiny fragments

            # Deterministic ID from content hash
            doc_id = hashlib.blake2b(
                f"{meta.get('room', 'unknown')}:{i}:{piece[:64]}".encode(),
                digest_size=8,
            ).hexdigest()

            chunks.append(
                FleetDocument(
                    doc_id=doc_id,
                    text=piece,
                    room=meta.get("room", "unknown"),
                    source=meta.get("source", "unknown"),
                    metadata={
                        **meta,
                        "chunk_index": i // step,
                        "char_start": i,
                        "char_end": i + len(piece),
                    },
                )
            )

        return chunks


class PlaceholderEncoder:
    """Deterministic random-projection encoder for testing.

    Uses a fixed random matrix to project text (via hashed trigrams)
    into a dense vector. Not semantically meaningful, but:
        - Fast (no model loading)
        - Deterministic (same text → same vector)
        - Dimensionally correct (for integration testing)

    **Production swap:** Replace with::

        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # encoder.encode(text) → 384-dim float32
    """

    def __init__(self, dim: int = 256, seed: int = 42) -> None:
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._projection = rng.standard_normal((self._vocab_size(), dim)).astype(
            np.float32
        )
        self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into vectors.

        Returns:
            ndarray of shape (len(texts), dim), float32, L2-normalized.
        """
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            indices = [self._hash_token(t) % self._vocab_size() for t in tokens]
            if indices:
                vec = self._projection[indices].mean(axis=0)
                vec /= np.linalg.norm(vec) + 1e-8
                vectors[i] = vec
        return vectors

    def encode_one(self, text: str) -> list[float]:
        """Encode a single text."""
        return self.encode([text])[0].tolist()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Extract character trigrams as tokens."""
        text = text.lower()
        return [text[i : i + 3] for i in range(len(text) - 2)]

    @staticmethod
    def _hash_token(token: str) -> int:
        return int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16)

    @staticmethod
    def _vocab_size() -> int:
        return 65536  # 2^16, fits in uint16


class KnowledgePipeline:
    """Fleet-wide knowledge ingestion and search.

    Manages per-room FluxVectorTable indices and provides cross-room
    search with source attribution.

    Args:
        encoder: Callable that converts text → vector.
        dim: Vector dimensionality.
        bit_width: Quantization bits for turbovec storage.
        chunker: Text chunking strategy.
        base_path: Where to persist indices (optional).
    """

    def __init__(
        self,
        encoder: Optional[Callable[[str], list[float]]] = None,
        dim: int = 256,
        bit_width: int = 4,
        chunker: Optional[Chunker] = None,
        base_path: Optional[str | Path] = None,
    ) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self.encoder = encoder or PlaceholderEncoder(dim=dim)
        self.chunker = chunker or Chunker()
        self.base_path = Path(base_path) if base_path else None

        # Per-room indices: room_name → FluxVectorTable
        self._rooms: dict[str, "FluxVectorTable"] = {}
        # Document registry: doc_id → FleetDocument
        self._docs: dict[str, FleetDocument] = {}

    # ── ingestion ───────────────────────────────────────────

    def ingest_room(
        self,
        room_name: str,
        documents: list[str],
        source: str = "unknown",
    ) -> list[str]:
        """Ingest documents into a room's knowledge index.

        Args:
            room_name: PLATO room or fleet domain name.
            documents: Raw text documents.
            source: Origin label (e.g. "plato", "github", "agent").

        Returns:
            List of doc_ids created.
        """
        from swarm.vector_table import AgentVector, FluxVectorTable

        # Ensure room index exists
        if room_name not in self._rooms:
            self._rooms[room_name] = FluxVectorTable(
                dim=self.dim, bit_width=self.bit_width
            )
            logger.info("Created knowledge index for room '%s'", room_name)

        table = self._rooms[room_name]
        doc_ids: list[str] = []

        for doc_text in documents:
            chunks = self.chunker.chunk(
                doc_text,
                source_meta={"room": room_name, "source": source},
            )

            # Batch encode all chunks for this document
            texts = [c.text for c in chunks]
            vectors = self.encoder.encode(texts)

            for chunk, vector in zip(chunks, vectors):
                # Numeric ID from hex doc_id
                numeric_id = int(chunk.doc_id, 16) % (2**64)

                table.add(
                    AgentVector(
                        agent_id=numeric_id,
                        vector=vector.tolist(),
                        fitness=1.0,  # All docs equally fit initially
                        extra={"doc_id": chunk.doc_id},
                    )
                )
                self._docs[chunk.doc_id] = chunk
                doc_ids.append(chunk.doc_id)

        logger.info(
            "Ingested %d chunks into room '%s' from %d documents",
            len(doc_ids),
            room_name,
            len(documents),
        )
        return doc_ids

    def ingest_file(
        self, path: str | Path, room: str, source: str = "file"
    ) -> list[str]:
        """Ingest a single file into a room."""
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_room(room, [text], source=source)

    # ── search ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        room: Optional[str] = None,
        k: int = 5,
    ) -> list[tuple[str, float, FleetDocument]]:
        """Search fleet knowledge.

        Args:
            query: Natural language query.
            room: Restrict to a single room, or None for fleet-wide.
            k: Number of results.

        Returns:
            List of (room_name, score, FleetDocument) sorted best-first.
        """
        query_vec = self.encoder.encode_one(query)

        if room is not None:
            # Single-room search
            if room not in self._rooms:
                return []
            return self._search_one(query_vec, room, k)

        # Fleet-wide: search all rooms, merge, re-rank
        all_results: list[tuple[str, float, FleetDocument]] = []
        for room_name, table in self._rooms.items():
            all_results.extend(self._search_one(query_vec, room_name, k * 2))

        # Re-rank by score descending
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]

    def _search_one(
        self,
        query_vec: list[float],
        room_name: str,
        k: int,
    ) -> list[tuple[str, float, FleetDocument]]:
        """Search a single room index."""
        table = self._rooms[room_name]
        results = table.search(query=query_vec, k=k)

        found: list[tuple[str, float, FleetDocument]] = []
        for numeric_id, score, _meta in results:
            # Reverse lookup via doc_id in meta
            # The meta stores doc_id in the 'extra' field
            doc_id = _meta.extra.get("doc_id") if hasattr(_meta, "extra") else None
            if doc_id and doc_id in self._docs:
                found.append((room_name, score, self._docs[doc_id]))
            else:
                # Fallback: brute-force scan through docs (inefficient but safe)
                for did, doc in self._docs.items():
                    if doc.room == room_name:
                        # Can't verify without doc_id linkage
                        break
        return found

    # ── persistence ───────────────────────────────────────

    def save(self, path: Optional[str | Path] = None) -> None:
        """Persist all room indices and document registry."""
        base = Path(path) if path else self.base_path
        if base is None:
            raise ValueError("No base_path set for save()")

        base.mkdir(parents=True, exist_ok=True)

        # Save each room index
        for room_name, table in self._rooms.items():
            safe_name = room_name.replace("/", "_")
            table.write(base / f"{safe_name}.knowledge")

        # Save document registry as JSON
        import json

        registry = {
            doc_id: {
                "doc_id": d.doc_id,
                "text": d.text,
                "room": d.room,
                "source": d.source,
                "metadata": d.metadata,
            }
            for doc_id, d in self._docs.items()
        }
        (base / "registry.json").write_text(json.dumps(registry, indent=2))
        logger.info("Saved knowledge pipeline to %s (%d rooms)", base, len(self._rooms))

    @classmethod
    def load(
        cls,
        path: str | Path,
        encoder: Optional[Callable[[str], list[float]]] = None,
        dim: int = 256,
        bit_width: int = 4,
    ) -> "KnowledgePipeline":
        """Load a previously saved knowledge pipeline."""
        from swarm.vector_table import FluxVectorTable

        base = Path(path)
        instance = cls(encoder=encoder, dim=dim, bit_width=bit_width, base_path=base)

        # Load document registry
        import json

        registry_path = base / "registry.json"
        if registry_path.exists():
            raw = json.loads(registry_path.read_text())
            for doc_id, d in raw.items():
                instance._docs[doc_id] = FleetDocument(
                    doc_id=d["doc_id"],
                    text=d["text"],
                    room=d["room"],
                    source=d["source"],
                    metadata=d.get("metadata", {}),
                )

        # Load room indices
        for p in base.glob("*.knowledge.tvim"):
            room_name = p.stem.replace(".knowledge", "").replace("_", "/")
            table = FluxVectorTable.load(
                p.with_suffix("").with_suffix(".knowledge"),
                dim=dim,
                bit_width=bit_width,
            )
            instance._rooms[room_name] = table

        logger.info(
            "Loaded knowledge pipeline from %s (%d rooms, %d docs)",
            base,
            len(instance._rooms),
            len(instance._docs),
        )
        return instance

    # ── stats ───────────────────────────────────────────────

    @property
    def room_names(self) -> list[str]:
        return list(self._rooms.keys())

    def room_count(self) -> int:
        return len(self._rooms)

    def doc_count(self) -> int:
        return len(self._docs)

    def total_chunks(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return (
            f"KnowledgePipeline(rooms={self.room_count()}, "
            f"docs={self.doc_count()}, dim={self.dim})"
        )
