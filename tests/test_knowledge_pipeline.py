"""Tests for KnowledgePipeline — text chunking, encoding, and per-room search.

Uses mocked FluxVectorTable to avoid turbovec dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.knowledge_pipeline import (
    Chunker,
    FleetDocument,
    KnowledgePipeline,
    PlaceholderEncoder,
)


class FakeFluxTable:
    """Minimal mock for KnowledgePipeline tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self._vectors: dict[int, np.ndarray] = {}
        self._meta: dict[int, object] = {}

    def add(self, av) -> None:
        self._vectors[av.agent_id] = np.array(av.vector, dtype=np.float32)
        self._meta[av.agent_id] = av

    def search(self, query, k=10, **kwargs):
        return []

    def __len__(self) -> int:
        return len(self._vectors)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class TestChunker:
    def test_empty_text(self):
        c = Chunker(chunk_size=64, overlap=8)
        assert c.chunk("") == []
        assert c.chunk("   ") == []

    def test_chunking(self):
        c = Chunker(chunk_size=64, overlap=8)
        text = "a" * 200
        chunks = c.chunk(text)
        assert len(chunks) > 1
        for ch in chunks:
            assert isinstance(ch, FleetDocument)
            assert ch.source == "unknown"

    def test_chunk_metadata(self):
        c = Chunker(chunk_size=64, overlap=8)
        text = "hello world " * 10
        chunks = c.chunk(text, source_meta={"room": "test", "source": "plato"})
        assert len(chunks) > 0
        assert chunks[0].room == "test"
        assert chunks[0].source == "plato"

    def test_chunk_id_deterministic(self):
        c = Chunker(chunk_size=64, overlap=8)
        text = "hello world " * 10
        chunks1 = c.chunk(text)
        chunks2 = c.chunk(text)
        assert len(chunks1) > 0
        assert chunks1[0].doc_id == chunks2[0].doc_id

    def test_skip_tiny_fragments(self):
        c = Chunker(chunk_size=64, overlap=8)
        chunks = c.chunk("hi")
        assert chunks == []  # < 32 chars


# ---------------------------------------------------------------------------
# PlaceholderEncoder
# ---------------------------------------------------------------------------


class TestPlaceholderEncoder:
    def test_encode_one(self):
        enc = PlaceholderEncoder(dim=8, seed=42)
        vec = enc.encode_one("hello")
        assert len(vec) == 8
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_encode_batch(self):
        enc = PlaceholderEncoder(dim=8, seed=42)
        vecs = enc.encode(["hello", "world"])
        assert vecs.shape == (2, 8)
        assert vecs.dtype == np.float32

    def test_deterministic(self):
        enc = PlaceholderEncoder(dim=8, seed=42)
        v1 = enc.encode_one("test")
        v2 = enc.encode_one("test")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        enc = PlaceholderEncoder(dim=8, seed=42)
        v1 = enc.encode_one("hello")
        v2 = enc.encode_one("world")
        assert v1 != v2

    def test_tokenize(self):
        tokens = PlaceholderEncoder._tokenize("hello")
        assert len(tokens) == 3  # "hel", "ell", "llo"

    def test_hash_token(self):
        h1 = PlaceholderEncoder._hash_token("abc")
        h2 = PlaceholderEncoder._hash_token("abc")
        assert h1 == h2
        h3 = PlaceholderEncoder._hash_token("def")
        assert h3 != h1

    def test_vocab_size(self):
        assert PlaceholderEncoder._vocab_size() == 65536


# ---------------------------------------------------------------------------
# KnowledgePipeline
# ---------------------------------------------------------------------------


class TestKnowledgePipeline:
    def test_init(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        assert kp.dim == 8
        assert len(kp._rooms) == 0

    def test_ingest_room(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        kp.ingest_room("forge", documents=["The forge burns hot. " * 10])
        assert "forge" in kp._rooms
        assert len(kp._rooms["forge"]) > 0

    def test_ingest_room_empty(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        kp.ingest_room("empty", documents=[""])
        assert "empty" in kp._rooms

    def test_search_no_rooms(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        assert kp.search("test") == []

    def test_search_with_room(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        kp.ingest_room("engine", documents=["engine room thermal hot. " * 10])
        results = kp.search("thermal", room="engine", k=5)
        assert isinstance(results, list)

    def test_search_all_rooms(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        kp.ingest_room("a", documents=["doc a hot. " * 10])
        kp.ingest_room("b", documents=["doc b hot. " * 10])
        results = kp.search("doc", k=5)
        assert isinstance(results, list)

    def test_room_count(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        kp.ingest_room("a", documents=["doc"])
        kp.ingest_room("b", documents=["doc"])
        assert kp.room_count() == 2

    def test_total_chunks(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        kp.ingest_room("a", documents=["hello world hot. " * 10])
        assert kp.total_chunks() > 0

    def test_repr(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        r = repr(kp)
        assert "KnowledgePipeline" in r

    def test_chunker_override(self):
        custom = Chunker(chunk_size=32, overlap=4)
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2, chunker=custom)
        assert kp.chunker.chunk_size == 32

    def test_ingest_room_with_source(self):
        enc = PlaceholderEncoder(dim=8)
        kp = KnowledgePipeline(encoder=enc, dim=8, bit_width=2)
        kp.ingest_room("room1", documents=["text hot. " * 10], source="github")
        # should not raise
