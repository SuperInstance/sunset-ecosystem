"""Tests for JEPA Room local inference system.

Covers tile encoding, prediction, API fallback, and statistics.
"""

import numpy as np
import pytest

from jepa.jepa_room import JEPARoom, JEPAPrediction, _MockEncoder, _MockPredictor


class TestMockEncoder:
    def test_encode_deterministic(self):
        enc = _MockEncoder(128)
        e1 = enc.encode("hello")
        e2 = enc.encode("hello")
        assert np.allclose(e1, e2)
        assert len(e1) == 128

    def test_encode_different_texts(self):
        enc = _MockEncoder(128)
        e1 = enc.encode("hello")
        e2 = enc.encode("world")
        assert not np.allclose(e1, e2)

    def test_encode_normalized(self):
        enc = _MockEncoder(128)
        e = enc.encode("test")
        assert abs(np.linalg.norm(e) - 1.0) < 1e-6


class TestMockPredictor:
    def test_predict_shape(self):
        pred = _MockPredictor(128)
        ctx = np.random.randn(128).astype(np.float32)
        ctx = ctx / np.linalg.norm(ctx)
        out = pred.predict(ctx)
        assert len(out) == 128


class TestJEPARoom:
    def test_init(self):
        room = JEPARoom(room_id="harbor", dim=128)
        assert room.room_id == "harbor"
        assert room.dim == 128
        assert room._tile_history == []

    def test_feed_tile(self):
        room = JEPARoom(room_id="harbor", dim=128)
        emb = room.feed_tile({"question": "Q", "answer": "A"})
        assert len(emb) == 128
        assert len(room._tile_history) == 1

    def test_feed_multiple_tiles(self):
        room = JEPARoom(room_id="harbor", dim=128)
        for i in range(5):
            room.feed_tile({"question": f"Q{i}", "answer": f"A{i}"})
        assert len(room._tile_history) == 5
        assert len(room._tile_embeddings) == 5

    def test_predict_next_tile_empty(self):
        room = JEPARoom(room_id="harbor", dim=128)
        pred = room.predict_next_tile()
        assert pred.confidence == 0.0
        assert pred.predicted_tile is None

    def test_predict_next_tile(self):
        room = JEPARoom(room_id="harbor", dim=128)
        room.feed_tile({"question": "Q1", "answer": "A1"})
        room.feed_tile({"question": "Q2", "answer": "A2"})
        pred = room.predict_next_tile()
        assert pred.confidence >= 0.0
        assert pred.latency_ms >= 0.0
        assert pred.source == "jepa"

    def test_query_high_confidence(self):
        room = JEPARoom(room_id="harbor", dim=128)
        # Feed similar tiles to build context
        for i in range(10):
            room.feed_tile({"question": f"fleet status {i}", "answer": f"ok {i}"})
        result = room.query("fleet status", min_confidence=0.0)
        assert result.source in ("jepa", "api")
        assert result.confidence >= 0.0

    def test_query_low_confidence_fallback(self):
        room = JEPARoom(room_id="harbor", dim=128)
        # Only 1 tile, confidence will be low
        room.feed_tile({"question": "Q", "answer": "A"})
        result = room.query("something new", min_confidence=0.99)
        assert result.source == "api"
        assert result.confidence == 1.0

    def test_api_fallback(self):
        room = JEPARoom(room_id="harbor", dim=128)
        result = room.api_fallback("test question")
        assert result.source == "api"
        assert result.predicted_tile is not None
        assert "API response" in result.predicted_tile["answer"]

    def test_stats(self):
        room = JEPARoom(room_id="harbor", dim=128)
        for i in range(5):
            room.feed_tile({"question": f"Q{i}", "answer": f"A{i}"})
        room.query("Q0", min_confidence=0.0)
        stats = room.get_stats()
        assert stats["room_id"] == "harbor"
        assert stats["tile_count"] == 6  # 5 + query tile
        assert stats["total_queries"] >= 1

    def test_similar_tiles(self):
        room = JEPARoom(room_id="harbor", dim=128)
        room.feed_tile({"question": "fleet status", "answer": "ok"})
        room.feed_tile({"question": "fleet health", "answer": "good"})
        room.feed_tile({"question": "weather", "answer": "sunny"})
        pred = room.predict_next_tile()
        assert len(pred.similar_tiles) <= 3

    def test_cosine_sim(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert JEPARoom._cosine_sim(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0, 0.0])
        assert JEPARoom._cosine_sim(a, c) == pytest.approx(0.0)


class TestJEPAPrediction:
    def test_defaults(self):
        p = JEPAPrediction(predicted_embedding=np.zeros(128), confidence=0.8)
        assert p.latency_ms == 0.0
        assert p.source == "jepa"
        assert p.similar_tiles == []
