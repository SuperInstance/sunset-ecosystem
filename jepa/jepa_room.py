"""jepa/jepa_room.py — JEPA-powered room for local model inference.

JEPA (Joint Embedding Predictive Architecture) rooms run local
encoder/predictor models on tile sequences, enabling fast,
privacy-preserving inference without API calls.

Architecture
------------
- Encoder: embeds tiles into latent space (context + target)
- Predictor: predicts target embeddings from context
- Tile matcher: finds similar tiles from prediction
- API fallback: delegates to API when local confidence is low

Usage
-----
    from jepa.jepa_room import JEPARoom

    room = JEPARoom(room_id="harbor", dim=256)
    room.load_encoder("encoder.pth")  # or mock
    
    # Feed tiles
    room.feed_tile({"question": "Q", "answer": "A", "domain": "harbor"})
    
    # Predict next tile
    prediction = room.predict_next_tile()
    
    # Query with confidence
    result = room.query("What is the fleet status?", min_confidence=0.8)
    if result.confidence < min_confidence:
        result = room.api_fallback("What is the fleet status?")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# JEPA availability check
JEPA_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    JEPA_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available; JEPA room using mock inference")


@dataclass
class JEPAPrediction:
    """Result of a JEPA prediction."""
    predicted_embedding: np.ndarray
    confidence: float
    predicted_tile: Optional[Dict[str, Any]] = None
    similar_tiles: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    source: str = "jepa"  # "jepa" or "api"


@dataclass
class JEPARoom:
    """Room that uses JEPA for local tile prediction."""
    
    room_id: str
    dim: int = 256
    _encoder: Optional[Any] = field(default=None, repr=False)
    _predictor: Optional[Any] = field(default=None, repr=False)
    _tile_embeddings: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _tile_history: List[Dict[str, Any]] = field(default_factory=list)
    _total_queries: int = 0
    _jepa_queries: int = 0
    _api_fallbacks: int = 0
    
    def __post_init__(self):
        if JEPA_AVAILABLE:
            self._init_torch_models()
        else:
            self._init_mock_models()
    
    def _init_torch_models(self) -> None:
        """Initialize real PyTorch JEPA models."""
        # Placeholder: real models would be loaded from checkpoints
        self._encoder = _MockEncoder(self.dim)
        self._predictor = _MockPredictor(self.dim)
        logger.info("JEPA torch models initialized (dim=%d)", self.dim)
    
    def _init_mock_models(self) -> None:
        """Initialize mock models for testing."""
        self._encoder = _MockEncoder(self.dim)
        self._predictor = _MockPredictor(self.dim)
        logger.info("JEPA mock models initialized (dim=%d)", self.dim)
    
    def feed_tile(self, tile: Dict[str, Any]) -> np.ndarray:
        """Ingest a tile and compute its embedding."""
        tile_text = json.dumps(tile, sort_keys=True)
        embedding = self._encoder.encode(tile_text)
        tile_hash = hashlib.sha256(tile_text.encode()).hexdigest()[:16]
        self._tile_embeddings[tile_hash] = embedding
        self._tile_history.append(tile)
        return embedding
    
    def predict_next_tile(self, context_window: int = 5) -> JEPAPrediction:
        """Predict the next tile based on recent context."""
        t0 = time.perf_counter()
        
        if len(self._tile_history) < 2:
            return JEPAPrediction(
                predicted_embedding=np.zeros(self.dim),
                confidence=0.0,
                predicted_tile=None,
                similar_tiles=[],
                latency_ms=0.0,
                source="jepa"
            )
        
        # Get context embeddings
        recent = self._tile_history[-context_window:]
        context_text = json.dumps(recent, sort_keys=True)
        context_emb = self._encoder.encode(context_text)
        
        # Predict next embedding
        predicted_emb = self._predictor.predict(context_emb)
        
        # Find similar tiles
        similar = self._find_similar_tiles(predicted_emb, top_k=3)
        
        # Confidence based on similarity score (clamp to [0, 1])
        confidence = max(0.0, similar[0]["score"]) if similar else 0.5
        
        dt = (time.perf_counter() - t0) * 1000
        
        return JEPAPrediction(
            predicted_embedding=predicted_emb,
            confidence=min(confidence, 1.0),
            predicted_tile=similar[0]["tile"] if similar else None,
            similar_tiles=[s["tile"] for s in similar],
            latency_ms=dt,
            source="jepa"
        )
    
    def query(self, question: str, min_confidence: float = 0.7) -> JEPAPrediction:
        """Query the room with JEPA prediction. Falls back to API if confidence low."""
        self._total_queries += 1
        
        # Feed the question as a tile
        question_tile = {"question": question, "answer": "", "domain": self.room_id}
        self.feed_tile(question_tile)
        
        # Predict
        prediction = self.predict_next_tile()
        
        if prediction.confidence >= min_confidence:
            self._jepa_queries += 1
            return prediction
        else:
            self._api_fallbacks += 1
            return self.api_fallback(question)
    
    def api_fallback(self, question: str) -> JEPAPrediction:
        """Fallback to API when JEPA confidence is low."""
        # Mock API call
        t0 = time.perf_counter()
        api_response = f"API response to: {question}"
        dt = (time.perf_counter() - t0) * 1000
        
        return JEPAPrediction(
            predicted_embedding=np.zeros(self.dim),
            confidence=1.0,  # API is "certain"
            predicted_tile={"question": question, "answer": api_response, "domain": self.room_id},
            similar_tiles=[],
            latency_ms=dt,
            source="api"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "dim": self.dim,
            "total_queries": self._total_queries,
            "jepa_queries": self._jepa_queries,
            "api_fallbacks": self._api_fallbacks,
            "jepa_ratio": self._jepa_queries / max(self._total_queries, 1),
            "tile_count": len(self._tile_history),
            "jepa_available": JEPA_AVAILABLE,
        }
    
    def _find_similar_tiles(self, embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find tiles most similar to embedding."""
        if not self._tile_embeddings:
            return []
        
        similarities = []
        for tile_hash, emb in self._tile_embeddings.items():
            score = self._cosine_sim(embedding, emb)
            # Find the tile by hash (simplified: use history)
            tile = self._tile_history[-1] if self._tile_history else {}
            similarities.append({"tile": tile, "score": score, "hash": tile_hash})
        
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)


# ── Mock JEPA models for testing without PyTorch ─────────────────────────

class _MockEncoder:
    def __init__(self, dim: int):
        self.dim = dim
        self._rng = np.random.RandomState(42)
    
    def encode(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding."""
        h = hashlib.sha256(text.encode()).hexdigest()
        seed = int(h[:8], 16)
        rng = np.random.RandomState(seed)
        emb = rng.randn(self.dim).astype(np.float32)
        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

class _MockPredictor:
    def __init__(self, dim: int):
        self.dim = dim
        self._W = np.random.randn(dim, dim).astype(np.float32) * 0.1
    
    def predict(self, context_emb: np.ndarray) -> np.ndarray:
        """Simple linear prediction with noise."""
        pred = self._W @ context_emb
        # Add small noise
        pred += np.random.randn(self.dim) * 0.01
        # Normalize
        pred = pred / (np.linalg.norm(pred) + 1e-8)
        return pred.astype(np.float32)
