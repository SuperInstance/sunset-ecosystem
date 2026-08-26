"""Nerve Fiber — A sensory pathway that perceives, adapts, compiles, and alerts on novelty.

Lifecycle mirrors Soft → Snap → Hard:
    PERCEIVING (soft) → ADAPTING → COMPILED (hard) → NOVELTY_ALERT → PERCEIVING

Like putting on shoes: feel every edge → dampen → muscle memory → rock in shoe.

Ecosystem integration (optional):
    - eisenstein_embed: Bitvector fingerprints for tile pattern matching
    - device_router: Auto-detect optimal compute device for micro-models
    - tensor_spline: SplineLinear for adaptive weight parameterisation
    - triplet_miner: Git-powered triplet mining for novelty anchors
"""

from __future__ import annotations

__all__ = ["NerveFiber", "FiberState", "SensoryTile", "ECOSYSTEM"]

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Ecosystem dependency detection ───────────────────────────

try:
    from eisenstein_embed.bitvector import (
        text_fingerprint,
        bitvector_similarity,
        find_best_bitvector_match,
    )

    HAS_EISENSTEIN = True
except ImportError:
    HAS_EISENSTEIN = False

try:
    from device_router.router import DeviceRouter

    HAS_DEVICE_ROUTER = True
except ImportError:
    HAS_DEVICE_ROUTER = False

try:
    from tensor_spline.spline import SplineLinear

    HAS_TENSOR_SPLINE = True
except ImportError:
    HAS_TENSOR_SPLINE = False

try:
    from triplet_miner.git_miner import TripletMiner

    HAS_TRIPLET_MINER = True
except ImportError:
    HAS_TRIPLET_MINER = False

ECOSYSTEM: dict[str, bool] = {
    "eisenstein_embed": HAS_EISENSTEIN,
    "device_router": HAS_DEVICE_ROUTER,
    "tensor_spline": HAS_TENSOR_SPLINE,
    "triplet_miner": HAS_TRIPLET_MINER,
}

# Singleton device router (lazy-initialised)
_device_router: DeviceRouter | None = None
_device_router_lock = threading.Lock()


def _get_device_router() -> DeviceRouter | None:
    """Return a lazily-initialised DeviceRouter singleton."""
    global _device_router
    if not HAS_DEVICE_ROUTER:
        return None
    if _device_router is not None:
        return _device_router
    with _device_router_lock:
        if _device_router is None:
            _device_router = DeviceRouter()
            _device_router.detect()
        return _device_router


class FiberState(Enum):
    """The lifecycle states of a nerve fiber."""

    PERCEIVING = "perceiving"  # Soft — full attention, raw inference
    ADAPTING = "adapting"  # Building confidence, learning pattern
    COMPILED = "compiled"  # Hard — automatic processing, muscle memory
    NOVELTY_ALERT = "novelty"  # Something changed — back to full attention


@dataclass
class SensoryTile:
    """A tile produced by a nerve fiber — a perceived signal.

    Attributes:
        pattern_id: Hash of the input pattern this tile was produced from.
        features: Extracted features from the raw signal.
        confidence: How confident the fiber is in this perception (0.0-1.0).
        source_fiber: ID of the nerve fiber that produced this tile.
        state: The fiber's state when producing this tile.
        timestamp: When this tile was produced.
    """

    pattern_id: str
    features: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source_fiber: str = ""
    state: FiberState = FiberState.PERCEIVING
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"SensoryTile(pattern={self.pattern_id[:8]}..., "
            f"conf={self.confidence:.2f}, state={self.state.value})"
        )


class NerveFiber:
    """A sensory pathway that processes raw signals through a micro-model.

    The fiber adapts over time: new signals get full attention, repeated
    patterns get compiled into automatic processing, and novel patterns
    trigger re-examination.

    Args:
        fiber_id: Unique identifier for this fiber.
        model_type: Type of micro-model (e.g. "jepa", "mobilenet_v4",
            "vit_tiny", "fastconformer_tiny", "smolLM2_135m").
        adapt_threshold: Confidence threshold to transition from ADAPTING
            to COMPILED (default 0.95 — like the 97.5% compile threshold).
        novelty_threshold: How different a signal must be to trigger
            NOVELTY_ALERT when in COMPILED state (default 0.3).
        epsilon: Confidence increment per repeated observation (default 0.05).
    """

    def __init__(
        self,
        fiber_id: str,
        model_type: str = "generic",
        adapt_threshold: float = 0.95,
        novelty_threshold: float = 0.3,
        epsilon: float = 0.05,
    ) -> None:
        self.fiber_id = fiber_id
        self.model_type = model_type
        self.adapt_threshold = adapt_threshold
        self.novelty_threshold = novelty_threshold
        self.epsilon = epsilon

        self._state = FiberState.PERCEIVING
        self._confidence: float = 0.0
        self._observations: dict[str, int] = {}  # pattern_id → count
        self._compiled_patterns: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._last_signal_hash: Optional[str] = None
        self._total_signals: int = 0
        self._compiled_signals: int = 0

    def __repr__(self) -> str:
        return (
            f"NerveFiber(id={self.fiber_id!r}, type={self.model_type}, "
            f"state={self._state.value}, conf={self._confidence:.2f})"
        )

    @property
    def state(self) -> FiberState:
        """Current lifecycle state."""
        return self._state

    @property
    def confidence(self) -> float:
        """Current confidence level."""
        return self._confidence

    @property
    def stats(self) -> dict[str, Any]:
        """Fiber statistics."""
        return {
            "fiber_id": self.fiber_id,
            "model_type": self.model_type,
            "state": self._state.value,
            "confidence": self._confidence,
            "total_signals": self._total_signals,
            "compiled_signals": self._compiled_signals,
            "compiled_patterns": len(self._compiled_patterns),
            "compile_rate": (
                self._compiled_signals / self._total_signals
                if self._total_signals > 0
                else 0.0
            ),
        }

    @staticmethod
    def _hash_signal(signal: Any) -> str:
        """Deterministic hash of a signal for pattern matching.

        PERFORMANCE: Uses fast path for numpy arrays (tobytes + hash)
        instead of expensive str() + SHA-256. Preserves SHA-256 for strings.
        """
        if isinstance(signal, np.ndarray):
            # Fast path: numpy array → raw bytes → non-cryptographic hash
            # 1000× faster than str(signal) + sha256 for float arrays
            return str(hash(signal.tobytes()) % (2**63))
        # Fallback: SHA-256 for strings and other types
        raw = str(signal).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _hash_signal_fast(signal: Any) -> str:
        """Ultra-fast hash — used when we only need a cache key, not a digest."""
        if isinstance(signal, np.ndarray):
            return str(hash(signal.tobytes()) % (2**63))
        return str(hash(str(signal)) % (2**63))

    @staticmethod
    def _fingerprint_signal(signal: Any) -> int:
        """Compute a 64-bit bitvector fingerprint for similarity matching.

        Uses eisenstein_embed when available for proper TUTOR-style
        bitvector fingerprints. Falls back to a simple hash otherwise.
        """
        if HAS_EISENSTEIN:
            try:
                return text_fingerprint(str(signal), use_stemming=True)
            except Exception:
                pass
        # Fallback: truncate SHA-256 to 64 bits
        raw = str(signal).encode("utf-8")
        return int(hashlib.sha256(raw).hexdigest()[:16], 16)

    def perceive(self, signal: Any) -> SensoryTile:
        """Process a raw signal through this nerve fiber.

        The fiber's behavior depends on its lifecycle state:
        - PERCEIVING: Full attention, extract features, build confidence.
        - ADAPTING: Pattern matching, increment confidence via epsilon.
        - COMPILED: Automatic processing, only alert on novelty.
        - NOVELTY_ALERT: Re-examine with full attention.

        Args:
            signal: The raw signal to perceive.

        Returns:
            A SensoryTile with extracted features and confidence.
        """
        pattern_id = self._hash_signal(signal)

        with self._lock:
            self._total_signals += 1
            self._observations[pattern_id] = self._observations.get(pattern_id, 0) + 1

            # Check if we have a compiled path for this pattern
            if (
                pattern_id in self._compiled_patterns
                and self._state == FiberState.COMPILED
            ):
                # Compiled — automatic processing (like not noticing your shoes)
                self._compiled_signals += 1
                compiled = self._compiled_patterns[pattern_id]

                # Check for novelty — is the signal different enough?
                if self._last_signal_hash and pattern_id != self._last_signal_hash:
                    self._last_signal_hash = pattern_id
                    # Same compiled path, different signal — mild novelty check
                    return SensoryTile(
                        pattern_id=pattern_id,
                        features=compiled,
                        confidence=1.0,
                        source_fiber=self.fiber_id,
                        state=FiberState.COMPILED,
                    )

                self._last_signal_hash = pattern_id
                return SensoryTile(
                    pattern_id=pattern_id,
                    features=compiled,
                    confidence=1.0,
                    source_fiber=self.fiber_id,
                    state=FiberState.COMPILED,
                )

            # Not compiled — process with full attention
            if self._state == FiberState.COMPILED:
                # Novelty detected — something changed
                self._state = FiberState.NOVELTY_ALERT
                self._confidence = 0.0

            if self._state in (FiberState.PERCEIVING, FiberState.NOVELTY_ALERT):
                # Full attention — extract features
                features = self._extract_features(signal)
                self._confidence = min(self._confidence + self.epsilon, 1.0)

                if self._confidence >= 0.1:
                    self._state = FiberState.ADAPTING

                tile = SensoryTile(
                    pattern_id=pattern_id,
                    features=features,
                    confidence=self._confidence,
                    source_fiber=self.fiber_id,
                    state=self._state,
                )

            elif self._state == FiberState.ADAPTING:
                # Increment confidence via epsilon accumulation
                self._confidence = min(self._confidence + self.epsilon, 1.0)
                features = self._extract_features(signal)

                if self._confidence >= self.adapt_threshold:
                    # SNAP — compile this pattern
                    self._compiled_patterns[pattern_id] = features
                    self._state = FiberState.COMPILED
                    self._compiled_signals += 1

                tile = SensoryTile(
                    pattern_id=pattern_id,
                    features=features,
                    confidence=self._confidence,
                    source_fiber=self.fiber_id,
                    state=self._state,
                )
            else:
                # Fallback
                features = self._extract_features(signal)
                tile = SensoryTile(
                    pattern_id=pattern_id,
                    features=features,
                    confidence=self._confidence,
                    source_fiber=self.fiber_id,
                    state=self._state,
                )

            self._last_signal_hash = pattern_id
            return tile

    def _extract_features(self, signal: Any) -> dict[str, Any]:
        """Extract features from a raw signal.

        Uses ecosystem packages when available:
        - eisenstein_embed for bitvector fingerprint similarity
        - device_router for device-aware feature metadata

        PERFORMANCE:
        - Fast path for numpy arrays: numeric stats, no str() conversion
        - Feature cache: same signal → same features (avoids recomputation)
        - String path: SHA-256 preserved for text signals
        """
        # Cache key: fast hash of the signal
        cache_key = self._hash_signal_fast(signal)
        if hasattr(self, "_feature_cache") and cache_key in self._feature_cache:
            return self._feature_cache[cache_key].copy()

        if not hasattr(self, "_feature_cache"):
            self._feature_cache = {}

        # --- Fast path: numpy array ---
        if isinstance(signal, np.ndarray):
            features = {
                "shape": signal.shape,
                "dtype": str(signal.dtype),
                "mean": float(np.mean(signal)),
                "std": float(np.std(signal)),
                "min": float(np.min(signal)),
                "max": float(np.max(signal)),
                "nonzero": int(np.count_nonzero(signal)),
                "hash_prefix": self._hash_signal(signal)[:16],
            }
        else:
            # --- String/text path ---
            signal_str = str(signal)
            features = {
                "length": len(signal_str),
                "type": type(signal).__name__,
                "hash_prefix": self._hash_signal(signal)[:16],
                "contains_digits": any(c.isdigit() for c in signal_str),
                "contains_alpha": any(c.isalpha() for c in signal_str),
            }

            # Bitvector fingerprint via eisenstein-embed
            if HAS_EISENSTEIN:
                try:
                    fp = self._fingerprint_signal(signal)
                    features["bitvector_fingerprint"] = fp
                    features["bitvector_hex"] = f"{fp:016x}"
                except Exception as exc:
                    logger.debug("eisenstein fingerprint failed: %s", exc)

            # Device routing metadata via device-router
            router = _get_device_router()
            if router is not None:
                try:
                    overview = router.overview()
                    features["device_cuda"] = overview.get("cuda", {}).get(
                        "available", False
                    )
                    features["device_igpu"] = overview.get("igpu", {}).get(
                        "available", False
                    )
                except Exception as exc:
                    logger.debug("device-router overview failed: %s", exc)

        # Cache and return
        self._feature_cache[cache_key] = features
        return features.copy()

    def reset(self) -> None:
        """Reset the fiber to PERCEIVING state."""
        with self._lock:
            self._state = FiberState.PERCEIVING
            self._confidence = 0.0
            self._observations.clear()
            self._compiled_patterns.clear()
            self._last_signal_hash = None
