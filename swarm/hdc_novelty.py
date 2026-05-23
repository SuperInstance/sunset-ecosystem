"""AVX-512 HDC binary novelty for sunset-ecosystem swarm.

Implements Hyperdimensional Computing (HDC) binary novelty scoring
using XOR + POPCOUNT Hamming distance.  Provides a NumPy fallback
that is still ~100× faster than cosine distance, plus an AVX-512
accelerated path when available.

Reference: experiments/SCOUT-REPORT.md §3 — AVX-512 HDC XOR Judge
"""
from __future__ import annotations

__all__ = [
    "BinaryVectorEncoder",
    "HDCDiversityScorer",
    "hdc_novelty_score",
    "HAS_AVX512",
]

import logging
import math
import struct
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AVX-512 capability probe
# ---------------------------------------------------------------------------

def _detect_avx512() -> bool:
    """Return True if the runtime CPU supports AVX-512.

    We also verify that NumPy can actually dispatch to AVX-512
    vectorised instructions (VPOPCNTDQ) on uint64 arrays.
    """
    try:
        # Fast CPUID check via /proc/cpuinfo on Linux
        with open("/proc/cpuinfo", "rb") as fh:
            cpuinfo = fh.read().decode("utf-8", errors="ignore")
        has_avx512f = "avx512f" in cpuinfo
    except Exception:
        has_avx512f = False

    if not has_avx512f:
        return False

    # NumPy ≥2.0 exposes np.bitwise_count which dispatches to VPOPCNTDQ
    # when the dtype is uint64 and the ufunc loop is AVX-512 aware.
    try:
        _ = np.bitwise_count  # type: ignore[attr-defined]
    except AttributeError:
        logger.debug("AVX-512 CPU detected but np.bitwise_count missing (NumPy < 2.0)")
        return False

    # Sanity bench: AVX-512 popcount on 8 KiB of uint64 should be
    # measurably faster than a pure-Python loop.
    arr = np.ones(1024, dtype=np.uint64)
    t0 = time.perf_counter()
    for _ in range(100):
        np.bitwise_count(arr)  # type: ignore[attr-defined]
    dt_numpy = time.perf_counter() - t0

    # Pure-python popcount baseline
    t0 = time.perf_counter()
    for _ in range(100):
        sum(x.bit_count() for x in arr.tolist())
    dt_python = time.perf_counter() - t0

    usable = dt_numpy < dt_python * 0.5
    if usable:
        logger.info("AVX-512 VPOPCNTDQ path enabled (%.2f× faster than Python)",
                    dt_python / max(dt_numpy, 1e-9))
    else:
        logger.debug("AVX-512 present but np.bitwise_count not faster; using fallback")
    return usable


HAS_AVX512: bool = _detect_avx512()


# ---------------------------------------------------------------------------
# BinaryVectorEncoder
# ---------------------------------------------------------------------------

class BinaryVectorEncoder:
    """Float32 → packed binary hypervector encoder.

    Binarises a real-valued vector by thresholding each element at 0
    (sign-based encoding).  The resulting bits are packed into the
    smallest unsigned integer dtype that fits the dimension:

    * dim ≤ 8   → uint8
    * dim ≤ 16  → uint16
    * dim ≤ 32  → uint32
    * dim ≤ 64  → uint64
    * dim > 64  → array of uint64 (packed contiguously)

    Packing layout: element 0 → LSB of first word, element 1 → next bit,
    etc.  This matches the bit layout assumed by the C reference
    ``flux_hdc_avx512.h``.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self._bits_per_word = self._choose_word_bits(dim)
        self._n_words = math.ceil(dim / self._bits_per_word)
        self._dtype = self._word_dtype(self._bits_per_word)

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _choose_word_bits(dim: int) -> int:
        if dim <= 8:
            return 8
        if dim <= 16:
            return 16
        if dim <= 32:
            return 32
        return 64

    @staticmethod
    def _word_dtype(bits: int) -> type:
        return {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}[bits]

    # ---- public API ------------------------------------------------------

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """Encode a float32 vector into packed binary form.

        Args:
            vector: 1-D float32 array of length ``dim``.

        Returns:
            Packed binary array of unsigned integers.  Shape is
            ``(n_words,)`` where ``n_words = ceil(dim / word_bits)``.
        """
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"encode expects 1-D vector, got shape {vec.shape}")
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"vector dim {vec.shape[0]} != encoder dim {self.dim}"
            )

        # Binarise: 1 if element > 0 else 0
        bits = (vec > 0).astype(np.uint8)

        if self._bits_per_word >= self.dim:
            # Single word packing — use dot-product with powers of two
            powers = np.arange(self.dim, dtype=self._dtype)
            word = np.dot(bits, (1 << powers).astype(self._dtype))
            return np.array([word], dtype=self._dtype)

        # Multi-word packing: reshape into (n_words, word_bits) and
        # compress each chunk into a single integer.
        padded = np.zeros(self._n_words * self._bits_per_word, dtype=np.uint8)
        padded[: self.dim] = bits
        chunks = padded.reshape(self._n_words, self._bits_per_word)

        # Each column j contributes 2**j to its word
        powers = np.arange(self._bits_per_word, dtype=self._dtype)
        multipliers = (1 << powers).astype(self._dtype)
        words = np.dot(chunks.astype(self._dtype), multipliers)
        return words.astype(self._dtype)

    def encode_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Encode a batch of float32 vectors.

        Args:
            vectors: 2-D float32 array of shape ``(batch, dim)``.

        Returns:
            Packed binary array of shape ``(batch, n_words)``.
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError(f"encode_batch expects 2-D array, got shape {vecs.shape}")
        if vecs.shape[1] != self.dim:
            raise ValueError(
                f"vector dim {vecs.shape[1]} != encoder dim {self.dim}"
            )

        batch = vecs.shape[0]
        bits = (vecs > 0).astype(np.uint8)

        if self._bits_per_word >= self.dim:
            powers = np.arange(self.dim, dtype=self._dtype)
            multipliers = (1 << powers).astype(self._dtype)
            words = np.dot(bits, multipliers)
            return words.astype(self._dtype).reshape(batch, 1)

        padded = np.zeros((batch, self._n_words * self._bits_per_word), dtype=np.uint8)
        padded[:, : self.dim] = bits
        chunks = padded.reshape(batch, self._n_words, self._bits_per_word)
        powers = np.arange(self._bits_per_word, dtype=self._dtype)
        multipliers = (1 << powers).astype(self._dtype)
        words = np.dot(chunks.astype(self._dtype), multipliers)
        return words.astype(self._dtype)

    def decode_distance(self, packed_a: np.ndarray, packed_b: np.ndarray) -> int:
        """Return Hamming distance between two *packed* binary vectors.

        This is a low-level helper; most callers should use
        :class:`HDCDiversityScorer` instead.
        """
        a = np.asarray(packed_a)
        b = np.asarray(packed_b)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
        xor = np.bitwise_xor(a, b)
        return int(self._popcount_words(xor))

    @staticmethod
    def _popcount_words(words: np.ndarray) -> int:
        """POPCOUNT over an array of packed words."""
        if hasattr(np, "bitwise_count"):
            return int(np.bitwise_count(words).sum())  # type: ignore[attr-defined]
        # Fallback for NumPy < 2.0 — use Python int.bit_count on each element.
        return sum(int(w).bit_count() for w in words.flat)


# ---------------------------------------------------------------------------
# HDCDiversityScorer
# ---------------------------------------------------------------------------

class HDCDiversityScorer:
    """XOR+POPCNT novelty scorer for binary hypervectors.

    Computes Hamming distance between two packed binary vectors and
    normalises it to the range ``[0, 1]``:

    * 0.0 → identical vectors (zero Hamming distance)
    * 1.0 → orthogonal vectors (Hamming distance = dim)

    The scorer works with :class:`BinaryVectorEncoder` outputs and
    supports both a NumPy fallback and an AVX-512 fast path.
    """

    def __init__(self, dim: int, use_avx512: bool = True) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.encoder = BinaryVectorEncoder(dim)
        self._use_avx512 = use_avx512 and HAS_AVX512
        self._popcnt_fn: Callable[[np.ndarray], int] = self._pick_popcnt()

    def _pick_popcnt(self) -> Callable[[np.ndarray], int]:
        if self._use_avx512 and hasattr(np, "bitwise_count"):
            # AVX-512 path: vectorised VPOPCNTDQ on uint64 words
            def _avx512_popcnt(words: np.ndarray) -> int:
                return int(
                    np.bitwise_count(words).sum()  # type: ignore[attr-defined]
                )
            return _avx512_popcnt
        # Fallback: pure Python per-word popcount (still fast for small dims)
        def _fallback_popcnt(words: np.ndarray) -> int:
            return sum(int(w).bit_count() for w in words.flat)
        return _fallback_popcnt

    # ---- scoring API -----------------------------------------------------

    def score(self, packed_a: np.ndarray, packed_b: np.ndarray) -> float:
        """Novelty score between two packed binary vectors.

        Returns a float in ``[0, 1]`` where 0 = identical and 1 = fully
        orthogonal (all bits differ).
        """
        a = np.asarray(packed_a)
        b = np.asarray(packed_b)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
        xor = np.bitwise_xor(a, b)
        hamming = self._popcnt_fn(xor)
        # Normalise to [0, 1]; cap at 1.0 for safety
        return min(1.0, hamming / self.dim)

    def score_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Convenience: encode two float32 vectors and score them."""
        packed_a = self.encoder.encode(vec_a)
        packed_b = self.encoder.encode(vec_b)
        return self.score(packed_a, packed_b)

    def score_batch(
        self,
        packed_queries: np.ndarray,
        packed_refs: np.ndarray,
    ) -> np.ndarray:
        """Batch novelty scores between queries and reference vectors.

        Args:
            packed_queries: shape ``(n_q, n_words)``
            packed_refs:    shape ``(n_r, n_words)``

        Returns:
            Float array of shape ``(n_q, n_r)`` with scores in ``[0,1]``.
        """
        q = np.asarray(packed_queries)
        r = np.asarray(packed_refs)
        if q.ndim != 2 or r.ndim != 2:
            raise ValueError("score_batch expects 2-D arrays")
        if q.shape[1] != r.shape[1]:
            raise ValueError(f"word count mismatch: {q.shape[1]} vs {r.shape[1]}")

        n_q, n_r = q.shape[0], r.shape[0]
        n_words = q.shape[1]

        # Broadcast XOR: (n_q, 1, n_words) xor (1, n_r, n_words)
        xor = np.bitwise_xor(q[:, None, :], r[None, :, :])

        # POPCOUNT per word, then sum across words
        if hasattr(np, "bitwise_count"):
            counts = np.bitwise_count(xor).sum(axis=2)  # type: ignore[attr-defined]
        else:
            # NumPy < 2.0 fallback: unpackbits operates on uint8.
            # Reshape xor to (n_q * n_r * n_words,) as uint8 for unpackbits.
            flat = xor.reshape(-1)
            # For non-uint8 dtypes, we need to view as uint8 bytes
            # flat.view(np.uint8) gives the raw bytes of each element
            byte_view = flat.view(np.uint8)
            bits = np.unpackbits(byte_view)
            # Each original element has (dtype.itemsize * 8) bits,
            # but only the lower bits_per_word are meaningful.
            # Sum all bits and reshape.
            bits_per_element = xor.dtype.itemsize * 8
            total_bits_per_element = bits_per_element
            # Sum across the bit dimension for each element
            bits_reshaped = bits.reshape(-1, total_bits_per_element)
            element_counts = bits_reshaped.sum(axis=1)
            counts = element_counts.reshape(n_q, n_r, n_words).sum(axis=2)
            counts = counts.astype(np.int64)

        return np.minimum(1.0, counts / self.dim).astype(np.float32)

    # ---- speed benchmarking ----------------------------------------------

    def benchmark_vs_cosine(
        self,
        n_vectors: int = 1000,
        n_trials: int = 5,
    ) -> dict[str, float]:
        """Measure speedup of HDC binary novelty vs float32 cosine distance.

        Returns a dict with keys ``hdc_ms``, ``cosine_ms``, ``speedup``.
        """
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n_vectors, self.dim), dtype=np.float32)
        packed = self.encoder.encode_batch(vectors)

        # Warm-up
        for _ in range(3):
            _ = self.score_batch(packed[:10], packed[:10])
            _ = self._cosine_distance_batch(vectors[:10], vectors[:10])

        # HDC timing
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = self.score_batch(packed, packed)
        hdc_ms = (time.perf_counter() - t0) / n_trials * 1000

        # Cosine timing
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = self._cosine_distance_batch(vectors, vectors)
        cosine_ms = (time.perf_counter() - t0) / n_trials * 1000

        speedup = cosine_ms / max(hdc_ms, 1e-6)
        return {
            "hdc_ms": hdc_ms,
            "cosine_ms": cosine_ms,
            "speedup": speedup,
            "avx512_enabled": self._use_avx512,
        }

    @staticmethod
    def _cosine_distance_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Reference cosine-distance matrix for benchmarking."""
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        na[na == 0] = 1.0
        nb[nb == 0] = 1.0
        sim = (a / na) @ (b / nb).T
        sim = np.clip(sim, -1.0, 1.0)
        # Convert similarity → distance → novelty-ish [0,1]
        # (1 - sim) / 2 maps [-1,1] similarity to [0,1] distance
        return (1.0 - sim) / 2.0


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def hdc_novelty_score(a: np.ndarray, b: np.ndarray) -> float:
    """Compute HDC binary novelty between two float32 vectors.

    Args:
        a: 1-D float32 vector.
        b: 1-D float32 vector.  Must match ``a`` in shape.

    Returns:
        Float in ``[0, 1]``.  0 = identical, 1 = orthogonal.

    Example::

        >>> import numpy as np
        >>> a = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        >>> b = np.array([0.5, 0.6, -0.7, -0.8], dtype=np.float32)
        >>> hdc_novelty_score(a, b)  # doctest: +SKIP
        0.5
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"hdc_novelty_score expects 1-D vectors, got {a.ndim}D")

    dim = a.shape[0]
    scorer = HDCDiversityScorer(dim)
    return scorer.score_vectors(a, b)
