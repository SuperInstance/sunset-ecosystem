"""bloom_filter.py — Python bindings for bit-packed Bloom filter.

Wraps the C implementation in nerve/bloom_filter.so.
"""
from __future__ import annotations

__all__ = ["BloomFilter"]

import ctypes
import hashlib
from pathlib import Path

# Load shared library
_so_path = Path(__file__).parent / "bloom_filter.so"
if not _so_path.exists():
    _so_path = Path(__file__).parent / "bloom_filter.so"
_bf = ctypes.CDLL(str(_so_path))

_bf.bf_create.argtypes = [ctypes.c_uint64, ctypes.c_double]
_bf.bf_create.restype = ctypes.c_void_p

_bf.bf_destroy.argtypes = [ctypes.c_void_p]
_bf.bf_destroy.restype = None

_bf.bf_num_bits.argtypes = [ctypes.c_void_p]
_bf.bf_num_bits.restype = ctypes.c_uint64

_bf.bf_num_hashes.argtypes = [ctypes.c_void_p]
_bf.bf_num_hashes.restype = ctypes.c_uint64

_bf.bf_num_items.argtypes = [ctypes.c_void_p]
_bf.bf_num_items.restype = ctypes.c_uint64

_bf.bf_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_bf.bf_add.restype = None

_bf.bf_test.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_bf.bf_test.restype = ctypes.c_int

_bf.bf_clear.argtypes = [ctypes.c_void_p]
_bf.bf_clear.restype = None


class BloomFilter:
    """Bit-packed Bloom filter — probabilistic set membership.

    Backed by C implementation with FNV-1a + Murmur-like multi-hashing.
    False positive rate configurable at creation.
    """

    def __init__(self, expected_items: int = 10_000, false_positive_rate: float = 0.01) -> None:
        self._handle = _bf.bf_create(expected_items, false_positive_rate)
        if not self._handle:
            raise MemoryError("Failed to create bloom filter")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _bf.bf_destroy(self._handle)
            self._handle = None

    # ── properties ────────────────────────────────────────

    @property
    def num_bits(self) -> int:
        return int(_bf.bf_num_bits(self._handle))

    @property
    def num_hashes(self) -> int:
        return int(_bf.bf_num_hashes(self._handle))

    @property
    def num_items(self) -> int:
        return int(_bf.bf_num_items(self._handle))

    @property
    def size_bytes(self) -> int:
        return self.num_bits // 8

    # ── core operations ─────────────────────────────────

    def add(self, key: str | bytes) -> None:
        if isinstance(key, str):
            key = key.encode("utf-8")
        _bf.bf_add(self._handle, key, len(key))

    def __contains__(self, key: str | bytes) -> bool:
        if isinstance(key, str):
            key = key.encode("utf-8")
        return bool(_bf.bf_test(self._handle, key, len(key)))

    def test(self, key: str | bytes) -> bool:
        return key in self

    def clear(self) -> None:
        _bf.bf_clear(self._handle)

    # ── bulk helpers ──────────────────────────────────────

    def update(self, keys: list[str | bytes]) -> None:
        for k in keys:
            self.add(k)

    def batch_test(self, keys: list[str | bytes]) -> list[bool]:
        return [self.test(k) for k in keys]

    # ── diagnostics ───────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BloomFilter(bits={self.num_bits}, hashes={self.num_hashes}, "
            f"items={self.num_items}, ~{self.size_bytes}B)"
        )

    def estimated_false_positive_rate(self) -> float:
        """Current estimated FPR based on actual item count."""
        n = self.num_items
        m = self.num_bits
        k = self.num_hashes
        if n == 0:
            return 0.0
        return (1.0 - (1.0 - 1.0 / m) ** (k * n)) ** k
