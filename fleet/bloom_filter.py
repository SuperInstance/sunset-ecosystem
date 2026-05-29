"""Bloom filter for probabilistic set membership testing.

Space-efficient structure for testing whether an element is possibly in a set
or definitely not. Used for deduplication, cache filtering, and quick existence
checks where false positives are acceptable.

Usage:
    bf = BloomFilter(capacity=10000, fp_rate=0.01)
    bf.add("item-1")
    assert "item-1" in bf
    assert "item-2" not in bf  # probably
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable


class BloomFilter:
    """
    Bloom filter with configurable capacity and false-positive rate.

    :param capacity: Expected number of elements.
    :param fp_rate: Target false-positive rate (e.g. 0.01 = 1%).
    """

    def __init__(self, capacity: int = 1000, fp_rate: float = 0.01):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not 0 < fp_rate < 1:
            raise ValueError("fp_rate must be in (0, 1)")

        self._capacity = capacity
        self._fp_rate = fp_rate

        # Optimal bit array size and hash count
        self._size = self._optimal_size(capacity, fp_rate)
        self._hash_count = self._optimal_hash_count(self._size, capacity)

        self._bits = bytearray(self._size // 8 + 1)
        self._count = 0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        return max(1, int(m / n * math.log(2)))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, item: Any) -> None:
        """Add an item to the filter."""
        for idx in self._hashes(item):
            self._set_bit(idx)
        self._count += 1

    def update(self, items: Iterable[Any]) -> None:
        """Add multiple items."""
        for item in items:
            self.add(item)

    def __contains__(self, item: Any) -> bool:
        """Check if item is probably in the filter."""
        return all(self._get_bit(idx) for idx in self._hashes(item))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of items added."""
        return self._count

    @property
    def bit_size(self) -> int:
        return self._size

    @property
    def hash_count(self) -> int:
        return self._hash_count

    def estimated_fp_rate(self) -> float:
        """Estimate current false-positive rate based on fill."""
        return (1 - math.exp(-self._hash_count * self._count / self._size)) ** self._hash_count

    def fill_ratio(self) -> float:
        """Proportion of bits that are set."""
        set_bits = sum(bin(b).count("1") for b in self._bits)
        return set_bits / self._size if self._size else 0.0

    def clear(self) -> None:
        """Reset the filter."""
        self._bits = bytearray(self._size // 8 + 1)
        self._count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hashes(self, item: Any) -> Iterable[int]:
        """Generate hash positions for an item."""
        item_bytes = str(item).encode("utf-8")
        h1 = int(hashlib.md5(item_bytes).hexdigest(), 16)
        h2 = int(hashlib.sha256(item_bytes).hexdigest(), 16)
        for i in range(self._hash_count):
            yield (h1 + i * h2) % self._size

    def _set_bit(self, idx: int) -> None:
        byte_idx = idx // 8
        bit_idx = idx % 8
        self._bits[byte_idx] |= 1 << bit_idx

    def _get_bit(self, idx: int) -> bool:
        byte_idx = idx // 8
        bit_idx = idx % 8
        return bool(self._bits[byte_idx] & (1 << bit_idx))

    def __repr__(self) -> str:
        return (
            f"<BloomFilter size={self._size} "
            f"hashes={self._hash_count} items={self._count}>"
        )
