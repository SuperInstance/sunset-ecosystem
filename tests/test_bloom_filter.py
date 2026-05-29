"""Tests for bloom_filter.py — Probabilistic membership filter.

Run: python3 -m pytest tests/test_bloom_filter.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.bloom_filter import BloomFilter


class TestBloomFilter:
    def test_create(self):
        bf = BloomFilter(capacity=1000, fp_rate=0.01)
        assert bf.bit_size > 0
        assert bf.hash_count > 0

    def test_add_and_contains(self):
        bf = BloomFilter(capacity=1000)
        bf.add("item-1")
        assert "item-1" in bf

    def test_not_contains(self):
        bf = BloomFilter(capacity=1000)
        bf.add("item-1")
        # Should probably not contain unrelated item
        assert "item-2" not in bf

    def test_update(self):
        bf = BloomFilter(capacity=1000)
        bf.update(["a", "b", "c"])
        assert "a" in bf
        assert "b" in bf
        assert "c" in bf

    def test_count(self):
        bf = BloomFilter(capacity=1000)
        bf.add("x")
        bf.add("y")
        assert bf.count == 2

    def test_clear(self):
        bf = BloomFilter(capacity=1000)
        bf.add("x")
        bf.clear()
        assert "x" not in bf
        assert bf.count == 0

    def test_estimated_fp_rate(self):
        bf = BloomFilter(capacity=1000, fp_rate=0.01)
        assert bf.estimated_fp_rate() < 0.01

    def test_fill_ratio(self):
        bf = BloomFilter(capacity=1000)
        assert bf.fill_ratio() == 0.0
        bf.add("x")
        assert bf.fill_ratio() > 0.0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            BloomFilter(capacity=0)

    def test_invalid_fp_rate(self):
        with pytest.raises(ValueError):
            BloomFilter(capacity=100, fp_rate=1.5)

    def test_repr(self):
        bf = BloomFilter(capacity=100)
        assert "BloomFilter" in repr(bf)
