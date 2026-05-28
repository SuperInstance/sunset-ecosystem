"""Tests for bloom_filter.c via ctypes bindings.

Compile first: gcc -shared -fPIC -O3 -o nerve/bloom_filter.so nerve/bloom_filter.c -lm
Run: python3 -m pytest tests/test_bloom_filter.py -v --tb=short
"""
from __future__ import annotations

import pytest

from nerve.bloom_filter_wrapper import BloomFilter


class TestBloomFilterBasics:
    def test_create_properties(self):
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        assert bf.num_bits > 0
        assert bf.num_hashes > 0
        assert bf.num_items == 0
        assert bf.size_bytes > 0

    def test_add_and_test(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("hello")
        assert "hello" in bf
        assert bf.test("hello")

    def test_missing_item(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("present")
        assert "missing" not in bf
        assert not bf.test("missing")

    def test_multiple_items(self):
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        keys = [f"key_{i}" for i in range(100)]
        for k in keys:
            bf.add(k)
        for k in keys:
            assert k in bf, f"{k} should be present"

    def test_item_count(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        for i in range(50):
            bf.add(f"item_{i}")
        assert bf.num_items == 50

    def test_clear(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("item")
        assert "item" in bf
        bf.clear()
        assert bf.num_items == 0
        assert "item" not in bf

    def test_bytes_key(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add(b"\x00\x01\x02\x03")
        assert b"\x00\x01\x02\x03" in bf
        assert b"\x00\x01\x02\x04" not in bf

    def test_update_and_batch(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.update(["a", "b", "c"])
        assert "a" in bf
        assert "b" in bf
        assert "c" in bf
        results = bf.batch_test(["a", "z", "b"])
        assert results == [True, False, True]

    def test_repr(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        r = repr(bf)
        assert "BloomFilter" in r
        assert "bits=" in r
        assert "hashes=" in r

    def test_estimated_fpr(self):
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        assert bf.estimated_false_positive_rate() == pytest.approx(0.0, abs=1e-10)
        for i in range(500):
            bf.add(f"item_{i}")
        fpr = bf.estimated_false_positive_rate()
        assert 0.0 < fpr < 1.0

    def test_no_false_negatives(self):
        """A Bloom filter should never have false negatives."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.1)
        added = {f"key_{i}" for i in range(500)}
        for k in added:
            bf.add(k)
        for k in added:
            assert k in bf, f"False negative for {k}"

    def test_large_scale(self):
        bf = BloomFilter(expected_items=100_000, false_positive_rate=0.01)
        for i in range(10_000):
            bf.add(f"large_key_{i}")
        assert bf.num_items == 10_000
        # All should be present
        for i in range(10_000):
            assert f"large_key_{i}" in bf
