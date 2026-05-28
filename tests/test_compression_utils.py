"""Tests for compression_utils.py — Vector and data compression.

Run: python3 -m pytest tests/test_compression_utils.py -v --tb=short
"""
from __future__ import annotations

import numpy as np
import pytest

from logos.compression_utils import (
    CompressionResult,
    DictCompressor,
    FloatCompressor,
)


class TestFloatCompressor:
    def test_compress_decompress_8bit(self):
        arr = np.linspace(0.0, 100.0, 1000)
        result = FloatCompressor.compress(arr, quantization_bits=8)
        assert result.ratio > 1.0
        decompressed = FloatCompressor.decompress(result)
        assert len(decompressed) == len(arr)
        # 8-bit quantization over range 0-100: step ~0.39, max error ~0.2
        np.testing.assert_allclose(decompressed, arr, atol=0.4)

    def test_compress_decompress_16bit(self):
        arr = np.linspace(0.0, 100.0, 1000)
        result = FloatCompressor.compress(arr, quantization_bits=16)
        assert result.ratio > 1.0
        decompressed = FloatCompressor.decompress(result)
        assert len(decompressed) == len(arr)
        np.testing.assert_allclose(decompressed, arr, atol=0.002)

    def test_compress_decompress_32bit(self):
        arr = np.random.randn(1000)
        result = FloatCompressor.compress(arr, quantization_bits=32)
        assert result.ratio > 1.0
        decompressed = FloatCompressor.decompress(result)
        assert len(decompressed) == len(arr)
        np.testing.assert_allclose(decompressed, arr, atol=1e-5)

    def test_compression_levels(self):
        arr = np.random.randn(1000)
        fast = FloatCompressor.compress(arr, level="fast")
        auto = FloatCompressor.compress(arr, level="auto")
        max_c = FloatCompressor.compress(arr, level="max")
        # max should have best or equal ratio
        assert max_c.ratio >= auto.ratio * 0.9

    def test_savings_percent(self):
        arr = np.zeros(1000)
        result = FloatCompressor.compress(arr, quantization_bits=8)
        # Zeros compress extremely well
        assert result.savings_percent > 50.0

    def test_empty_array(self):
        arr = np.array([], dtype=np.float64)
        result = FloatCompressor.compress(arr, quantization_bits=32)
        decompressed = FloatCompressor.decompress(result)
        assert len(decompressed) == 0

    def test_single_element(self):
        arr = np.array([42.0])
        result = FloatCompressor.compress(arr, quantization_bits=16)
        decompressed = FloatCompressor.decompress(result)
        assert len(decompressed) == 1
        assert decompressed[0] == pytest.approx(42.0, abs=0.01)

    def test_benchmark(self):
        arr = np.random.randn(10000)
        bench = FloatCompressor.benchmark(arr)
        assert "best" in bench
        assert "all" in bench
        assert len(bench["all"]) == 9  # 3 bits × 3 levels


class TestDictCompressor:
    def test_compress_decompress(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        result = DictCompressor.compress(data)
        assert result.ratio > 0.5
        decompressed = DictCompressor.decompress(result)
        assert decompressed == data

    def test_large_dict(self):
        data = {f"key_{i}": i * 2 for i in range(1000)}
        result = DictCompressor.compress(data)
        assert result.original_size > 0
        assert result.compressed_size > 0


class TestCompressionResult:
    def test_create(self):
        r = CompressionResult(
            data=b"test",
            original_size=100,
            compressed_size=50,
            ratio=2.0,
            algorithm="test",
            time_ms=1.0,
        )
        assert r.savings_percent == 50.0

    def test_zero_original(self):
        r = CompressionResult(
            data=b"",
            original_size=0,
            compressed_size=0,
            ratio=1.0,
            algorithm="test",
            time_ms=0.0,
        )
        assert r.savings_percent == 0.0
