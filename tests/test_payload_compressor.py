"""Tests for payload_compressor.py — Payload compression with algorithm selection.

Run: python3 -m pytest tests/test_payload_compressor.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.payload_compressor import PayloadCompressor


class TestPayloadCompressor:
    def test_create(self):
        comp = PayloadCompressor()
        assert comp.stats()["compressions"] == 0

    def test_compress_gzip(self):
        comp = PayloadCompressor()
        data = b"Hello, World! " * 100
        compressed = comp.compress(data, algorithm="gzip")
        assert len(compressed) < len(data)
        assert comp.stats()["compressions"] == 1

    def test_decompress_gzip(self):
        comp = PayloadCompressor()
        data = b"Hello, World! " * 100
        compressed = comp.compress(data, algorithm="gzip")
        decompressed = comp.decompress(compressed, algorithm="gzip")
        assert decompressed == data

    def test_compress_deflate(self):
        comp = PayloadCompressor()
        data = b"Test payload" * 50
        compressed = comp.compress(data, algorithm="deflate")
        decompressed = comp.decompress(compressed, algorithm="deflate")
        assert decompressed == data

    def test_compress_zlib(self):
        comp = PayloadCompressor()
        data = b"Test payload" * 50
        compressed = comp.compress(data, algorithm="zlib")
        decompressed = comp.decompress(compressed, algorithm="zlib")
        assert decompressed == data

    def test_default_algorithm(self):
        comp = PayloadCompressor(default_algorithm="zlib")
        data = b"Test payload" * 50
        compressed = comp.compress(data)
        decompressed = comp.decompress(compressed)
        assert decompressed == data

    def test_select_algorithm_small(self):
        comp = PayloadCompressor()
        assert comp.select_algorithm(b"tiny") == "identity"

    def test_select_algorithm_large(self):
        comp = PayloadCompressor()
        assert comp.select_algorithm(b"x" * 1000) == "gzip"

    def test_auto_compress(self):
        comp = PayloadCompressor()
        data = b"x" * 1000
        compressed, algo = comp.auto_compress(data)
        assert algo == "gzip"
        assert len(compressed) < len(data)

    def test_auto_compress_identity(self):
        comp = PayloadCompressor()
        data = b"tiny"
        compressed, algo = comp.auto_compress(data)
        assert algo == "identity"
        assert compressed == data

    def test_stats(self):
        comp = PayloadCompressor()
        data = b"Hello, World! " * 100
        compressed = comp.compress(data)
        comp.decompress(compressed)
        stats = comp.stats()
        assert stats["compressions"] == 1
        assert stats["decompressions"] == 1
        assert stats["bytes_in"] == len(data)
        assert stats["compression_ratio"] > 0

    def test_repr(self):
        comp = PayloadCompressor()
        assert "PayloadCompressor" in repr(comp)
