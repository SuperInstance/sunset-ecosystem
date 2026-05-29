import pytest
from fleet.compression_engine import CompressionEngine


class TestCompressionEngine:
    def test_init(self):
        ce = CompressionEngine()
        assert ce.fleet_node_id == "default"

    def test_compress_decompress(self):
        ce = CompressionEngine()
        original = "hello world " * 100
        compressed = ce.compress(original)
        assert len(compressed) < len(original.encode())
        decompressed = ce.decompress(compressed)
        assert decompressed == original

    def test_compress_json(self):
        ce = CompressionEngine()
        obj = {"name": "test", "values": [1, 2, 3]}
        compressed = ce.compress_json(obj)
        decompressed = ce.decompress_json(compressed)
        assert decompressed == obj

    def test_get_stats(self):
        ce = CompressionEngine()
        ce.compress("hello world " * 100)
        stats = ce.get_stats()
        assert stats["compressed"] == 1
        assert stats["bytes_saved"] > 0

    def test_to_dict(self):
        ce = CompressionEngine()
        d = ce.to_dict()
        assert "stats" in d
