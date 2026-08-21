"""Payload compression with algorithm selection.

Compresses and decompresses payloads with multiple algorithm support
and automatic selection based on content characteristics. Used for
fleet message compression, log batching, and API response optimization.

Usage:
    compressor = PayloadCompressor()
    compressed = compressor.compress("large payload...", algorithm="gzip")
    original = compressor.decompress(compressed, algorithm="gzip")
"""

from __future__ import annotations

import gzip
import zlib
from typing import Any, Dict, List, Optional


class PayloadCompressor:
    """
    Payload compressor with algorithm selection.

    :param default_algorithm: Default compression algorithm.
    """

    def __init__(self, default_algorithm: str = "gzip"):
        self._default = default_algorithm
        self._stats: Dict[str, Any] = {
            "compressions": 0,
            "decompressions": 0,
            "bytes_in": 0,
            "bytes_out": 0,
        }

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(
        self,
        data: bytes,
        algorithm: Optional[str] = None,
        level: int = 6,
    ) -> bytes:
        """
        Compress payload.

        :param data: Raw payload bytes.
        :param algorithm: Compression algorithm (gzip, deflate, zlib).
        :param level: Compression level (1-9).
        :returns: Compressed bytes.
        """
        algo = algorithm or self._default
        self._stats["compressions"] += 1
        self._stats["bytes_in"] += len(data)

        if algo == "gzip":
            result = gzip.compress(data, compresslevel=level)
        elif algo == "deflate":
            compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
            result = compressor.compress(data) + compressor.flush()
        elif algo == "zlib":
            result = zlib.compress(data, level=level)
        else:
            result = data

        self._stats["bytes_out"] += len(result)
        return result

    def decompress(self, data: bytes, algorithm: Optional[str] = None) -> bytes:
        """
        Decompress payload.

        :param data: Compressed bytes.
        :param algorithm: Compression algorithm.
        :returns: Decompressed bytes.
        """
        algo = algorithm or self._default
        self._stats["decompressions"] += 1

        if algo == "gzip":
            return gzip.decompress(data)
        elif algo == "deflate":
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            return decompressor.decompress(data) + decompressor.flush()
        elif algo == "zlib":
            return zlib.decompress(data)
        else:
            return data

    # ------------------------------------------------------------------
    # Algorithm selection
    # ------------------------------------------------------------------

    def select_algorithm(self, data: bytes) -> str:
        """
        Select best algorithm based on payload characteristics.

        :param data: Payload to analyze.
        :returns: Recommended algorithm name.
        """
        if len(data) < 100:
            return "identity"  # Too small to compress
        # Simple heuristic: use gzip for most cases
        return "gzip"

    def auto_compress(self, data: bytes) -> tuple[bytes, str]:
        """
        Auto-select and compress.

        :param data: Raw payload.
        :returns: Tuple of (compressed_data, algorithm_used).
        """
        algo = self.select_algorithm(data)
        if algo == "identity":
            return data, algo
        return self.compress(data, algorithm=algo), algo

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total_in = self._stats["bytes_in"]
        total_out = self._stats["bytes_out"]
        ratio = (1 - total_out / total_in) if total_in > 0 else 0.0
        return {
            "compressions": self._stats["compressions"],
            "decompressions": self._stats["decompressions"],
            "bytes_in": total_in,
            "bytes_out": total_out,
            "compression_ratio": round(ratio, 4),
        }

    def __repr__(self) -> str:
        return f"<PayloadCompressor default={self._default} compressions={self._stats['compressions']}>"
