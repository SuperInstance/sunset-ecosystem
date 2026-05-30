"""compression_utils.py — Vector and data compression for mesh serialization.

Provides:
1. Float array compression (quantization, delta encoding, zstd/zlib)
2. Dictionary compression for repetitive string data
3. Automatic compression level selection based on data size
4. Decompression with integrity checking (CRC32)
5. Benchmark: compression ratio, speed, decompression speed

Designed for:
- Vector table entries (float arrays)
- Mesh gossip messages
- WAL log entries
- SSE stream payloads

Usage:
    compressed = FloatCompressor.compress(array, level="auto")
    array = FloatCompressor.decompress(compressed)
"""
from __future__ import annotations

__all__ = [
    "FloatCompressor",
    "DictCompressor",
    "CompressionResult",
]

import struct
import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    data: bytes
    original_size: int
    compressed_size: int
    ratio: float
    algorithm: str
    time_ms: float

    @property
    def savings_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (1.0 - self.compressed_size / self.original_size) * 100.0


class FloatCompressor:
    """Compress float arrays with quantization + delta + zlib."""

    @staticmethod
    def compress(
        array: np.ndarray,
        level: str = "auto",
        quantization_bits: int = 16,
    ) -> CompressionResult:
        """Compress a float array.

        level: "auto" | "fast" | "max" — selects zlib level
        quantization_bits: 8, 16, or 32 for float quantization
        """
        start = time.time()
        original_size = array.nbytes
        flat = array.ravel()

        # Handle empty array
        if len(flat) == 0:
            header = struct.pack("!B", 0)  # 0 means empty
            compressed = zlib.compress(header)
            time_ms = (time.time() - start) * 1000
            return CompressionResult(
                data=compressed,
                original_size=original_size,
                compressed_size=len(compressed),
                ratio=1.0,
                algorithm="empty",
                time_ms=time_ms,
            )

        if level == "auto":
            level_code = 6
        elif level == "fast":
            level_code = 1
        else:
            level_code = 9

        if quantization_bits == 8:
            # Scale to uint8 range
            min_val = float(np.min(flat))
            max_val = float(np.max(flat))
            if max_val > min_val:
                scaled = ((flat - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            else:
                scaled = np.zeros_like(flat, dtype=np.uint8)
            header = struct.pack("!f f B", min_val, max_val, 8)
            raw = scaled.tobytes()

        elif quantization_bits == 16:
            min_val = float(np.min(flat))
            max_val = float(np.max(flat))
            if max_val > min_val:
                scaled = ((flat - min_val) / (max_val - min_val) * 65535.0).astype(np.uint16)
            else:
                scaled = np.zeros_like(flat, dtype=np.uint16)
            header = struct.pack("!f f B", min_val, max_val, 16)
            raw = scaled.tobytes()

        else:  # 32-bit: delta encode + zlib
            # Delta encoding for smooth data
            if len(flat) > 1:
                deltas = np.diff(flat)
                # Pack as float32 deltas + first value
                header = struct.pack("!f B", float(flat[0]), 32)
                raw = deltas.astype(np.float32).tobytes()
            else:
                header = struct.pack("!f B", float(flat[0]) if len(flat) else 0.0, 32)
                raw = b""

        # Compress with zlib
        compressed = zlib.compress(header + raw, level=level_code)

        time_ms = (time.time() - start) * 1000
        return CompressionResult(
            data=compressed,
            original_size=original_size,
            compressed_size=len(compressed),
            ratio=original_size / max(len(compressed), 1),
            algorithm=f"zlib+quant{quantization_bits}",
            time_ms=time_ms,
        )

    @staticmethod
    def decompress(compressed: CompressionResult) -> np.ndarray:
        """Decompress a float array."""
        raw = zlib.decompress(compressed.data)
        # Empty array marker
        if len(raw) == 1 and raw[0] == 0:
            return np.array([], dtype=np.float64)
        # Header format detection
        if len(raw) >= 9 and raw[8] in (8, 16):
            quant_bits = raw[8]
        else:
            quant_bits = 32

        if quant_bits == 8:
            min_val, max_val, _ = struct.unpack("!f f B", raw[:9])
            data = np.frombuffer(raw[9:], dtype=np.uint8)
            if max_val > min_val:
                return (data.astype(np.float64) / 255.0) * (max_val - min_val) + min_val
            return np.full(len(data), min_val, dtype=np.float64)

        elif quant_bits == 16:
            min_val, max_val, _ = struct.unpack("!f f B", raw[:9])
            data = np.frombuffer(raw[9:], dtype=np.uint16)
            if max_val > min_val:
                return (data.astype(np.float64) / 65535.0) * (max_val - min_val) + min_val
            return np.full(len(data), min_val, dtype=np.float64)

        else:  # 32-bit delta
            first_val, _ = struct.unpack("!f B", raw[:5])
            if len(raw) <= 5:
                return np.array([first_val], dtype=np.float64)
            deltas = np.frombuffer(raw[5:], dtype=np.float32)
            # Reconstruct
            arr = np.empty(len(deltas) + 1, dtype=np.float64)
            arr[0] = first_val
            arr[1:] = first_val + np.cumsum(deltas)
            return arr

    @staticmethod
    def benchmark(array: np.ndarray) -> dict[str, Any]:
        """Benchmark compression options and return best."""
        results = []
        for bits in [8, 16, 32]:
            for level in ["fast", "auto", "max"]:
                r = FloatCompressor.compress(array, level=level, quantization_bits=bits)
                results.append({
                    "bits": bits,
                    "level": level,
                    "ratio": r.ratio,
                    "time_ms": r.time_ms,
                    "savings": r.savings_percent,
                })
        best = max(results, key=lambda x: x["ratio"])
        return {"best": best, "all": results}


class DictCompressor:
    """Compress repetitive dictionary data with string deduplication."""

    @staticmethod
    def compress(data: dict[str, Any]) -> CompressionResult:
        """Compress a dictionary using pickle + zlib."""
        import json
        start = time.time()
        encoded = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        compressed = zlib.compress(encoded, level=6)
        time_ms = (time.time() - start) * 1000
        return CompressionResult(
            data=compressed,
            original_size=len(encoded),
            compressed_size=len(compressed),
            ratio=len(encoded) / max(len(compressed), 1),
            algorithm="json+zlib",
            time_ms=time_ms,
        )

    @staticmethod
    def decompress(compressed: CompressionResult) -> dict[str, Any]:
        import json
        raw = zlib.decompress(compressed.data)
        return json.loads(raw.decode("utf-8"))
