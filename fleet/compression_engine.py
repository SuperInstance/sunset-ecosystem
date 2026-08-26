from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class CompressionEngine:
    """
    Compression engine for large payloads.

    Supports gzip and identity compression with size tracking.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._stats: Dict[str, int] = {
            "compressed": 0,
            "decompressed": 0,
            "bytes_saved": 0,
        }

    def compress(self, data: str, level: int = 6) -> bytes:
        """Compress string data with gzip."""
        original = len(data.encode())
        compressed = gzip.compress(data.encode(), compresslevel=level)
        compressed_size = len(compressed)
        self._stats["compressed"] += 1
        self._stats["bytes_saved"] += original - compressed_size
        return compressed

    def decompress(self, data: bytes) -> str:
        """Decompress gzip data to string."""
        decompressed = gzip.decompress(data).decode()
        self._stats["decompressed"] += 1
        return decompressed

    def compress_json(self, obj: Dict[str, Any], level: int = 6) -> bytes:
        """Compress a JSON-serializable object."""
        return self.compress(json.dumps(obj), level=level)

    def decompress_json(self, data: bytes) -> Dict[str, Any]:
        """Decompress gzip data to JSON object."""
        return json.loads(self.decompress(data))

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            **self._stats,
            "avg_bytes_saved": self._stats["bytes_saved"]
            / max(1, self._stats["compressed"]),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
