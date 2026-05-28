"""Distributed ID generator (ULID-style monotonic sortable IDs).

Generates 128-bit time-ordered unique identifiers without coordination.
Compatible with Snowflake's philosophy but simpler: 48-bit timestamp + 80-bit random.

Usage:
    gen = IDGenerator(node_id=1)
    uid = gen.generate()  # e.g. "01J2X3Y4Z5A6B7C8D9E0F1G2H3"

Properties:
- Sortable by generation time (prefix is millisecond timestamp).
- ~1.2e24 possible values per millisecond (80 bits random).
- No central coordination needed (node_id embedded for traceability).
- String representation is Crockford Base32 (URL-safe, unambiguous).
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Crockford Base32 alphabet (excludes I, L, O, U to avoid ambiguity)
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_base32(data: bytes) -> str:
    """Encode bytes to Crockford Base32 (no padding, uppercase)."""
    val = int.from_bytes(data, "big")
    result = []
    for _ in range((len(data) * 8 + 4) // 5):
        result.append(_CROCKFORD[val & 0x1F])
        val >>= 5
    return "".join(reversed(result))


class IDGenerator:
    """
    Generate time-ordered unique identifiers.

    :param node_id: A 16-bit identifier for this node (0-65535).
    :param clock: Optional monotonic time source (for testing).
    """

    def __init__(
        self,
        node_id: int = 0,
        clock: Optional[Callable[[], float]] = None,
    ):
        if not (0 <= node_id <= 0xFFFF):
            raise ValueError("node_id must be 0-65535")
        self._node_id = node_id & 0xFFFF
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._last_time: int = 0
        self._counter: int = 0
        self._stats: Dict[str, int] = {"generated": 0, "collisions_avoided": 0}

    def generate(self) -> str:
        """Generate a new unique ID."""
        with self._lock:
            now_ms = int(self._clock() * 1000)
            if now_ms == self._last_time:
                self._counter += 1
                if self._counter >= 4096:
                    # Wait 1ms if counter exhausted
                    self._collisions_avoided += 1
                    time.sleep(0.001)
                    now_ms = int(self._clock() * 1000)
                    self._counter = 0
            else:
                self._last_time = now_ms
                self._counter = 0

            # 48-bit timestamp (6 bytes)
            ts_bytes = now_ms.to_bytes(6, "big")

            # 16-bit node id + 12-bit counter + 52-bit random = 80 bits (10 bytes)
            counter_bytes = ((self._node_id << 12) | self._counter).to_bytes(3, "big")
            random_bytes = secrets.token_bytes(7)

            payload = ts_bytes + counter_bytes + random_bytes
            self._stats["generated"] += 1
            return _encode_base32(payload)

    def extract_timestamp(self, uid: str) -> float:
        """Extract the Unix timestamp (seconds) embedded in the ID."""
        # First 10 chars encode exactly 48 bits = the 6-byte timestamp
        ts_ms = _decode_base32(uid[:10])
        return ts_ms / 1000.0

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"<IDGenerator node={self._node_id} generated={self._stats['generated']}>"


def _decode_base32(s: str) -> int:
    """Decode Crockford Base32 string to integer."""
    val = 0
    for ch in s.upper():
        idx = _CROCKFORD.index(ch)
        val = (val << 5) | idx
    return val


class IDBatchGenerator(IDGenerator):
    """Generate IDs in batches for high-throughput scenarios."""

    def generate_batch(self, count: int) -> list:
        """Generate *count* IDs atomically."""
        with self._lock:
            results = []
            for _ in range(count):
                now_ms = int(self._clock() * 1000)
                if now_ms == self._last_time:
                    self._counter += 1
                    if self._counter >= 4096:
                        time.sleep(0.001)
                        now_ms = int(self._clock() * 1000)
                        self._counter = 0
                        self._last_time = now_ms
                else:
                    self._last_time = now_ms
                    self._counter = 0

                ts_bytes = now_ms.to_bytes(6, "big")
                counter_bytes = ((self._node_id << 12) | self._counter).to_bytes(3, "big")
                random_bytes = secrets.token_bytes(7)
                payload = ts_bytes + counter_bytes + random_bytes
                results.append(_encode_base32(payload))
                self._stats["generated"] += 1
            return results
