"""Snowflake-style distributed unique ID generator.

Generates time-ordered, unique 64-bit IDs without coordination.
Inspired by Twitter Snowflake. Used for fleet-wide unique identifiers
that are sortable by time.

Usage:
    gen = IDGenerator(node_id=1)
    id1 = gen.next()
    id2 = gen.next()
    assert id2 > id1
"""

from __future__ import annotations

import threading
import time


class IDGenerator:
    """
    Snowflake-style ID generator.

    64-bit ID layout:
    - 41 bits: timestamp (ms since epoch)
    - 10 bits: node ID (0-1023)
    - 12 bits: sequence (0-4095)

    :param node_id: Unique node identifier (0-1023).
    :param epoch: Custom epoch in ms (default: 2024-01-01).
    """

    NODE_BITS = 10
    SEQUENCE_BITS = 12
    MAX_NODE = (1 << NODE_BITS) - 1
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1
    TIMESTAMP_SHIFT = NODE_BITS + SEQUENCE_BITS
    NODE_SHIFT = SEQUENCE_BITS
    DEFAULT_EPOCH = 1704067200000  # 2024-01-01 00:00:00 UTC

    def __init__(self, node_id: int = 0, epoch: int = DEFAULT_EPOCH):
        if not 0 <= node_id <= self.MAX_NODE:
            raise ValueError(f"node_id must be 0-{self.MAX_NODE}")
        self._node_id = node_id
        self._epoch = epoch
        self._last_timestamp = -1
        self._sequence = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def next(self) -> int:
        """Generate the next unique ID."""
        with self._lock:
            ts = self._current_timestamp()
            if ts < self._last_timestamp:
                # Clock moved backwards, wait until caught up
                ts = self._wait_for_clock(self._last_timestamp)
            if ts == self._last_timestamp:
                self._sequence = (self._sequence + 1) & self.MAX_SEQUENCE
                if self._sequence == 0:
                    ts = self._wait_for_clock(ts + 1)
            else:
                self._sequence = 0
            self._last_timestamp = ts
            return (
                ((ts - self._epoch) << self.TIMESTAMP_SHIFT)
                | (self._node_id << self.NODE_SHIFT)
                | self._sequence
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_timestamp(self) -> int:
        return int(time.time() * 1000)

    def _wait_for_clock(self, target: int) -> int:
        while True:
            ts = self._current_timestamp()
            if ts >= target:
                return ts
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def node_id(self) -> int:
        return self._node_id

    def parse(self, id_val: int) -> dict:
        """Parse a generated ID into components."""
        timestamp = (id_val >> self.TIMESTAMP_SHIFT) + self._epoch
        node = (id_val >> self.NODE_SHIFT) & self.MAX_NODE
        sequence = id_val & self.MAX_SEQUENCE
        return {
            "timestamp": timestamp,
            "node_id": node,
            "sequence": sequence,
        }

    def __repr__(self) -> str:
        return f"<IDGenerator node={self._node_id}>"
