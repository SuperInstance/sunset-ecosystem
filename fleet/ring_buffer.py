"""Circular ring buffer for fixed-size log and telemetry streams.

Stores items in a fixed-capacity circular buffer. When full, oldest items
are overwritten. Used for recent-event windows, rolling logs, and telemetry
capture where unbounded growth is not acceptable.

Usage:
    buf = RingBuffer(capacity=100)
    buf.append("event-1")
    buf.append("event-2")
    assert len(buf) == 2
    assert buf[0] == "event-1"
"""

from __future__ import annotations

from collections import deque
from typing import Any, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Fixed-capacity circular buffer.

    :param capacity: Maximum number of items to retain.
    """

    def __init__(self, capacity: int = 1000):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def append(self, item: T) -> None:
        """Append an item, evicting oldest if at capacity."""
        self._buf.append(item)

    def extend(self, items: List[T]) -> None:
        """Append multiple items."""
        for item in items:
            self._buf.append(item)

    def peek(self) -> Optional[T]:
        """Return the oldest item without removing."""
        return self._buf[0] if self._buf else None

    def pop(self) -> Optional[T]:
        """Remove and return the oldest item."""
        return self._buf.popleft() if self._buf else None

    def clear(self) -> None:
        """Clear the buffer."""
        self._buf.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, index: int) -> T:
        return self._buf[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._buf)

    def __contains__(self, item: object) -> bool:
        return item in self._buf

    def is_full(self) -> bool:
        return len(self._buf) == self._capacity

    def to_list(self) -> List[T]:
        return list(self._buf)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def __repr__(self) -> str:
        return f"<RingBuffer {len(self._buf)}/{self._capacity}>"
