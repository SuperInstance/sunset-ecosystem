"""CRDT-based conflict resolution for distributed fleet state.

Provides last-write-wins (LWW) registers, G-Counter (grow-only counter),
PN-Counter (increment/decrement), and OR-Set (observed-remove set) for
eventually consistent distributed state.

Usage:
    reg = LWWRegister(node_id="node-1")
    reg.set(42)
    other = LWWRegister(node_id="node-2")
    other.merge(reg)
    print(other.value)  # 42
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class CRDT:
    """Base class for CRDTs."""

    def merge(self, other: "CRDT") -> None:
        raise NotImplementedError

    def value(self) -> Any:
        raise NotImplementedError


# ------------------------------------------------------------------
# LWW Register
# ------------------------------------------------------------------

@dataclass
class LWWRegister(CRDT):
    """Last-Write-Wins register."""

    node_id: str
    _value: Any = None
    _timestamp: float = field(default_factory=lambda: -float("inf"))

    def set(self, value: Any) -> None:
        now = time.time()
        if now > self._timestamp:
            self._value = value
            self._timestamp = now

    def merge(self, other: "LWWRegister") -> None:
        if other._timestamp > self._timestamp:
            self._value = other._value
            self._timestamp = other._timestamp

    def value(self) -> Any:
        return self._value


# ------------------------------------------------------------------
# G-Counter
# ------------------------------------------------------------------

class GCounter(CRDT):
    """Grow-only counter (monotonic increment)."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._counts: Dict[str, int] = {node_id: 0}

    def increment(self, amount: int = 1) -> None:
        self._counts[self.node_id] += amount

    def merge(self, other: "GCounter") -> None:
        for node, count in other._counts.items():
            self._counts[node] = max(self._counts.get(node, 0), count)

    def value(self) -> int:
        return sum(self._counts.values())

    def state(self) -> Dict[str, int]:
        return dict(self._counts)


# ------------------------------------------------------------------
# PN-Counter
# ------------------------------------------------------------------

class PNCounter(CRDT):
    """Positive-Negative counter (increment and decrement)."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._p: Dict[str, int] = {node_id: 0}
        self._n: Dict[str, int] = {node_id: 0}

    def increment(self, amount: int = 1) -> None:
        self._p[self.node_id] += amount

    def decrement(self, amount: int = 1) -> None:
        self._n[self.node_id] += amount

    def merge(self, other: "PNCounter") -> None:
        for node, count in other._p.items():
            self._p[node] = max(self._p.get(node, 0), count)
        for node, count in other._n.items():
            self._n[node] = max(self._n.get(node, 0), count)

    def value(self) -> int:
        return sum(self._p.values()) - sum(self._n.values())

    def state(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        return dict(self._p), dict(self._n)


# ------------------------------------------------------------------
# OR-Set
# ------------------------------------------------------------------

class ORSet(CRDT):
    """Observed-Remove Set (add-wins semantics)."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._elements: Dict[Any, Set[Tuple[str, int]]] = {}
        self._tag_counter = 0

    def add(self, element: Any) -> None:
        self._tag_counter += 1
        tag = (self.node_id, self._tag_counter)
        if element not in self._elements:
            self._elements[element] = set()
        self._elements[element].add(tag)

    def remove(self, element: Any) -> None:
        if element in self._elements:
            del self._elements[element]

    def merge(self, other: "ORSet") -> None:
        for element, tags in other._elements.items():
            if element not in self._elements:
                self._elements[element] = set()
            self._elements[element].update(tags)

    def value(self) -> Set[Any]:
        return set(self._elements.keys())

    def contains(self, element: Any) -> bool:
        return element in self._elements

    def __len__(self) -> int:
        return len(self._elements)
