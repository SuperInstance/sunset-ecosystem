"""
Consistent Hash Ring (Ketama-style) for Fleet Node Task Distribution
===================================================================

Distributes "breeding tasks" (or any key-space workload) across a dynamic
fleet of nodes using a consistent hash ring with virtual node replication.

Key Properties:
- Virtual node replication: each physical node is mapped to V virtual nodes
  on the ring, smoothing out load imbalance when node counts are low.
- Minimal remapping on churn: when a node is added or removed, only ~1/N of
  keys move (where N = node count), unlike modulo-based hashing which can
  cause near-total remapping.
- MD5-based 32-bit hash: compatible with the original Ketama memcached
  client algorithm, yielding uniform distribution across the 2^32 space.

Usage:
    ring = HashRing()
    ring.add_node("node-1", weight=1)
    ring.add_node("node-2", weight=2)  # 2x capacity
    ring.add_node("node-3", weight=1)

    node = ring.get_node("task:breed:alpha-42")
    # -> "node-2" (with probability proportional to weight)

    ring.remove_node("node-2")
    # Only ~1/4 of keys previously on node-2 move; the rest stay put.

Reference:
- https://github.com/RJ/ketama (original C implementation)
- "Consistent Hashing and Random Trees" (Karger et al., 1997)
"""

from __future__ import annotations

import bisect
import hashlib
import struct
from typing import Dict, Iterable, List, Optional, Tuple


class HashRing:
    """
    Ketama-style consistent hash ring with virtual node replication.

    Each physical node is mapped to `replicas * weight` virtual points on a
    circular 32-bit hash space.  When looking up a key, we hash it to the same
    space and walk clockwise until we hit the first virtual point; the owning
    physical node is returned.

    The ring is kept as two parallel sorted lists:
        _keys   : sorted 32-bit hash values (int)
        _nodes  : physical node name corresponding to each hash value
    This makes lookup O(log V) where V = total virtual nodes.
    """

    DEFAULT_REPLICAS = 160  # Virtual nodes per physical node per weight unit.
                          # 160 is the classic Ketama default for memcached.

    def __init__(self, replicas: int = DEFAULT_REPLICAS):
        self._replicas = replicas

        # Sorted structures (kept in sync)
        self._keys: List[int] = []   # sorted 32-bit hash integers
        self._nodes: List[str] = []  # physical node name per hash

        # Metadata
        self._weights: Dict[str, int] = {}          # configured weight per node
        self._node_hashes: Dict[str, List[int]] = {}  # hashes owned by each node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, node: str, weight: int = 1) -> None:
        """
        Add a physical node to the ring.

        :param node:   unique node identifier (e.g. "<BOAT_IP>:8847")
        :param weight: capacity weight.  weight=2 receives ~2x the keys of
                       weight=1, but uses 2x virtual nodes (more memory).
        """
        if node in self._weights:
            # Idempotent: remove first, then re-add with new weight.
            self.remove_node(node)

        self._weights[node] = weight
        hashes: List[int] = []

        total_replicas = self._replicas * weight
        for replica_idx in range(total_replicas):
            h = _ketama_hash(f"{node}:{replica_idx}")
            hashes.append(h)

        self._node_hashes[node] = hashes
        self._insert_hashes(node, hashes)

    def remove_node(self, node: str) -> None:
        """
        Remove a physical node from the ring.

        Only keys that mapped to this node will be reassigned; all other keys
        keep their existing mapping.
        """
        if node not in self._weights:
            return

        hashes = self._node_hashes.pop(node, [])
        self._delete_hashes(hashes)
        self._weights.pop(node, None)

    def get_node(self, key: str) -> Optional[str]:
        """
        Return the physical node responsible for *key*.

        Walks clockwise from the key's hash position on the ring until the
        first virtual node is encountered.

        :returns: node name, or None if the ring is empty.
        """
        if not self._keys:
            return None

        h = _key_hash(key)
        idx = bisect.bisect_right(self._keys, h)
        if idx == len(self._keys):
            idx = 0  # wrap around the ring
        return self._nodes[idx]

    def get_nodes(self, key: str, n: int = 3) -> List[str]:
        """
        Return the top *n* distinct physical nodes for *key*, walking clockwise.

        Useful for replica placement (primary + backups).
        """
        if n <= 0:
            return []
        if not self._keys:
            return []

        h = _key_hash(key)
        idx = bisect.bisect_right(self._keys, h)
        seen: set = set()
        result: List[str] = []

        for _ in range(len(self._keys)):
            if idx == len(self._keys):
                idx = 0
            node = self._nodes[idx]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) == n:
                    break
            idx += 1

        return result

    def iterate_nodes(self, key: str):
        """
        Generator yielding distinct physical nodes clockwise from *key*.
        """
        if not self._keys:
            return

        h = _key_hash(key)
        idx = bisect.bisect_right(self._keys, h)
        seen: set = set()

        for _ in range(len(self._keys)):
            if idx == len(self._keys):
                idx = 0
            node = self._nodes[idx]
            if node not in seen:
                seen.add(node)
                yield node
            idx += 1

    # ------------------------------------------------------------------
    # Ring introspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[str]:
        """List of currently registered physical nodes."""
        return list(self._weights.keys())

    @property
    def node_count(self) -> int:
        return len(self._weights)

    @property
    def virtual_node_count(self) -> int:
        return len(self._keys)

    def get_key_distribution(self) -> Dict[str, int]:
        """
        Return the number of virtual-node arcs owned by each physical node.
        Proportional to the share of hash space each node covers.
        """
        dist: Dict[str, int] = {}
        for node in self._nodes:
            dist[node] = dist.get(node, 0) + 1
        return dist

    def get_weight(self, node: str) -> int:
        return self._weights.get(node, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_hashes(self, node: str, hashes: Iterable[int]) -> None:
        """Insert sorted (hash, node) pairs into the ring structures."""
        for h in hashes:
            idx = bisect.bisect_left(self._keys, h)
            self._keys.insert(idx, h)
            self._nodes.insert(idx, node)

    def _delete_hashes(self, hashes: Iterable[int]) -> None:
        """Remove hash values from the ring structures."""
        for h in hashes:
            idx = bisect.bisect_left(self._keys, h)
            # Safety: verify the slot matches.
            if idx < len(self._keys) and self._keys[idx] == h:
                self._keys.pop(idx)
                self._nodes.pop(idx)

    def __repr__(self) -> str:
        return (
            f"<HashRing nodes={self.node_count} "
            f"virtual={self.virtual_node_count}>"
        )


# ----------------------------------------------------------------------
# Hash functions (Ketama-compatible)
# ----------------------------------------------------------------------

def _ketama_hash(point: str) -> int:
    """
    Compute a 32-bit unsigned hash for a ring point (node:replica).

    Uses MD5, then unpacks the first 4 bytes as a little-endian uint32.
    This matches the original Ketama C implementation:
        digest = md5(point)
        value  = digest[0..3] as little-endian uint32
    """
    digest = hashlib.md5(point.encode("utf-8")).digest()
    # struct.unpack returns a signed int; mask to force unsigned.
    return struct.unpack("<I", digest[:4])[0]


def _key_hash(key: str) -> int:
    """
    Compute a 32-bit unsigned hash for a user key.

    Same algorithm as ring points so everything lives in the same space.
    """
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return struct.unpack("<I", digest[:4])[0]


# ====================================================================
# Example / self-test
# ====================================================================

if __name__ == "__main__":
    # 1. Build a ring with 4 fleet nodes, heterogeneous weights.
    ring = HashRing(replicas=120)
    ring.add_node("plato-<BOAT_IP>:8847", weight=2)
    ring.add_node("traps-147.224.38.132:4042", weight=1)
    ring.add_node("health-147.224.38.133:8080", weight=1)
    ring.add_node("oracle-147.224.38.134:5000", weight=1)

    print(ring)
    print("Distribution:", ring.get_key_distribution())

    # 2. Map 10,000 breeding tasks and measure load balance.
    import random
    from collections import Counter

    task_prefix = "task:breed:alpha"
    counts: Counter = Counter()
    for i in range(10_000):
        node = ring.get_node(f"{task_prefix}-{i:05d}")
        counts[node] += 1

    print("\nTask distribution (10k keys):")
    for node, c in counts.most_common():
        pct = c / 10000 * 100
        print(f"  {node}: {c} ({pct:.1f}%)")

    # 3. Remove a node and measure key churn.
    removed = "traps-147.224.38.132:4042"
    before = {}
    for i in range(10_000):
        before[i] = ring.get_node(f"{task_prefix}-{i:05d}")

    ring.remove_node(removed)
    moved = 0
    for i in range(10_000):
        after = ring.get_node(f"{task_prefix}-{i:05d}")
        if after != before[i]:
            moved += 1

    print(f"\nAfter removing {removed}:")
    print(f"  Keys moved: {moved} / 10,000 ({moved/100:.1f}%)")
    print(f"  Theoretical ideal: ~{100/ring.node_count:.1f}%")

    # 4. Replica placement example (primary + 2 backups).
    sample_task = "task:breed:beta-9999"
    nodes = ring.get_nodes(sample_task, n=3)
    print(f"\nReplica placement for '{sample_task}':")
    for rank, node in enumerate(nodes, start=1):
        print(f"  rank {rank}: {node}")
