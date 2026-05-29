"""Merkle tree for efficient state comparison and synchronization.

Builds a hash tree over a list of items. Enables O(log N) verification
of which items differ between two nodes — useful for fleet state sync.

Usage:
    tree = MerkleTree(["a", "b", "c", "d"])
    root = tree.root_hash()
    # Compare roots to detect differences, then drill down
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MerkleNode:
    """A node in the Merkle tree."""

    hash: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    leaf_index: int = -1  # only set for leaf nodes


class MerkleTree:
    """
    Binary Merkle tree with SHA-256 leaf hashing.

    :param items: List of string items to hash into leaves.
    """

    def __init__(self, items: List[str]):
        self._items = list(items)
        self._root = self._build_tree(items)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_tree(self, items: List[str]) -> Optional[MerkleNode]:
        if not items:
            return None
        leaves = [self._leaf_hash(i, item) for i, item in enumerate(items)]
        return self._build_level(leaves)

    def _build_level(self, nodes: List[MerkleNode]) -> MerkleNode:
        if len(nodes) == 1:
            return nodes[0]
        next_level: List[MerkleNode] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else left
            parent_hash = self._hash_pair(left.hash, right.hash)
            next_level.append(
                MerkleNode(hash=parent_hash, left=left, right=right)
            )
        return self._build_level(next_level)

    def _leaf_hash(self, index: int, item: str) -> MerkleNode:
        h = hashlib.sha256(f"leaf:{index}:{item}".encode("utf-8")).hexdigest()
        return MerkleNode(hash=h, leaf_index=index)

    def _hash_pair(self, left: str, right: str) -> str:
        combined = f"node:{left}:{right}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def root_hash(self) -> Optional[str]:
        return self._root.hash if self._root else None

    def leaf_hash_at(self, index: int) -> Optional[str]:
        if 0 <= index < len(self._items):
            return self._leaf_hash(index, self._items[index]).hash
        return None

    def proof(self, index: int) -> List[Tuple[str, bool]]:
        """
        Generate a Merkle proof for the leaf at *index*.

        Returns list of (sibling_hash, is_left) tuples where is_left
        indicates whether the leaf was in the left subtree at that level.
        """
        if not self._root or index < 0 or index >= len(self._items):
            return []
        proof: List[Tuple[str, bool]] = []
        self._collect_proof(self._root, index, proof)
        return proof

    def verify_proof(self, index: int, item: str, proof: List[Tuple[str, bool]]) -> bool:
        """Verify a Merkle proof for an item."""
        current = self._leaf_hash(index, item).hash
        for sibling, is_left in reversed(proof):
            if is_left:
                current = self._hash_pair(current, sibling)
            else:
                current = self._hash_pair(sibling, current)
        return current == self.root_hash()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def diff_indices(self, other: "MerkleTree") -> List[int]:
        """Find indices where leaf hashes differ."""
        if not self._root or not other._root:
            return list(range(max(len(self._items), len(other._items))))
        if self.root_hash() == other.root_hash():
            return []
        return self._diff_recursive(self._root, other._root)

    def _diff_recursive(
        self, a: Optional[MerkleNode], b: Optional[MerkleNode]
    ) -> List[int]:
        if a is None and b is None:
            return []
        if a is None or b is None:
            # One tree is deeper at this path
            return self._collect_leaves(a or b)
        if a.hash == b.hash:
            return []
        if a.leaf_index >= 0 and b.leaf_index >= 0:
            # Both are leaves and hashes differ
            return [a.leaf_index]
        # Recurse into children
        left_diff = self._diff_recursive(a.left, b.left)
        right_diff = self._diff_recursive(a.right, b.right)
        return left_diff + right_diff

    def _collect_leaves(self, node: Optional[MerkleNode]) -> List[int]:
        if node is None:
            return []
        if node.leaf_index >= 0:
            return [node.leaf_index]
        return self._collect_leaves(node.left) + self._collect_leaves(node.right)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_proof(self, node: MerkleNode, index: int, proof: List[Tuple[str, bool]]) -> None:
        if node.leaf_index >= 0:
            return
        mid = self._count_leaves(node.left) if node.left else 0
        if index < mid and node.left:
            if node.right:
                proof.append((node.right.hash, True))
            self._collect_proof(node.left, index, proof)
        elif node.right:
            if node.left:
                proof.append((node.left.hash, False))
            self._collect_proof(node.right, index - mid, proof)

    def _count_leaves(self, node: Optional[MerkleNode]) -> int:
        if node is None:
            return 0
        if node.leaf_index >= 0:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def __repr__(self) -> str:
        return f"<MerkleTree leaves={len(self._items)} root={self.root_hash()[:8] if self._root else 'None'}>"
