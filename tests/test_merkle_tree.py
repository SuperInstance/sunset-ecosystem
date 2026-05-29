"""Tests for merkle_tree.py — Merkle tree for state comparison.

Run: python3 -m pytest tests/test_merkle_tree.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.merkle_tree import MerkleTree


class TestMerkleTree:
    def test_create(self):
        tree = MerkleTree(["a", "b", "c"])
        assert tree.root_hash() is not None

    def test_root_consistent(self):
        tree1 = MerkleTree(["a", "b"])
        tree2 = MerkleTree(["a", "b"])
        assert tree1.root_hash() == tree2.root_hash()

    def test_root_changes(self):
        tree1 = MerkleTree(["a", "b"])
        tree2 = MerkleTree(["a", "c"])
        assert tree1.root_hash() != tree2.root_hash()

    def test_empty_tree(self):
        tree = MerkleTree([])
        assert tree.root_hash() is None

    def test_leaf_hash_at(self):
        tree = MerkleTree(["a", "b"])
        assert tree.leaf_hash_at(0) is not None
        assert tree.leaf_hash_at(1) is not None
        assert tree.leaf_hash_at(2) is None

    def test_proof(self):
        tree = MerkleTree(["a", "b", "c", "d"])
        proof = tree.proof(0)
        assert len(proof) > 0
        assert isinstance(proof[0], tuple)

    def test_verify_proof(self):
        tree = MerkleTree(["a", "b", "c", "d"])
        proof = tree.proof(1)
        assert tree.verify_proof(1, "b", proof) is True
        assert tree.verify_proof(1, "wrong", proof) is False

    def test_diff_indices(self):
        tree1 = MerkleTree(["a", "b", "c"])
        tree2 = MerkleTree(["a", "x", "c"])
        diff = tree1.diff_indices(tree2)
        assert diff == [1]

    def test_diff_no_difference(self):
        tree1 = MerkleTree(["a", "b", "c"])
        tree2 = MerkleTree(["a", "b", "c"])
        diff = tree1.diff_indices(tree2)
        assert diff == []

    def test_diff_different_lengths(self):
        tree1 = MerkleTree(["a", "b"])
        tree2 = MerkleTree(["a", "b", "c"])
        diff = tree1.diff_indices(tree2)
        assert 2 in diff

    def test_repr(self):
        tree = MerkleTree(["a"])
        assert "MerkleTree" in repr(tree)
