"""Tests for crdt_hdc_hybrid.py — CRDT + diversity merge."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from swarm.crdt_hdc_hybrid import merge_with_diversity
from swarm.flux_vector_table import FluxVectorTable, AgentVector, AgentMeta


class TestMergeWithDiversity:
    def test_merge_empty_tables(self):
        local = FluxVectorTable(dim=8)
        remote = FluxVectorTable(dim=8)
        merged = merge_with_diversity(local, remote)
        assert isinstance(merged, FluxVectorTable)
        assert merged.dim == 8

    def test_merge_single_agent_no_diversity_check(self):
        local = FluxVectorTable(dim=8)
        local.add(AgentVector(1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], AgentMeta(1, "a", 0.5)))
        remote = FluxVectorTable(dim=8)
        merged = merge_with_diversity(local, remote)
        assert 1 in merged._meta

    def test_merge_diversity_flag_set(self):
        local = FluxVectorTable(dim=8)
        local.add(AgentVector(1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], AgentMeta(1, "a", 0.5)))
        local.add(AgentVector(2, [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], AgentMeta(2, "b", 0.5)))

        remote = FluxVectorTable(dim=8)
        merged = merge_with_diversity(local, remote, diversity_threshold=0.5)

        assert 1 in merged._meta
        assert 2 in merged._meta
        # Novelty scores should be set
        assert "novelty_score" in merged._meta[1].extra
        assert "diversity_flag" in merged._meta[1].extra
        assert isinstance(merged._meta[1].extra["diversity_flag"], bool)

    def test_merge_combines_both_sources(self):
        local = FluxVectorTable(dim=8)
        local.add(AgentVector(1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], AgentMeta(1, "a", 0.5)))

        remote = FluxVectorTable(dim=8)
        remote.add(AgentVector(2, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], AgentMeta(2, "b", 0.5)))

        merged = merge_with_diversity(local, remote)
        assert 1 in merged._meta
        assert 2 in merged._meta
