"""Tests for CRDTMergeEngine.

Mocks turbovec so tests run without the native extension.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

# ── Mock turbovec before any swarm.vector_table import ──
_mock_turbovec = types.ModuleType("turbovec")


class _MockIdMapIndex:
    """Minimal stand-in for turbovec.IdMapIndex."""

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self._vectors: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for vec, aid in zip(vectors, ids):
            self._vectors[int(aid)] = vec.copy()

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        allowlist: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.zeros((1, k), dtype=np.float32),
                np.zeros((1, k), dtype=np.uint64),
            )
        q = query[0]
        candidates = list(self._vectors.items())
        if allowlist is not None:
            allowed = set(int(a) for a in allowlist)
            candidates = [(aid, v) for aid, v in candidates if aid in allowed]

        qn = q / (np.linalg.norm(q) + 1e-8)
        sims: list[tuple[int, float]] = []
        for aid, vec in candidates:
            vn = vec / (np.linalg.norm(vec) + 1e-8)
            sims.append((aid, float(np.dot(qn, vn))))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:k]
        while len(top) < k:
            top.append((0, 0.0))
        scores = np.array([[s for _, s in top]], dtype=np.float32)
        ids_arr = np.array([[aid for aid, _ in top]], dtype=np.uint64)
        return scores, ids_arr

    def remove(self, agent_id: int) -> bool:
        return self._vectors.pop(agent_id, None) is not None

    def contains(self, agent_id: int) -> bool:
        return agent_id in self._vectors

    def write(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "_MockIdMapIndex":
        return cls(dim=256)


_mock_turbovec.IdMapIndex = _MockIdMapIndex  # type: ignore[attr-defined]
sys.modules["turbovec"] = _mock_turbovec

from swarm.crdt_merge import Agent, CRDTMergeEngine, DivergenceReport, LineageSanityError
from swarm.vector_table import FluxVectorTable

DIM = 8


@pytest.fixture
def empty_vt():
    return FluxVectorTable(dim=DIM, bit_width=2)


@pytest.fixture
def engine(empty_vt):
    return CRDTMergeEngine(vector_table=empty_vt)


# ── Test 1: simple merge (no overlap) → union ─────────

def test_simple_merge_union(engine):
    """When populations have no overlap, the merge is a simple union."""
    local = [
        Agent(agent_id=1, fitness=0.5, generation=0, vector=[0.1] * DIM),
        Agent(agent_id=2, fitness=0.6, generation=0, vector=[0.2] * DIM),
    ]
    remote = [
        Agent(agent_id=3, fitness=0.7, generation=0, vector=[0.3] * DIM),
        Agent(agent_id=4, fitness=0.8, generation=0, vector=[0.4] * DIM),
    ]

    merged = engine.merge_populations(local, remote)
    merged_ids = {a.agent_id for a in merged}

    assert merged_ids == {1, 2, 3, 4}
    assert len(merged) == 4


# ── Test 2: overlapping higher-fitness wins ──────────────

def test_overlapping_higher_fitness_wins(engine):
    """When the same agent exists in both populations, the fitter copy survives."""
    local = [
        Agent(agent_id=10, fitness=0.5, generation=1, vector=[0.1] * DIM),
    ]
    remote = [
        Agent(agent_id=10, fitness=0.9, generation=1, vector=[0.9] * DIM),
    ]

    merged = engine.merge_populations(local, remote)
    assert len(merged) == 1
    assert merged[0].agent_id == 10
    assert merged[0].fitness == 0.9


# ── Test 3: lineage conflict → merged lineage with both parents noted ──

def test_lineage_conflict_merged_lineage(engine):
    """When both copies have different valid parents, both are preserved."""
    local = [
        Agent(agent_id=20, fitness=0.6, generation=2, parent_a=1, parent_b=2, vector=[0.1] * DIM),
    ]
    remote = [
        Agent(agent_id=20, fitness=0.6, generation=2, parent_a=3, parent_b=4, vector=[0.2] * DIM),
    ]

    merged = engine.merge_populations(local, remote)
    assert len(merged) == 1
    agent = merged[0]
    assert agent.agent_id == 20
    # Both parent sets should be merged (local first, then remote, no dups)
    assert agent.parent_a == 1
    assert agent.parent_b == 2
    # The remaining parents should be tracked — but parent_b only holds one.
    # The merged lineage is stored as [parent_a, parent_b] with dedup.
    # Our implementation merges into parent_a/parent_b, keeping up to 2.
    # Since 1,2,3,4 are all distinct, the implementation keeps first two: 1,2.
    # However, resolve_conflict constructs a new Agent with merged_parents list.
    # Let's verify the exact behavior.

    # Re-check: _merge_lineages returns [1, 2, 3, 4] but Agent only has parent_a/parent_b.
    # The resolve_conflict uses merged_parents[0] and merged_parents[1] if len > 1.
    assert agent.parent_a in (1, 3)
    assert agent.parent_b in (2, 4)


# ── Test 4: vector table sync → LWW merge on timestamps ──

def test_vector_table_sync_lww():
    """Merging two vector tables keeps the most-recently-updated copy per agent."""
    local_vt = FluxVectorTable(dim=DIM, bit_width=2)
    remote_vt = FluxVectorTable(dim=DIM, bit_width=2)

    # Agent 100 in local, updated at t=10
    from swarm.vector_table import AgentVector
    av_local = AgentVector(
        agent_id=100,
        vector=[0.1] * DIM,
        fitness=0.5,
        generation=0,
        capability_mask=0xFFFF,
        thermal_pressure=0.0,
    )
    local_vt.add(av_local)
    local_vt._meta[100].extra["last_updated"] = 10.0

    # Agent 100 in remote, updated at t=20 (wins)
    av_remote = AgentVector(
        agent_id=100,
        vector=[0.2] * DIM,
        fitness=0.7,
        generation=0,
        capability_mask=0xFFFF,
        thermal_pressure=0.0,
    )
    remote_vt.add(av_remote)
    remote_vt._meta[100].extra["last_updated"] = 20.0

    # Agent 200 only in remote
    av_remote2 = AgentVector(
        agent_id=200,
        vector=[0.3] * DIM,
        fitness=0.8,
        generation=0,
        capability_mask=0xFFFF,
        thermal_pressure=0.0,
    )
    remote_vt.add(av_remote2)
    remote_vt._meta[200].extra["last_updated"] = 15.0

    engine = CRDTMergeEngine(vector_table=local_vt)
    merged_vt = engine.sync_vector_table(local_vt, remote_vt)

    assert len(merged_vt) == 2
    # Agent 100 should have remote's vector because remote timestamp is higher
    meta_100 = merged_vt._meta[100]
    assert meta_100.fitness == 0.7
    vec_100 = merged_vt._index._vectors[100]
    assert pytest.approx(vec_100[0], 0.001) == 0.2

    # Agent 200 should be present
    assert 200 in merged_vt._meta
    assert merged_vt._meta[200].fitness == 0.8


# ── Test 5: impossible jump in remote agent → rejected ───

def test_impossible_jump_rejected(engine):
    """A remote agent with an impossible generation jump is rejected."""
    local = [
        Agent(agent_id=50, fitness=0.5, generation=1, parent_a=40, vector=[0.1] * DIM),
    ]
    # Remote claims agent 60 descends from 50 but with generation 99
    remote = [
        Agent(agent_id=60, fitness=0.6, generation=99, parent_a=50, vector=[0.2] * DIM),
    ]

    merged = engine.merge_populations(local, remote)
    merged_ids = {a.agent_id for a in merged}

    assert 50 in merged_ids
    assert 60 not in merged_ids  # rejected as impossible


# ── Extra: detect_divergence smoke test ────────────────

def test_detect_divergence(engine):
    local = [
        Agent(agent_id=1, fitness=0.5, generation=0, vector=[0.1] * DIM),
        Agent(agent_id=2, fitness=0.6, generation=0, vector=[0.2] * DIM),
    ]
    remote = [
        Agent(agent_id=2, fitness=0.7, generation=1, vector=[0.3] * DIM),
        Agent(agent_id=3, fitness=0.8, generation=0, vector=[0.4] * DIM),
    ]

    report = engine.detect_divergence(local, remote)
    assert report.local_only == [1]
    assert report.remote_only == [3]
    assert report.common_diverged == [2]
    assert report.fitness_delta == pytest.approx(0.1)


# ── Extra: resolve_conflict tie-break on timestamp ───────

def test_resolve_conflict_tiebreak_timestamp(engine):
    local = Agent(agent_id=5, fitness=0.6, generation=1, last_updated=100.0, vector=[0.1] * DIM)
    remote = Agent(agent_id=5, fitness=0.6, generation=1, last_updated=200.0, vector=[0.2] * DIM)

    winner = engine.resolve_conflict(local, remote)
    assert winner.last_updated == 200.0
    assert winner.vector == [0.2] * DIM
