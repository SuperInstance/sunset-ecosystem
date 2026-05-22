"""Tests for CRDTMergeEngine — divergent population merge after network partition.

Mocks turbovec so tests run without the native extension.
"""

from __future__ import annotations

import sys
import types

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

# Now safe to import swarm modules
from swarm.crdt_merge import (
    Agent,
    CRDTMergeEngine,
    DivergenceReport,
    LineageSanityError,
)
from swarm.vector_table import AgentVector, FluxVectorTable


# ── helpers ─────────────────────────────────────────────────

def _make_table(dim: int = 64) -> FluxVectorTable:
    return FluxVectorTable(dim=dim, bit_width=4)


def _agent(
    agent_id: int,
    fitness: float = 0.5,
    generation: int = 0,
    parent_a: int | None = None,
    parent_b: int | None = None,
    vector: list[float] | None = None,
    last_updated: float | None = None,
) -> Agent:
    import time
    vec = vector if vector is not None else [0.1] * 64
    ts = last_updated if last_updated is not None else time.time()
    return Agent(
        agent_id=agent_id,
        fitness=fitness,
        generation=generation,
        parent_a=parent_a,
        parent_b=parent_b,
        vector=vec,
        last_updated=ts,
        capability_mask=0xFFFF,
    )


# ── tests ───────────────────────────────────────────────────

class TestSimpleMerge:
    """Union of two disjoint populations."""

    def test_disjoint_populations_union(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, fitness=0.5), _agent(2, fitness=0.6)]
        remote = [_agent(3, fitness=0.7), _agent(4, fitness=0.8)]

        merged = engine.merge_populations(local, remote)
        merged_ids = {a.agent_id for a in merged}

        assert merged_ids == {1, 2, 3, 4}
        assert len(merged) == 4

    def test_empty_remote(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1), _agent(2)]
        merged = engine.merge_populations(local, [])

        assert len(merged) == 2
        assert {a.agent_id for a in merged} == {1, 2}

    def test_empty_local(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        remote = [_agent(1), _agent(2)]
        merged = engine.merge_populations([], remote)

        assert len(merged) == 2
        assert {a.agent_id for a in merged} == {1, 2}


class TestHigherFitnessWins:
    """When agent exists on both sides, keep higher-fitness copy."""

    def test_local_higher_fitness_kept(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, fitness=0.9)]
        remote = [_agent(1, fitness=0.3)]

        merged = engine.merge_populations(local, remote)
        winner = next(a for a in merged if a.agent_id == 1)

        assert winner.fitness == 0.9

    def test_remote_higher_fitness_wins(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, fitness=0.3)]
        remote = [_agent(1, fitness=0.9)]

        merged = engine.merge_populations(local, remote)
        winner = next(a for a in merged if a.agent_id == 1)

        assert winner.fitness == 0.9

    def test_tie_breaks_on_last_updated(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, fitness=0.5, last_updated=100.0)]
        remote = [_agent(1, fitness=0.5, last_updated=200.0)]

        merged = engine.merge_populations(local, remote)
        winner = next(a for a in merged if a.agent_id == 1)

        assert winner.last_updated == 200.0

    def test_equal_fitness_local_wins_when_same_timestamp(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, fitness=0.5, last_updated=100.0)]
        remote = [_agent(1, fitness=0.5, last_updated=100.0)]

        merged = engine.merge_populations(local, remote)
        winner = next(a for a in merged if a.agent_id == 1)

        # When fitness and timestamp are equal, local wins (LWW tie-break)
        # Implementation picks local when local_agent.last_updated >= remote_agent.last_updated
        assert winner.last_updated == 100.0


class TestLineageConflictMerge:
    """When both copies have valid but different lineages, merge them."""

    def test_different_parents_merged(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        # Child 10: local says parents (1, 2); remote says parents (3, 4)
        # All parents exist in the combined population so both lineages are "valid"
        local = [
            _agent(1, generation=0),
            _agent(2, generation=0),
            _agent(10, generation=1, parent_a=1, parent_b=2),
        ]
        remote = [
            _agent(3, generation=0),
            _agent(4, generation=0),
            _agent(10, generation=1, parent_a=3, parent_b=4),
        ]

        merged = engine.merge_populations(local, remote)
        child = next(a for a in merged if a.agent_id == 10)

        # Merged lineage should include all unique parents
        assert set(child.all_parents) == {1, 2, 3, 4}
        assert child.parent_a in (1, 2, 3, 4)
        assert child.parent_b in (1, 2, 3, 4)
        assert child.parent_a != child.parent_b

    def test_same_parents_no_conflict(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, generation=1, parent_a=10, parent_b=11)]
        remote = [_agent(1, generation=1, parent_a=10, parent_b=11)]

        merged = engine.merge_populations(local, remote)
        winner = next(a for a in merged if a.agent_id == 1)

        assert winner.parent_a == 10
        assert winner.parent_b == 11

    def test_one_side_no_parents(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, generation=0, parent_a=None, parent_b=None, last_updated=200.0)]
        remote = [_agent(1, generation=1, parent_a=10, parent_b=11, last_updated=100.0)]

        merged = engine.merge_populations(local, remote)
        winner = next(a for a in merged if a.agent_id == 1)

        # Local wins (higher timestamp) and has no parents → no lineage conflict
        assert winner.parent_a is None
        assert winner.parent_b is None
        assert winner.all_parents == []


class TestVectorTableLWWSync:
    """sync_vector_table merges two FluxVectorTables with last-write-wins."""

    def test_local_wins_when_newer(self):
        local_vt = _make_table()
        remote_vt = _make_table()

        vec = [0.1] * 64
        local_vt.add(AgentVector(agent_id=1, vector=vec, fitness=0.9))
        local_vt._meta[1].extra["last_updated"] = 200.0

        remote_vt.add(AgentVector(agent_id=1, vector=vec, fitness=0.3))
        remote_vt._meta[1].extra["last_updated"] = 100.0

        engine = CRDTMergeEngine(local_vt)
        merged_vt = engine.sync_vector_table(local_vt, remote_vt)

        assert merged_vt._meta[1].fitness == 0.9

    def test_remote_wins_when_newer(self):
        local_vt = _make_table()
        remote_vt = _make_table()

        vec = [0.1] * 64
        local_vt.add(AgentVector(agent_id=1, vector=vec, fitness=0.3))
        local_vt._meta[1].extra["last_updated"] = 100.0

        remote_vt.add(AgentVector(agent_id=1, vector=vec, fitness=0.9))
        remote_vt._meta[1].extra["last_updated"] = 200.0

        engine = CRDTMergeEngine(local_vt)
        merged_vt = engine.sync_vector_table(local_vt, remote_vt)

        assert merged_vt._meta[1].fitness == 0.9

    def test_union_of_distinct_ids(self):
        local_vt = _make_table()
        remote_vt = _make_table()

        local_vt.add(AgentVector(agent_id=1, vector=[0.1] * 64, fitness=0.5))
        remote_vt.add(AgentVector(agent_id=2, vector=[0.2] * 64, fitness=0.6))

        engine = CRDTMergeEngine(local_vt)
        merged_vt = engine.sync_vector_table(local_vt, remote_vt)

        assert 1 in merged_vt._meta
        assert 2 in merged_vt._meta
        assert len(merged_vt._meta) == 2

    def test_preserves_timestamps(self):
        local_vt = _make_table()
        remote_vt = _make_table()

        local_vt.add(AgentVector(agent_id=1, vector=[0.1] * 64, fitness=0.5))
        local_vt._meta[1].extra["last_updated"] = 42.0

        engine = CRDTMergeEngine(local_vt)
        merged_vt = engine.sync_vector_table(local_vt, remote_vt)

        assert merged_vt._meta[1].extra.get("last_updated") == 42.0


class TestImpossibleJumpRejection:
    """Remote agents with impossible lineage jumps are rejected."""

    def test_generation_too_far_from_parents_rejected(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [
            _agent(1, generation=0),
            _agent(2, generation=0),
        ]
        # Child claims parents (1,2) with gen=0, but child gen=10 (impossible jump)
        remote = [_agent(3, generation=10, parent_a=1, parent_b=2)]

        merged = engine.merge_populations(local, remote)
        merged_ids = {a.agent_id for a in merged}

        assert 3 not in merged_ids
        assert {1, 2} == merged_ids

    def test_orphan_rejected(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, generation=0)]
        # Claims parent 99 which doesn't exist in either population
        remote = [_agent(2, generation=1, parent_a=99)]

        merged = engine.merge_populations(local, remote)
        merged_ids = {a.agent_id for a in merged}

        assert 2 not in merged_ids
        assert 1 in merged_ids

    def test_seed_with_nonzero_generation_rejected(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        # Root agent (no parents) must have generation == 0
        remote = [_agent(7, generation=5, parent_a=None, parent_b=None)]

        merged = engine.merge_populations([], remote)
        merged_ids = {a.agent_id for a in merged}

        assert 7 not in merged_ids

    def test_valid_agent_accepted(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [
            _agent(1, generation=0),
            _agent(2, generation=0),
        ]
        remote = [_agent(3, generation=1, parent_a=1, parent_b=2)]

        merged = engine.merge_populations(local, remote)
        merged_ids = {a.agent_id for a in merged}

        assert 3 in merged_ids
        assert {1, 2, 3} == merged_ids


class TestDetectDivergence:
    """detect_divergence produces accurate DivergenceReport."""

    def test_fully_disjoint(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1), _agent(2)]
        remote = [_agent(3), _agent(4)]

        report = engine.detect_divergence(local, remote)

        assert report.local_only == [1, 2]
        assert report.remote_only == [3, 4]
        assert report.common_diverged == []
        assert report.lineage_conflicts == []
        assert report.fitness_delta == 0.0

    def test_common_diverged(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, fitness=0.5, generation=1)]
        remote = [_agent(1, fitness=0.7, generation=2)]

        report = engine.detect_divergence(local, remote)

        assert report.local_only == []
        assert report.remote_only == []
        assert report.common_diverged == [1]
        assert report.fitness_delta == pytest.approx(0.2)

    def test_lineage_conflict_detected(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [_agent(1, parent_a=10, parent_b=11)]
        remote = [_agent(1, parent_a=12, parent_b=13)]

        report = engine.detect_divergence(local, remote)

        assert report.lineage_conflicts == [1]

    def test_mixed_scenario(self):
        vt = _make_table()
        engine = CRDTMergeEngine(vt)

        local = [
            _agent(1, fitness=0.5),
            _agent(2, fitness=0.6, parent_a=10, parent_b=11),
        ]
        remote = [
            _agent(2, fitness=0.6, parent_a=12, parent_b=13),
            _agent(3, fitness=0.7),
        ]

        report = engine.detect_divergence(local, remote)

        assert report.local_only == [1]
        assert report.remote_only == [3]
        # Agent 2 exists on both sides with different parents → diverged
        assert report.common_diverged == [2]
        assert report.lineage_conflicts == [2]
        assert report.fitness_delta == 0.0
