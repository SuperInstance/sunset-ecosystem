"""Tests for CompactionManager — archiving, compaction triggers, and summary generation.

Uses a mock FluxVectorTable to avoid turbovec dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.compaction import (
    ArchiveSummary,
    CompactionManager,
    CompactionPolicy,
)


class FakeFluxTable:
    """Minimal mock of FluxVectorTable for compaction tests."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._meta: dict[int, object] = {}
        self._vectors: dict[int, np.ndarray] = {}
        self._index: dict[int, None] = {}  # placeholder for len(self.table._index)

    def _get_vector(self, agent_id: int) -> np.ndarray | None:
        return self._vectors.get(agent_id)

    def add(self, av) -> None:
        self._vectors[av.agent_id] = np.array(av.vector, dtype=np.float32)
        self._meta[av.agent_id] = av
        self._index[av.agent_id] = None

    def remove(self, agent_id: int) -> bool:
        self._vectors.pop(agent_id, None)
        self._meta.pop(agent_id, None)
        self._index.pop(agent_id, None)
        return True

    def search(self, query, k=10, **kwargs):
        return []

    def __len__(self) -> int:
        return len(self._meta)

    @property
    def _index(self) -> dict:
        return self.__dict__.setdefault("__index", {})

    @_index.setter
    def _index(self, value: dict) -> None:
        self.__dict__["__index"] = value


class FakeAgentVector:
    def __init__(
        self, agent_id, vector, fitness=0.0, generation=0, capability_mask=0xFFFF
    ):
        self.agent_id = agent_id
        self.vector = vector
        self.fitness = fitness
        self.generation = generation
        self.capability_mask = capability_mask


# ---------------------------------------------------------------------------
# CompactionPolicy
# ---------------------------------------------------------------------------


class TestCompactionPolicy:
    def test_defaults(self):
        p = CompactionPolicy()
        assert p.max_archive_size == 10_000
        assert p.max_generations_without_compact == 100
        assert p.preserve_recent_generations == 10
        assert p.min_archive_for_summary == 100

    def test_custom_values(self):
        p = CompactionPolicy(max_archive_size=5, preserve_recent_generations=2)
        assert p.max_archive_size == 5
        assert p.preserve_recent_generations == 2


# ---------------------------------------------------------------------------
# CompactionManager — Lifecycle
# ---------------------------------------------------------------------------


class TestCompactionLifecycle:
    def _make_table(self, dim=4):
        ft = FakeFluxTable(dim=dim)
        return ft

    def _add_agent(self, ft, aid, generation, vector=None):
        vec = vector or [0.1] * ft.dim
        ft.add(FakeAgentVector(aid, vec, fitness=0.5, generation=generation))

    def test_archive_sunset(self):
        ft = self._make_table()
        cm = CompactionManager(table=ft)
        cm.archive_sunset(1)
        assert cm.archive_size == 1
        assert 1 in cm._archived

    def test_record_birth(self):
        ft = self._make_table()
        cm = CompactionManager(table=ft)
        cm.record_birth(1, generation=5)
        assert cm._generations[1] == 5

    def test_should_compact_not_enough(self):
        ft = self._make_table()
        cm = CompactionManager(table=ft)
        assert not cm.should_compact()

    def test_should_compact_archive_size(self):
        ft = self._make_table()
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(max_archive_size=5, min_archive_for_summary=1),
        )
        for i in range(6):
            cm.archive_sunset(i)
        assert cm.should_compact()

    def test_should_compact_generation_gap(self):
        ft = self._make_table()
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(
                min_archive_for_summary=1, max_generations_without_compact=3
            ),
        )
        for i in range(3):
            cm.record_birth(i, generation=i)
            cm.archive_sunset(i)
        assert not cm.should_compact()
        cm.record_birth(3, generation=4)
        cm.archive_sunset(3)
        assert cm.should_compact()

    def test_compact_skips_when_not_ready(self):
        ft = self._make_table()
        cm = CompactionManager(table=ft)
        assert cm.compact() is None

    def test_compact_produces_summary(self):
        ft = self._make_table(dim=4)
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(
                max_archive_size=5,
                min_archive_for_summary=1,
                preserve_recent_generations=0,
            ),
        )
        for i in range(5):
            self._add_agent(ft, i, generation=i, vector=[float(i), 0.0, 0.0, 0.0])
            cm.record_birth(i, generation=i)
            cm.archive_sunset(i)
        summary = cm.compact()
        assert summary is not None
        assert isinstance(summary, ArchiveSummary)
        assert summary.archived_count == 5
        assert summary.generation_range == (0, 4)
        assert len(summary.centroid) == 4
        assert len(summary.variance) == 4

    def test_compact_preserves_recent_generations(self):
        ft = self._make_table(dim=4)
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(
                max_archive_size=10,
                min_archive_for_summary=1,
                preserve_recent_generations=2,
                max_generations_without_compact=3,
            ),
        )
        for i in range(5):
            self._add_agent(ft, i, generation=i, vector=[float(i), 0.0, 0.0, 0.0])
            cm.record_birth(i, generation=i)
            cm.archive_sunset(i)
        summary = cm.compact()
        # gens 3 and 4 should be preserved, so 0-2 compacted
        assert summary is not None
        assert summary.archived_count == 3
        assert summary.generation_range == (0, 2)

    def test_compact_removes_from_table(self):
        ft = self._make_table(dim=4)
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(
                max_archive_size=5,
                min_archive_for_summary=1,
                preserve_recent_generations=0,
            ),
        )
        for i in range(5):
            self._add_agent(ft, i, generation=i)
            cm.record_birth(i, generation=i)
            cm.archive_sunset(i)
        cm.compact()
        assert len(ft._vectors) == 0

    def test_compact_empty_vectors(self):
        ft = self._make_table(dim=4)
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(
                max_archive_size=3,
                min_archive_for_summary=1,
                preserve_recent_generations=0,
            ),
        )
        for i in range(3):
            cm.record_birth(i, generation=i)
            cm.archive_sunset(i)
        # agents not in table → no vectors
        assert cm.compact() is None

    def test_total_archived(self):
        ft = self._make_table(dim=4)
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(
                max_archive_size=5,
                min_archive_for_summary=1,
                preserve_recent_generations=0,
            ),
        )
        for i in range(5):
            self._add_agent(ft, i, generation=i)
            cm.record_birth(i, generation=i)
            cm.archive_sunset(i)
        cm.compact()
        assert cm.total_archived == 5  # all compacted, none left in archive set
        assert cm.archive_size == 0

    def test_summary_count(self):
        ft = self._make_table(dim=4)
        cm = CompactionManager(
            table=ft,
            policy=CompactionPolicy(
                max_archive_size=5,
                min_archive_for_summary=1,
                preserve_recent_generations=0,
            ),
        )
        for i in range(5):
            self._add_agent(ft, i, generation=i)
            cm.record_birth(i, generation=i)
            cm.archive_sunset(i)
        cm.compact()
        assert cm.summary_count == 1

    def test_repr(self):
        ft = self._make_table()
        cm = CompactionManager(table=ft)
        r = repr(cm)
        assert "CompactionManager" in r
        assert "living=" in r

    def test_union_capabilities(self):
        assert CompactionManager._union_capabilities([0b0001, 0b0010, 0b0100]) == 0b0111

    def test_search_with_summaries_empty(self):
        ft = self._make_table(dim=4)
        cm = CompactionManager(table=ft)
        results = cm.search_with_summaries([0.0, 0.0, 0.0, 0.0], k=5)
        assert results == []

    def test_default_generation_fn(self):
        assert CompactionManager._default_generation_fn(0x0000_0005_0000_0000) == 5
        assert CompactionManager._default_generation_fn(0x0000_0000_0000_0001) == 0
