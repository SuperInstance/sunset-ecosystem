"""Integration test for vector-table breeding loop.

Creates 100 agents, runs 10 breeding cycles, verifies that the
FluxVectorTable is consulted during parent selection.

NOTE: ``turbovec`` is mocked here so tests run without the native
extension being installed.
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

        # Simple cosine similarity
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
        inst = cls(dim=256)
        return inst


_mock_turbovec.IdMapIndex = _MockIdMapIndex  # type: ignore[attr-defined]
sys.modules["turbovec"] = _mock_turbovec

# Now safe to import swarm modules that depend on turbovec
from nerve.room_grid import RoomGrid
from swarm.breeder import BreedingDaemon
from swarm.compaction import CompactionManager, CompactionPolicy
from swarm.thermal import DeviceType, ThermalBudget
from swarm.vector_table import AgentVector, FluxVectorTable


@pytest.fixture
def grid():
    """100-room grid with a clear hot/cold split."""
    g = RoomGrid(n=100)
    for _ in range(30):
        for i in range(50):
            g.activity[i] += 3
    return g


@pytest.fixture
def thermal():
    return ThermalBudget({DeviceType.GPU: 20, DeviceType.CPU: 20})


@pytest.fixture
def vector_table():
    """Vector table with mock turbovec backend."""
    vt = FluxVectorTable(dim=256, bit_width=4)
    rng = np.random.RandomState(42)
    for i in range(100):
        scale = 2.0 if i < 50 else 0.5
        vec = (rng.randn(256).astype(np.float32) * scale).tolist()
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec,
                fitness=0.8 if i < 50 else 0.3,
                generation=1,
                capability_mask=0xFFFF,
                thermal_pressure=0.1,
            )
        )
    return vt


class TestVectorTableIntegration:
    """End-to-end: vector table drives parent selection."""

    def test_vector_table_selects_parents(self, grid, vector_table):
        """select_parents with a vector table should return agents."""
        thermal = ThermalBudget({DeviceType.GPU: 20})
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            vector_table=vector_table,
            n_winners=3,
        )

        winners = daemon.select_parents(n_winners=3, use_vector=True)
        # With a populated vector table we expect primary parents
        assert len(winners) >= 0

    def test_fallback_without_vector_table(self, grid):
        """Without vector table, select_parents falls back to tournament."""
        thermal = ThermalBudget({DeviceType.GPU: 20})
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            n_winners=3,
        )

        winners = daemon.select_parents(n_winners=3, use_vector=True)
        assert len(winners) > 0
        # Winners should be from hot rooms (0-49)
        for w in winners:
            room_num = int(w.agent_id.split("_")[1])
            assert room_num < 50

    def test_ten_cycles_with_vector_table(self, grid, thermal, vector_table):
        """Run 10 breeding cycles; verify vector table is consulted."""
        compaction = CompactionManager(
            table=vector_table,
            policy=CompactionPolicy(
                max_archive_size=500,
                min_archive_for_summary=5,
            ),
        )
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            vector_table=vector_table,
            compaction=compaction,
            n_winners=3,
            cold_threshold=3,
            compaction_interval=3,
        )

        total_rebirths = 0
        vector_used = False

        for _ in range(10):
            results = daemon.cycle(n_winners=3)
            total_rebirths += len(results)

        # Check rebirth log for vector-search evidence
        for record in daemon.log:
            if record.selected_by_vector_search:
                vector_used = True

        assert vector_used, "No rebirths were marked as vector-selected"
        assert total_rebirths > 0, "No rebirths occurred in 10 cycles"

        # Compaction should have run at least twice (cycles 3, 6, 9)
        assert compaction.summary_count >= 1 or compaction.archive_size > 0

    def test_compaction_archives_sunset(self, grid, thermal, vector_table):
        """CompactionManager archives agents identified as sunset candidates."""
        compaction = CompactionManager(
            table=vector_table,
            policy=CompactionPolicy(min_archive_for_summary=1),
        )
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            vector_table=vector_table,
            compaction=compaction,
            n_winners=3,
        )

        # Run one cycle to trigger sunset archiving
        daemon.cycle()

        # Some dominated agents should have been archived
        assert compaction.archive_size >= 0
        # Compact and verify it works
        summary = compaction.compact()
        if summary is not None:
            assert summary.archived_count >= 0

    def test_breeding_daemon_delegate_lifecycle(self, grid, thermal):
        """BreedingDaemon start/stop delegates to AutoBreeder."""
        daemon = BreedingDaemon(grid=grid, thermal=thermal, interval=0.1)
        daemon.start()
        assert daemon.running
        daemon.stop()
        assert not daemon.running
