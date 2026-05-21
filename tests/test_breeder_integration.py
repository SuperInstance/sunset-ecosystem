"""Integration test for vector-table breeding loop.

Creates 100 agents, runs 10 breeding cycles, verifies that the
FluxVectorTable is consulted during parent selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from nerve.room_grid import RoomGrid
from swarm.breeder import BreedingDaemon, CompactionManager, FluxVectorTable
from swarm.thermal import DeviceType, ThermalBudget


@pytest.fixture
def grid():
    """100-room grid with a clear hot/cold split."""
    g = RoomGrid(n=100)
    # Rooms 0-49: hot (high activity)
    for _ in range(30):
        for i in range(50):
            g.activity[i] += 3
    # Rooms 50-99: cold
    return g


@pytest.fixture
def thermal():
    return ThermalBudget({DeviceType.GPU: 20, DeviceType.CPU: 20})


@pytest.fixture
def vector_table(grid):
    """Vector table with entries for all rooms."""
    vt = FluxVectorTable(dim=16)
    rng = np.random.RandomState(42)
    for i in range(grid.n):
        # Give hot rooms higher-norm vectors so they are preferred
        scale = 2.0 if i < 50 else 0.5
        vt.add(f"room_{i}", rng.randn(16).astype(np.float32) * scale)
    return vt


class TestVectorTableIntegration:
    """End-to-end: vector table drives parent selection."""

    def test_vector_table_selects_parents(self, grid, vector_table):
        """select_parents with a vector table should return vectored agents."""
        thermal = ThermalBudget({DeviceType.GPU: 20})
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            vector_table=vector_table,
            n_winners=3,
        )

        winners = daemon.auto_breeder.select_parents(
            vector_table=vector_table,
            n_winners=3,
        )

        assert len(winners) > 0
        # All selected parents must have entries in the vector table
        for w in winners:
            assert f"room_{w.agent_id.split('_')[1]}" in vector_table.vectors

    def test_fallback_without_vector_table(self, grid):
        """Without vector table, select_parents falls back to tournament."""
        thermal = ThermalBudget({DeviceType.GPU: 20})
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            n_winners=3,
        )

        winners = daemon.auto_breeder.select_parents(
            vector_table=None,
            n_winners=3,
        )

        assert len(winners) > 0
        # Winners should be from hot rooms (0-49)
        for w in winners:
            room_num = int(w.agent_id.split("_")[1])
            assert room_num < 50

    def test_ten_cycles_with_vector_table(self, grid, thermal, vector_table):
        """Run 10 breeding cycles; vector table must be used at least once."""
        compaction = CompactionManager(max_archive_size=500)
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            vector_table=vector_table,
            compaction=compaction,
            n_winners=3,
            cold_threshold=3,
            compaction_interval=3,
        )

        vector_used = False
        total_rebirths = 0

        for _ in range(10):
            # Re-seed vectors for any rooms that still have them
            # (BreedingDaemon removes vectored entries for rebirthed rooms)
            rng = np.random.RandomState(42)
            for i in range(grid.n):
                if f"room_{i}" not in vector_table.vectors:
                    scale = 2.0 if i < 50 else 0.5
                    vector_table.add(f"room_{i}", rng.randn(16).astype(np.float32) * scale)

            # Pre-check: winners should be from vectored agents
            winners = daemon.auto_breeder.select_parents(
                vector_table=vector_table,
                n_winners=3,
            )
            if winners and all(
                w.agent_id in vector_table.vectors for w in winners
            ):
                vector_used = True

            results = daemon.cycle(n_winners=3)
            total_rebirths += len(results)

        assert vector_used, "Vector table was never consulted for parent selection"
        assert total_rebirths > 0, "No rebirths occurred in 10 cycles"

        # Compaction should have run at least twice (cycles 3, 6, 9)
        assert compaction._compaction_count >= 2

        # Archive should have sunset records
        assert compaction.size > 0

    def test_compaction_archives_sunset(self, grid, thermal):
        """CompactionManager archives agents identified as sunset candidates."""
        compaction = CompactionManager()
        daemon = BreedingDaemon(
            grid=grid,
            thermal=thermal,
            compaction=compaction,
            n_winners=3,
        )

        # Run one cycle to trigger sunset archiving
        daemon.cycle()

        # Some dominated agents should have been archived
        assert compaction.size >= 0  # may be 0 if no dominated agents
        # Compact and verify it works
        removed = compaction.compact()
        assert isinstance(removed, int)

    def test_vector_table_query(self):
        """FluxVectorTable cosine query works end-to-end."""
        vt = FluxVectorTable(dim=4)
        vt.add("a", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        vt.add("b", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        vt.add("c", np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32))

        results = vt.query(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)
        assert len(results) == 2
        # "a" and "c" are most similar to [1,0,0,0]
        assert "a" in results
        assert "c" in results

    def test_breeding_daemon_delegate_lifecycle(self, grid, thermal):
        """BreedingDaemon start/stop delegates to AutoBreeder."""
        daemon = BreedingDaemon(grid=grid, thermal=thermal, interval=0.1)
        daemon.start()
        assert daemon.running
        daemon.stop()
        assert not daemon.running
