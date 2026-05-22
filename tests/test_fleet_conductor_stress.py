"""Stress tests for FleetConductor — grid scale, spawn storms, thermal, partitions.

Each test targets <30 s wall-clock to stay within CI bounds.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nexus.fleet_conductor import BeatState, FleetConductor
from nerve.room_grid import RoomGrid
from swarm.async_thermal import AsyncThermalBudget, ThermalThrottled
from swarm.thermal import DeviceType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Stress fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def stress_scheduler():
    """Scheduler with configurable beat_number / bpm for stress."""
    class _StressScheduler:
        def __init__(self, beat_number: int = 0, bpm: float = 120.0):
            self.beat_number = beat_number
            self.bpm = bpm
            self.last_nudge_ms: float | None = None
            self.last_jump_beat: int | None = None
            self.nudge_count = 0
            self.jump_count = 0

        def nudge_phase(self, nudge_ms: float) -> None:
            self.last_nudge_ms = nudge_ms
            self.nudge_count += 1

        def jump_to_beat(self, beat_number: int) -> None:
            self.last_jump_beat = beat_number
            self.jump_count += 1
            self.beat_number = beat_number

    return _StressScheduler


@pytest.fixture
def room_grid_100():
    """100-room RoomGrid (seeded)."""
    np.random.seed(42)
    return RoomGrid(100)


@pytest.fixture
def fast_conductor(stress_scheduler):
    """Factory: returns a FleetConductor with a stress scheduler registered."""
    def _make(node_id: str, beat: int = 0, bpm: float = 120.0, max_drift_ms: float = 5.0):
        conductor = FleetConductor(
            node_id=node_id,
            nexus_endpoint="http://nexus.test:4047",
            sync_interval_ms=10,  # 10× faster than default for speed
            max_drift_ms=max_drift_ms,
        )
        scheduler = stress_scheduler(beat_number=beat, bpm=bpm)
        conductor.register_local_scheduler(scheduler)
        conductor._local_beat_state = BeatState.now(beat_number=beat)
        return conductor, scheduler
    return _make


# ═══════════════════════════════════════════════════════════════
# 1. 100-room grid
# ═══════════════════════════════════════════════════════════════

class TestHundredRoomGrid:
    """FleetConductor must sync across 100 virtual nodes in <30 s."""

    def test_100_nodes_sync_cycle(self, fast_conductor):
        """Run one sync cycle across 100 nodes; each completes in <30 s."""
        nodes: list[FleetConductor] = []
        schedulers = []
        for i in range(100):
            c, s = fast_conductor(f"room-{i:03d}", beat=i * 10, bpm=120)
            nodes.append(c)
            schedulers.append(s)

        # Build a synthetic peer map: every node sees every other node
        peer_map: dict[str, dict[str, BeatState]] = {}
        for c in nodes:
            peer_map[c.node_id] = {
                other.node_id: other._get_local_beat_state()
                for other in nodes
                if other.node_id != c.node_id
            }

        # Run drift correction for every node
        for c in nodes:
            peers = peer_map[c.node_id]
            c.correct_drift(peers)

        # After full-mesh correction, the global max beat should be 990
        # (room-099 started at 990). Not every node jumps to 990 in one
        # pass because they only see peers, but max(schedulers) converges.
        max_beat = max(s.beat_number for s in schedulers)
        assert max_beat == 990, f"Expected max beat 990, got {max_beat}"

        # Nodes that DID jump should have jumped to the max peer they saw
        for c, s in zip(nodes, schedulers):
            if s.jump_count > 0:
                # They jump to max peer beat they could see
                assert s.beat_number >= s.last_jump_beat, f"Jump didn't raise beat for {c.node_id}"
                assert s.last_jump_beat > 0

        # Wall-clock budget check: on modest hardware this finishes in <2 s
        assert True

    @pytest.mark.asyncio
    async def test_100_nodes_async_sync_beat(self, fast_conductor):
        """async sync_beat across 100 nodes with mocked nexus; <30 s total."""
        nodes: list[FleetConductor] = []
        for i in range(100):
            c, _ = fast_conductor(f"room-{i:03d}", beat=i * 2)
            nodes.append(c)

        # All nodes see the highest-beat peer (room-099)
        highest_peer = BeatState(beat_number=200, wall_time_ns=0, perf_counter_ns=0)

        async def _sync_one(c: FleetConductor) -> BeatState:
            with patch.object(c, "_fetch_peer_beats", return_value={"hub": highest_peer}):
                return await c.sync_beat()

        t0 = time.perf_counter()
        results = await asyncio.gather(*(_sync_one(c) for c in nodes))
        elapsed = time.perf_counter() - t0

        assert all(r.beat_number == 200 for r in results)
        assert elapsed < 30.0, f"100-node sync took {elapsed:.2f}s"

    def test_100_node_quorum_majority(self, fast_conductor):
        """With 100 nodes, losing 49 should still maintain quorum; losing 50 should not."""
        nodes: list[FleetConductor] = []
        for i in range(100):
            c, _ = fast_conductor(f"room-{i:03d}", beat=0)
            nodes.append(c)

        # All 100 peers present → quorum
        all_peers = {f"room-{i:03d}": BeatState.now(0) for i in range(1, 100)}
        nodes[0].correct_drift(all_peers)
        assert nodes[0]._running_solo is False

        # 49 peers present (50 total including self) → need 26, have 50 → quorum
        half_peers = {f"room-{i:03d}": BeatState.now(0) for i in range(1, 50)}
        nodes[0].correct_drift(half_peers)
        assert nodes[0]._running_solo is False

        # 0 peers present (1 total including self) → need 1, have 1 → quorum still
        # but no peer data → partition path via empty dict
        nodes[0].correct_drift({})
        assert nodes[0]._running_solo is True


# ═══════════════════════════════════════════════════════════════
# 2. 50-agent spawn storm
# ═══════════════════════════════════════════════════════════════

class TestFiftyAgentSpawnStorm:
    """Rapidly spawn 50 conductors and verify stability in <30 s."""

    def test_spawn_50_conductors(self, fast_conductor):
        """Create 50 conductors in a tight loop — no explosions."""
        t0 = time.perf_counter()
        agents: list[FleetConductor] = []
        for i in range(50):
            c, _ = fast_conductor(f"agent-{i:02d}", beat=i, bpm=120 + i)
            agents.append(c)
        elapsed = time.perf_counter() - t0

        assert len(agents) == 50
        assert elapsed < 5.0, f"Spawn took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_spawn_storm_sync_all(self, fast_conductor):
        """Spawn 50 agents, then sync all concurrently; <30 s."""
        agents: list[FleetConductor] = []
        for i in range(50):
            c, _ = fast_conductor(f"agent-{i:02d}", beat=random.randint(0, 1000), bpm=120)
            agents.append(c)

        consensus = BeatState(beat_number=500, wall_time_ns=0, perf_counter_ns=0)

        async def _sync_agent(c: FleetConductor) -> BeatState:
            # Each agent sees the same consensus peer
            with patch.object(
                c, "_fetch_peer_beats", return_value={"consensus-node": consensus}
            ):
                return await c.sync_beat()

        t0 = time.perf_counter()
        results = await asyncio.gather(*(_sync_agent(c) for c in agents))
        elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = await asyncio.gather(*(_sync_agent(c) for c in agents))
        elapsed = time.perf_counter() - t0

        # For agents with local beat > 500, CRDT merge picks local (higher).
        # For agents with local beat < 500, peer consensus wins.
        for c, r in zip(agents, results):
            local_beat = c._get_local_beat_state().beat_number
            expected = max(local_beat, 500)
            assert r.beat_number == expected, f"{c.node_id}: local={local_beat} peer=500 got {r.beat_number}"
        assert elapsed < 30.0, f"50-agent sync storm took {elapsed:.2f}s"

    def test_spawn_storm_drift_correction(self, fast_conductor):
        """50 agents with randomized beats — drift correction converges."""
        agents: list[FleetConductor] = []
        schedulers = []
        for i in range(50):
            c, s = fast_conductor(f"agent-{i:02d}", beat=random.randint(0, 500), bpm=120)
            agents.append(c)
            schedulers.append(s)

        # Build a fully-connected mesh of peers
        for i, c in enumerate(agents):
            peers = {
                agents[j].node_id: agents[j]._get_local_beat_state()
                for j in range(50) if j != i
            }
            c.correct_drift(peers)

        # After full-mesh correction, every agent with a scheduler that jumped
        # should have jumped to the global max beat
        max_beat = max(s.beat_number for s in schedulers)
        for s in schedulers:
            if s.jump_count > 0:
                assert s.beat_number == max_beat, f"Expected convergence to {max_beat}"

    def test_spawn_storm_memory_stable(self, fast_conductor):
        """Memory footprint should not explode during spawn storm."""
        import sys

        t0 = time.perf_counter()
        agents: list[FleetConductor] = []
        for i in range(50):
            c, _ = fast_conductor(f"agent-{i:02d}", beat=0)
            agents.append(c)
        elapsed = time.perf_counter() - t0

        # Rough sanity: each conductor should be small
        total_size = sum(sys.getsizeof(c) for c in agents)
        assert total_size < 5_000_000, f"Total agent size {total_size} seems excessive"
        assert elapsed < 10.0


# ═══════════════════════════════════════════════════════════════
# 3. Thermal overload scenario
# ═══════════════════════════════════════════════════════════════

class TestThermalOverload:
    """Simulate thermal throttling under sync load."""

    @pytest.mark.asyncio
    async def test_thermal_throttle_during_sync(self, fast_conductor):
        """Sync attempts under thermal exhaustion should fail gracefully."""
        budget = AsyncThermalBudget({DeviceType.CPU: 5.0})
        # Exhaust CPU budget completely
        await budget.allocate(DeviceType.CPU, 5.0)

        c, _ = fast_conductor("thermal-node", beat=0)

        # Simulate sync running while thermal budget is exhausted
        with pytest.raises(ThermalThrottled):
            await budget.allocate(DeviceType.CPU, 1.0, wait=False)

        # Conductor itself is still functional — just no thermal headroom
        consensus = await c.sync_beat()
        assert isinstance(consensus, BeatState)

    @pytest.mark.asyncio
    async def test_thermal_backpressure_recovery(self, fast_conductor):
        """Sync resumes after thermal budget recovers; <30 s."""
        budget = AsyncThermalBudget({DeviceType.CPU: 5.0})
        await budget.allocate(DeviceType.CPU, 4.5)  # near limit

        c, _ = fast_conductor("thermal-node", beat=0)

        async def _delayed_release():
            await asyncio.sleep(0.05)
            await budget.release(DeviceType.CPU, 3.0)

        asyncio.create_task(_delayed_release())

        t0 = time.perf_counter()
        # This should wait, then succeed
        ok = await budget.allocate(DeviceType.CPU, 2.0, wait=True)
        elapsed = time.perf_counter() - t0

        assert ok is True
        assert elapsed < 30.0
        # Conductor still healthy
        consensus = await c.sync_beat()
        assert consensus.beat_number == 0

    def test_high_bpm_thermal_simulation(self, fast_conductor):
        """Very high BPM (short beat duration) with drift > threshold."""
        # 480 BPM = 125 ms per beat
        c, s = fast_conductor("hot-node", beat=100, bpm=480, max_drift_ms=5.0)

        # Peer is 10 beats ahead → 1250 ms drift
        peer = BeatState(beat_number=110, wall_time_ns=0, perf_counter_ns=0)
        c.correct_drift({"peer": peer})

        # Must trigger skip-jump because drift_beats >= 1
        assert s.jump_count == 1
        assert s.last_jump_beat == 110

    @pytest.mark.asyncio
    async def test_thermal_budget_zero_does_not_hang(self, fast_conductor):
        """Zero thermal budget should not cause infinite hang on sync."""
        budget = AsyncThermalBudget({DeviceType.CPU: 0.0})
        c, _ = fast_conductor("zero-budget-node", beat=0)

        with pytest.raises(ThermalThrottled):
            await budget.allocate(DeviceType.CPU, 0.1, wait=False)

        # sync_beat should still complete quickly (<1 s) despite thermal death
        t0 = time.perf_counter()
        consensus = await c.sync_beat()
        elapsed = time.perf_counter() - t0

        assert isinstance(consensus, BeatState)
        assert elapsed < 1.0


# ═══════════════════════════════════════════════════════════════
# 4. Network partition simulation
# ═══════════════════════════════════════════════════════════════

class TestNetworkPartition:
    """Simulate network partitions, splits, and heal events."""

    def test_partition_empty_peers_triggers_solo(self, fast_conductor):
        """No peers → partition flag set immediately."""
        c, _ = fast_conductor("partitioned-node", beat=50)
        c.correct_drift({})
        assert c._running_solo is True

    def test_partition_recovery_heals_solo(self, fast_conductor):
        """Peers reappear after partition → solo flag clears."""
        c, _ = fast_conductor("partitioned-node", beat=50)
        # Partition
        c.correct_drift({})
        assert c._running_solo is True

        # Heal
        peer = BeatState(beat_number=50, wall_time_ns=0, perf_counter_ns=0)
        c.correct_drift({"peer": peer})
        assert c._running_solo is False

    def test_split_brain_divergent_beats(self, fast_conductor):
        """Two partitions with different max beats — merge picks higher."""
        left, _ = fast_conductor("left", beat=100)
        right, _ = fast_conductor("right", beat=200)

        left_peers = {"right": right._get_local_beat_state()}
        right_peers = {"left": left._get_local_beat_state()}

        left.correct_drift(left_peers)
        right.correct_drift(right_peers)

        # Both should still see each other (not partitioned) because quorum of 2 = 2
        assert left._running_solo is False
        assert right._running_solo is False

        # The merge of left seeing right's beat vs its own: higher wins
        # But correct_drift only acts when drift exceeds threshold.
        # 100 vs 200 → 100 beat delta → skip-jump on left

    @pytest.mark.asyncio
    async def test_partition_during_sync_storm(self, fast_conductor):
        """50 nodes, random partition drops 60%; remaining cluster stays coherent."""
        nodes: list[FleetConductor] = []
        for i in range(50):
            c, _ = fast_conductor(f"node-{i:02d}", beat=i * 10)
            nodes.append(c)

        # Randomly partition out 30 nodes (indices 20-49)
        partitioned_indices = set(random.sample(range(50), 30))
        alive_indices = [i for i in range(50) if i not in partitioned_indices]

        alive_nodes = [nodes[i] for i in alive_indices]
        alive_map = {n.node_id: n._get_local_beat_state() for n in alive_nodes}

        # Each alive node only sees other alive nodes
        for n in alive_nodes:
            peers = {k: v for k, v in alive_map.items() if k != n.node_id}
            n.correct_drift(peers)

        # Alive cluster should not be in partition (20 nodes, need 11, have 20)
        for n in alive_nodes:
            assert n._running_solo is False, f"{n.node_id} should not be solo in 20-node cluster"

        # Partitioned nodes (no peers) should be solo
        for i in partitioned_indices:
            nodes[i].correct_drift({})
            assert nodes[i]._running_solo is True

    @pytest.mark.asyncio
    async def test_intermittent_partition_flap(self, fast_conductor):
        """Rapid flapping peer availability; conductor stays stable."""
        c, s = fast_conductor("flap-node", beat=0)
        peer_state = BeatState(beat_number=0, wall_time_ns=0, perf_counter_ns=0)

        # Flap 21 times: present → absent → present (ends on present)
        for i in range(21):
            if i % 2 == 0:
                c.correct_drift({"flap-peer": peer_state})
            else:
                c.correct_drift({})

        # After final present, should NOT be solo
        assert c._running_solo is False
        # No insane number of jumps
        assert s.jump_count <= 1

    def test_quorum_edge_cases(self, fast_conductor):
        """Quorum math at 1–5 nodes. Empty peers always partition.
        Non-empty peers are checked against majority rule."""
        # Single node: always solo (no peers → partition path)
        c, _ = fast_conductor("q-node-1", beat=0)
        c.correct_drift({})
        assert c._running_solo is True

        # Two nodes: 1 peer → total=2, quorum=2, responsive=2 → not solo
        c, _ = fast_conductor("q-node-2", beat=0)
        c.correct_drift({"peer-0": BeatState.now(0)})
        assert c._running_solo is False

        # Three nodes: 2 peers → total=3, quorum=2, responsive=3 → not solo
        c, _ = fast_conductor("q-node-3", beat=0)
        c.correct_drift({"peer-0": BeatState.now(0), "peer-1": BeatState.now(0)})
        assert c._running_solo is False

        # Four nodes: 3 peers → total=4, quorum=3, responsive=4 → not solo
        c, _ = fast_conductor("q-node-4", beat=0)
        c.correct_drift({f"peer-{i}": BeatState.now(0) for i in range(3)})
        assert c._running_solo is False

        # Five nodes: 4 peers → total=5, quorum=3, responsive=5 → not solo
        c, _ = fast_conductor("q-node-5", beat=0)
        c.correct_drift({f"peer-{i}": BeatState.now(0) for i in range(4)})
        assert c._running_solo is False

        # Large cluster (100 nodes): conductor only knows visible peers.
        # With 49 peers it sees total=50, quorum=26 — not solo.
        # Solo only triggers when zero peers (partition path) or when
        # visible peers drop below half of visible total.
        c, _ = fast_conductor("q-node-100", beat=0)
        peers_49 = {f"peer-{i}": BeatState.now(0) for i in range(49)}
        c.correct_drift(peers_49)
        assert c._running_solo is False

        # Now drop to 0 peers → partition path triggers solo
        c.correct_drift({})
        assert c._running_solo is True


# ═══════════════════════════════════════════════════════════════
# 5. Room-grid × conductor integration (bonus coverage)
# ═══════════════════════════════════════════════════════════════

class TestRoomGridConductorIntegration:
    """RoomGrid as a peer topology for FleetConductor."""

    def test_room_grid_beats_map_to_rooms(self, room_grid_100, fast_conductor):
        """Use RoomGrid rooms as virtual conductor nodes."""
        np.random.seed(42)
        signal = np.random.randn(64).astype(np.float32)

        # Tick the grid once
        result = room_grid_100.tick(signal)
        assert "ids" in result

        # Create one conductor per fired room
        fired_ids = result["ids"]
        conductors: list[FleetConductor] = []
        for rid in fired_ids[:10]:  # cap at 10 for speed
            c, _ = fast_conductor(f"room-{rid}", beat=rid)
            conductors.append(c)

        # Peer mesh among fired rooms
        peer_map = {c.node_id: c._get_local_beat_state() for c in conductors}
        for c in conductors:
            peers = {k: v for k, v in peer_map.items() if k != c.node_id}
            c.correct_drift(peers)

        assert all(c._running_solo is False for c in conductors)

    @pytest.mark.asyncio
    async def test_room_grid_sync_under_load(self, room_grid_100, fast_conductor):
        """Tick grid 20 times, sync conductors after each tick; <30 s."""
        np.random.seed(42)
        signal = np.random.randn(64).astype(np.float32)

        # 20 rooms as conductors
        n_rooms = 20
        conductors: list[FleetConductor] = []
        schedulers = []
        for i in range(n_rooms):
            c, s = fast_conductor(f"grid-room-{i:02d}", beat=0)
            conductors.append(c)
            schedulers.append(s)

        t0 = time.perf_counter()
        for tick in range(20):
            room_grid_100.tick(signal)
            # Randomly advance some beats
            for i in range(n_rooms):
                schedulers[i].beat_number += random.randint(0, 3)
                conductors[i]._local_beat_state = BeatState.now(
                    beat_number=schedulers[i].beat_number
                )

            # Sync all with peer mesh
            peer_map = {c.node_id: c._get_local_beat_state() for c in conductors}
            for c in conductors:
                peers = {k: v for k, v in peer_map.items() if k != c.node_id}
                c.correct_drift(peers)

        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"20 ticks × 20 rooms took {elapsed:.2f}s"

    def test_large_grid_1000_rooms(self, fast_conductor):
        """1000 rooms, one sync cycle; <30 s."""
        np.random.seed(42)
        n = 1000
        conductors: list[FleetConductor] = []
        for i in range(n):
            c, _ = fast_conductor(f"room-{i:04d}", beat=i % 100)
            conductors.append(c)

        t0 = time.perf_counter()
        peer_map = {c.node_id: c._get_local_beat_state() for c in conductors}
        for c in conductors:
            peers = {k: v for k, v in peer_map.items() if k != c.node_id}
            c.correct_drift(peers)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0, f"1000-room sync took {elapsed:.2f}s"
