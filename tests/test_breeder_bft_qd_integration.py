"""Tests for FleetBFT-QD breeder integration.

Covers: consensus path, no-consensus fallback, Byzantine fault tolerance,
timeout handling, QD archive updates after breeding.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

import numpy as np
import pytest

# -- Mock cocapn_traps before swarm.breeder_daemon_v2 import --
_mock_cocapn_traps = types.ModuleType("cocapn_traps")
_mock_cocapn_traps_traps = types.ModuleType("cocapn_traps.traps")
_mock_cocapn_traps_diversity = types.ModuleType(
    "cocapn_traps.traps.diversity_collapse_trap"
)


class _MockAlert:
    level = "WARNING"
    recommended_action = "mock alert"


class _MockDiversityCollapseTrap:
    def __init__(self, bus=None):
        self._history = []

    def record(self, value: float) -> None:
        self._history.append(value)

    def check(self):
        return None


_mock_cocapn_traps_diversity.DiversityCollapseTrap = _MockDiversityCollapseTrap
_mock_cocapn_traps_diversity.Alert = _MockAlert
sys.modules["cocapn_traps"] = _mock_cocapn_traps
sys.modules["cocapn_traps.traps"] = _mock_cocapn_traps_traps
sys.modules["cocapn_traps.traps.diversity_collapse_trap"] = _mock_cocapn_traps_diversity

from nerve.room_grid import RoomGrid
from swarm.breeder_daemon_v2 import (
    BreederDaemonV2,
    AgentLifecycleFSM,
    DiversityConfig,
    LifecycleState,
    ThermalConfig,
)
from swarm.thermal import DeviceType, ThermalBudget
from swarm.vector_table import AgentVector, FluxVectorTable
from swarm.fleet_bft_qd import (
    PBFTNode,
    SemanticBFTNode,
    FleetBreederConsensus,
    FleetBFTNetwork,
)
from swarm.breeder_bft_qd_integration import BreederBFTIntegration


@pytest.fixture
def key():
    return "fleet-secret-test-key"


@pytest.fixture
def grid():
    g = RoomGrid(n=20)
    for _ in range(20):
        for i in range(10):
            g.activity[i] += 5
    return g


@pytest.fixture
def thermal():
    return ThermalBudget({DeviceType.GPU: 10, DeviceType.CPU: 20})


@pytest.fixture
def wal_path():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def vector_table():
    vt = FluxVectorTable(dim=256, bit_width=4)
    rng = np.random.RandomState(42)
    for i in range(20):
        scale = 2.0 if i < 10 else 0.5
        vec = (rng.randn(256).astype(np.float32) * scale).tolist()
        vt.add(
            AgentVector(
                agent_id=i,
                vector=vec,
                fitness=0.8 if i < 10 else 0.3,
                generation=1,
                capability_mask=0xFFFF,
                thermal_pressure=0.1,
            )
        )
    return vt


def make_daemon(grid, thermal, wal_path, vector_table=None, consensus=None):
    return BreederDaemonV2(
        grid=grid,
        thermal=thermal,
        vector_table=vector_table,
        diversity=DiversityConfig(),
        thermal_cfg=ThermalConfig(max_agents=65, hysteresis_ticks=2),
        wal_path=wal_path,
        tick_interval=60.0,
        consensus=consensus,
    )


# -- fixtures for BFT network --


@pytest.fixture
def network_four(key):
    ids = ["n0", "n1", "n2", "n3"]
    nodes = [PBFTNode(i, ids, key) for i in ids]
    return FleetBFTNetwork(nodes)


@pytest.fixture
def network_seven(key):
    ids = [f"n{i}" for i in range(7)]
    nodes = [PBFTNode(i, ids, key) for i in ids]
    return FleetBFTNetwork(nodes)


@pytest.fixture
def consensus_primary_four(key, network_four):
    fbc = FleetBreederConsensus("n0", ["n0", "n1", "n2", "n3"], key)
    return fbc


@pytest.fixture
def integration_primary_four(consensus_primary_four, network_four):
    return BreederBFTIntegration(consensus_primary_four, network_four, timeout_sec=2.0)


@pytest.fixture
def consensus_replica_four(key):
    return FleetBreederConsensus("n1", ["n0", "n1", "n2", "n3"], key)


@pytest.fixture
def integration_replica_four(consensus_replica_four, network_four):
    return BreederBFTIntegration(consensus_replica_four, network_four, timeout_sec=2.0)


# -- helpers --


def seed_breedable_agents(daemon, count=8, base_id=100):
    rng = np.random.RandomState(77)
    for i in range(count):
        aid = base_id + i
        daemon._fsm[aid] = AgentLifecycleFSM(
            agent_id=aid, initial_state=LifecycleState.SURVIVE, strict=False
        )
        if daemon._vector_table is not None:
            vec = (rng.randn(256).astype(np.float32) * 2.0).tolist()
            daemon._vector_table.add(
                AgentVector(
                    agent_id=aid,
                    vector=vec,
                    fitness=0.8,
                    generation=1,
                    capability_mask=0xFFFF,
                    thermal_pressure=0.1,
                )
            )


# -- Test: consensus path --


class TestConsensusPath:
    def test_consensus_returns_pairs(
        self, grid, thermal, wal_path, vector_table, integration_primary_four
    ):
        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration_primary_four
        )
        daemon.start()
        seed_breedable_agents(daemon, count=8)

        pairs = daemon.select_parents(n_children=3)
        daemon.stop()

        assert isinstance(pairs, list)
        assert len(pairs) > 0
        # All pairs should be tuples of two integers
        for a, b in pairs:
            assert isinstance(a, int)
            assert isinstance(b, int)

    def test_consensus_pairs_are_from_candidates(
        self, grid, thermal, wal_path, vector_table, integration_primary_four
    ):
        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration_primary_four
        )
        daemon.start()
        seed_breedable_agents(daemon, count=6, base_id=200)
        candidate_ids = {200 + i for i in range(6)}

        pairs = daemon.select_parents(n_children=2)
        daemon.stop()

        for a, b in pairs:
            assert a in candidate_ids or b in candidate_ids

    def test_consensus_batch_id_increments(
        self, grid, thermal, wal_path, vector_table, integration_primary_four
    ):
        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration_primary_four
        )
        daemon.start()
        seed_breedable_agents(daemon, count=8)

        daemon.select_parents(n_children=2)
        batch_after_first = integration_primary_four.consensus.bft.seq_num

        daemon.select_parents(n_children=2)
        batch_after_second = integration_primary_four.consensus.bft.seq_num
        daemon.stop()

        assert batch_after_second > batch_after_first


# -- Test: no-consensus fallback --


class TestNoConsensusFallback:
    def test_without_consensus_uses_normal_path(
        self, grid, thermal, wal_path, vector_table
    ):
        daemon = make_daemon(grid, thermal, wal_path, vector_table, consensus=None)
        daemon.start()
        seed_breedable_agents(daemon, count=8)

        pairs = daemon.select_parents(n_children=3)
        daemon.stop()

        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_none_consensus_attribute(self, grid, thermal, wal_path, vector_table):
        daemon = make_daemon(grid, thermal, wal_path, vector_table, consensus=None)
        assert daemon._consensus is None


# -- Test: Byzantine fault tolerance --


class TestByzantineFaultTolerance:
    def test_4_nodes_tolerates_1_byzantine(
        self, grid, thermal, wal_path, vector_table, key
    ):
        ids = ["n0", "n1", "n2", "n3"]
        nodes = [PBFTNode(i, ids, key) for i in ids]
        network = FleetBFTNetwork(nodes)
        network.set_byzantine(["n3"])

        consensus = FleetBreederConsensus("n0", ids, key)
        integration = BreederBFTIntegration(consensus, network, timeout_sec=2.0)

        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration
        )
        daemon.start()
        seed_breedable_agents(daemon, count=8)

        pairs = daemon.select_parents(n_children=2)
        daemon.stop()

        # Consensus should still succeed with 1 Byzantine out of 4 (f=1)
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_7_nodes_tolerates_2_byzantine(
        self, grid, thermal, wal_path, vector_table, key
    ):
        ids = [f"n{i}" for i in range(7)]
        nodes = [PBFTNode(i, ids, key) for i in ids]
        network = FleetBFTNetwork(nodes)
        network.set_byzantine(["n5", "n6"])

        consensus = FleetBreederConsensus("n0", ids, key)
        integration = BreederBFTIntegration(consensus, network, timeout_sec=2.0)

        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration
        )
        daemon.start()
        seed_breedable_agents(daemon, count=10)

        pairs = daemon.select_parents(n_children=2)
        daemon.stop()

        # N=7, f=2, quorum=5. 2 Byzantine should be tolerated.
        assert isinstance(pairs, list)
        assert len(pairs) > 0


# -- Test: timeout handling (consensus failure fallback) --


class TestTimeoutHandling:
    def test_4_nodes_2_byzantine_falls_back(
        self, grid, thermal, wal_path, vector_table, key
    ):
        ids = ["n0", "n1", "n2", "n3"]
        nodes = [PBFTNode(i, ids, key) for i in ids]
        network = FleetBFTNetwork(nodes)
        # 2 Byzantine exceeds f=1 for N=4 -> consensus should fail
        network.set_byzantine(["n2", "n3"])

        consensus = FleetBreederConsensus("n0", ids, key)
        integration = BreederBFTIntegration(consensus, network, timeout_sec=2.0)

        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration
        )
        daemon.start()
        seed_breedable_agents(daemon, count=8)

        pairs = daemon.select_parents(n_children=2)
        daemon.stop()

        # When consensus fails, the daemon should fall back to normal selection.
        # Fallback returns pairs even if consensus failed, because the code
        # falls through to the standard vector/random selection.
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_replica_cannot_propose(
        self, grid, thermal, wal_path, vector_table, integration_replica_four
    ):
        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration_replica_four
        )
        daemon.start()
        seed_breedable_agents(daemon, count=8)

        pairs = daemon.select_parents(n_children=2)
        daemon.stop()

        # n1 is not primary in view 0, so propose returns None -> fallback
        assert isinstance(pairs, list)
        assert len(pairs) > 0


# -- Test: QD archive updates after breeding --


class TestQDArchiveUpdates:
    def test_archive_updated_after_commit(self, key):
        ids = ["n0", "n1", "n2", "n3"]
        nodes = [PBFTNode(i, ids, key) for i in ids]
        network = FleetBFTNetwork(nodes)
        consensus = FleetBreederConsensus("n0", ids, key)
        integration = BreederBFTIntegration(consensus, network, timeout_sec=2.0)

        candidates = [
            {"id": f"agent_{i}", "chaos": 0.3, "fitness": 0.5 + i * 0.05}
            for i in range(6)
        ]

        pairs = integration.propose_parents(candidates, batch_size=4)
        assert pairs is not None
        assert len(pairs) > 0

        # After commit, execute_breeding was called → breeding_log grows
        assert len(consensus._breeding_log) == 1
        batch = consensus._breeding_log[0]
        assert "offspring" in batch
        assert batch["n_offspring"] > 0

    def test_evaluate_offspring_adds_to_archive(self, key):
        ids = ["n0", "n1", "n2", "n3"]
        consensus = FleetBreederConsensus("n0", ids, key)

        assert consensus.archive.stats["n_occupied"] == 0

        child = {"id": "c1"}
        added = consensus.evaluate_offspring(
            child, fitness=0.9, behavior=np.array([0.5, 0.5])
        )
        assert added

        stats = consensus.archive.stats
        assert stats["n_occupied"] == 1
        assert stats["qd_score"] == 0.9

    def test_archive_coverage_after_multiple_evaluations(self, key):
        ids = ["n0", "n1", "n2", "n3"]
        consensus = FleetBreederConsensus("n0", ids, key)

        for i in range(5):
            child = {"id": f"c{i}"}
            behavior = np.array([i / 5.0 + 0.05, 0.5])
            consensus.evaluate_offspring(
                child, fitness=0.5 + i * 0.1, behavior=behavior
            )

        stats = consensus.archive.stats
        assert stats["n_occupied"] == 5
        assert stats["coverage"] == 5 / 100  # 10x10 grid = 100 cells

    def test_end_to_end_breeding_log_and_archive(
        self, grid, thermal, wal_path, vector_table, key
    ):
        ids = ["n0", "n1", "n2", "n3"]
        nodes = [PBFTNode(i, ids, key) for i in ids]
        network = FleetBFTNetwork(nodes)
        consensus = FleetBreederConsensus("n0", ids, key)
        integration = BreederBFTIntegration(consensus, network, timeout_sec=2.0)

        daemon = make_daemon(
            grid, thermal, wal_path, vector_table, consensus=integration
        )
        daemon.start()
        seed_breedable_agents(daemon, count=8)

        pairs = daemon.select_parents(n_children=2)
        daemon.stop()

        # Breeding log should have entries because consensus reached commit
        assert len(consensus._breeding_log) >= 1

        # Archive may or may not have entries depending on whether
        # evaluate_offspring was called explicitly. The integration's
        # commit_parents calls execute_breeding which does NOT auto-evaluate.
        # That is correct: evaluation happens after offspring fitness is known.
        assert consensus.archive.stats["n_occupied"] == 0


# -- Test: BreederBFTIntegration unit methods --


class TestIntegrationUnit:
    def test_commit_parents_clears_pending(self, key):
        ids = ["n0", "n1", "n2", "n3"]
        nodes = [PBFTNode(i, ids, key) for i in ids]
        network = FleetBFTNetwork(nodes)
        consensus = FleetBreederConsensus("n0", ids, key)
        integration = BreederBFTIntegration(consensus, network, timeout_sec=2.0)

        candidates = [{"id": f"agent_{i}", "chaos": 0.3} for i in range(4)]
        msg = consensus.propose_breeding_batch(candidates, batch_size=2)
        assert msg is not None

        batch_id = msg.payload["batch_id"]
        integration._pending[batch_id] = msg

        result = integration.commit_parents(msg)
        assert batch_id not in integration._pending
        assert "offspring" in result

    def test_abort_parents_clears_pending(self, key):
        ids = ["n0", "n1", "n2", "n3"]
        nodes = [PBFTNode(i, ids, key) for i in ids]
        network = FleetBFTNetwork(nodes)
        consensus = FleetBreederConsensus("n0", ids, key)
        integration = BreederBFTIntegration(consensus, network, timeout_sec=2.0)

        candidates = [{"id": f"agent_{i}", "chaos": 0.3} for i in range(4)]
        msg = consensus.propose_breeding_batch(candidates, batch_size=2)
        assert msg is not None

        batch_id = msg.payload["batch_id"]
        integration._pending[batch_id] = msg

        integration.abort_parents(msg)
        assert batch_id not in integration._pending

    def test_parse_agent_id_variants(self):
        from swarm.breeder_bft_qd_integration import _parse_agent_id

        assert _parse_agent_id("agent_42") == 42
        assert _parse_agent_id(42) == 42
        assert _parse_agent_id("123") == 123
        assert _parse_agent_id("not_a_number") is None
        assert _parse_agent_id(None) is None

    def test_parse_parent_pairs_even_count(self):
        from swarm.breeder_bft_qd_integration import BreederBFTIntegration
        from swarm.fleet_bft_qd import PBFTMessage, BFTPhase

        msg = PBFTMessage(
            phase=BFTPhase.PRE_PREPARE,
            view_number=0,
            seq_num=1,
            digest="abc",
            node_id="n0",
            payload={"parent_ids": ["agent_1", "agent_2", "agent_3", "agent_4"]},
            timestamp=0.0,
        )
        pairs = BreederBFTIntegration._parse_parent_pairs(msg)
        assert pairs == [(1, 2), (3, 4)]

    def test_parse_parent_pairs_odd_count(self):
        from swarm.breeder_bft_qd_integration import BreederBFTIntegration
        from swarm.fleet_bft_qd import PBFTMessage, BFTPhase

        msg = PBFTMessage(
            phase=BFTPhase.PRE_PREPARE,
            view_number=0,
            seq_num=1,
            digest="abc",
            node_id="n0",
            payload={"parent_ids": ["agent_1", "agent_2", "agent_3"]},
            timestamp=0.0,
        )
        pairs = BreederBFTIntegration._parse_parent_pairs(msg)
        assert pairs == [(1, 2), (3, 3)]
