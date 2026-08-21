"""End-to-end test: BFT-QD consensus → mesh persistence → restore.

Simulates a multi-node fleet breeding round, persists results to SQLite,
and verifies restore integrity.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

import numpy as np
import pytest

# -- Mock cocapn_traps before imports --
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

from swarm.fleet_bft_qd import (
    PBFTNode,
    SemanticBFTNode,
    FleetBreederConsensus,
    FleetBFTNetwork,
)
from swarm.breeder_bft_qd_integration import BreederBFTIntegration
from swarm.mesh_table_store import MeshTableStore
from swarm.mesh_vector_tables import (
    MeshVectorTable,
    FleetVectorIndex,
    VectorTableEntry,
)


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestEndToEndConsensusPersistRestore:
    def _make_network(
        self, num_nodes: int, byzantine_ids: set[int]
    ) -> tuple[FleetBFTNetwork, FleetBreederConsensus]:
        """Helper to build a network with N nodes, some Byzantine."""
        all_ids = [f"node_{i}" for i in range(num_nodes)]
        nodes = []
        for i in range(num_nodes):
            node = SemanticBFTNode(
                node_id=f"node_{i}",
                all_nodes=all_ids,
                secret_key="fleet-secret-test-key",
            )
            nodes.append(node)
        network = FleetBFTNetwork(nodes=nodes)
        for bid in byzantine_ids:
            if 0 <= bid < num_nodes:
                network._byzantine_nodes.add(f"node_{bid}")
        # Primary consensus instance
        primary = FleetBreederConsensus(
            node_id="node_0",
            all_nodes=all_ids,
            secret_key="fleet-secret-test-key",
            archive_dims=(5, 5),
        )
        return network, primary

    def test_full_round_trip(self, db_path):
        """Simulate 4-node fleet: consensus → breed → persist → crash → restore."""
        store = MeshTableStore(db_path)
        network, consensus = self._make_network(4, set())
        integration = BreederBFTIntegration(
            consensus=consensus,
            network=network,
        )

        # Simulate candidates from vector table
        candidates = [{"id": f"agent_{i}", "fitness": 0.5 + 0.1 * i} for i in range(8)]

        # Run consensus
        pairs = integration.propose_parents(candidates, batch_size=4)
        assert pairs is not None
        assert len(pairs) > 0

        # Execute breeding (mock: just create entries)
        msg = integration.consensus.propose_breeding_batch(candidates, 4)
        result = integration.commit_parents(msg)

        # Persist to mesh store
        table = MeshVectorTable(table_id="gen_0")
        for i, (a, b) in enumerate(pairs):
            entry = VectorTableEntry(
                agent_id=f"offspring_{i}",
                vector=np.array([0.1 * a, 0.1 * b], dtype=np.float32),
                timestamp=__import__("time").time(),
                node_id="primary",
                generation=1,
                fitness=0.6,
                signature="sha256:consensus",
            )
            table._entries[entry.agent_id] = entry

        count = store.save_table(table)
        assert count == len(pairs)

        # Simulate crash — restore
        restored = store.load_table("gen_0")
        assert len(restored._entries) == len(pairs)

        for i in range(len(pairs)):
            assert f"offspring_{i}" in restored._entries

    def test_byzantine_then_persist(self, db_path):
        """7 nodes with 2 Byzantine — still reach consensus and persist."""
        network, consensus = self._make_network(7, {5, 6})
        integration = BreederBFTIntegration(
            consensus=consensus,
            network=network,
        )
        store = MeshTableStore(db_path)

        candidates = [{"id": f"a{i}", "fitness": 0.5} for i in range(6)]
        pairs = integration.propose_parents(candidates, batch_size=3)

        # Should succeed despite 2 bad nodes
        assert pairs is not None

        # Persist
        table = MeshVectorTable(table_id="gen_byz")
        for i, (a, b) in enumerate(pairs):
            entry = VectorTableEntry(
                agent_id=f"child_{i}",
                vector=np.array([float(a), float(b)], dtype=np.float32),
                timestamp=__import__("time").time(),
                node_id="node_0",
                generation=1,
                fitness=0.55,
                signature="sha256:byzantine_test",
            )
            table._entries[entry.agent_id] = entry

        store.save_table(table)
        restored = store.load_table("gen_byz")
        assert len(restored._entries) == len(pairs)

    def test_failed_consensus_no_persist(self, db_path):
        """4 nodes with 2 Byzantine — consensus fails, nothing persisted."""
        network, consensus = self._make_network(4, {2, 3})
        integration = BreederBFTIntegration(
            consensus=consensus,
            network=network,
        )
        store = MeshTableStore(db_path)

        candidates = [{"id": f"a{i}", "fitness": 0.5} for i in range(4)]
        pairs = integration.propose_parents(candidates, batch_size=2)

        # Should fail: 4 nodes, 2 bad = can't reach quorum
        assert pairs is None  # No consensus

        # Verify nothing was persisted
        assert store.count_entries() == 0
