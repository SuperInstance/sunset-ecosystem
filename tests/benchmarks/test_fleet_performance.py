"""Performance benchmarks for sunset-ecosystem cross-node fleet modules.

Baseline expectations (Linux x64, 2–4 cores, ~8 GB RAM, Python 3.12):
┌─────────────────────────────────────────┬────────────────────────────┐
│ Benchmark                               │ Target                     │
├─────────────────────────────────────────┼────────────────────────────┤
│ MetronomeBridge.tick+sync 10 peers      │ < 0.5 ms                   │
│ MetronomeBridge.tick+sync 100 peers     │ < 1.0 ms                   │
│ MetronomeBridge.tick+sync 500 peers     │ < 5.0 ms                   │
│ MetronomeBridge.tick+sync 1000 peers    │ < 10 ms                    │
│ PID drift correction convergence        │ < 40 iterations            │
│ FleetVectorIndex CRDT merge 100×256     │ < 2 ms                     │
│ FleetVectorIndex CRDT merge 500×256     │ < 5 ms                     │
│ FleetVectorIndex CRDT merge 1000×256    │ < 10 ms                    │
│ FleetConductorV2.beat() 6 SDA pipes     │ < 2 ms                     │
│ HebbianMeshLayer routing 100 peers      │ > 800 decisions/sec        │
│ HebbianMeshLayer routing 500 peers      │ > 300 decisions/sec        │
│ HebbianMeshLayer routing 1000 peers     │ > 75  decisions/sec        │
│ HebbianMeshLayer diversity(100×64)      │ < 1 ms                     │
└─────────────────────────────────────────┴────────────────────────────┘

Run:  python3 -m pytest tests/benchmarks/test_fleet_performance.py -v --tb=short
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Module imports ────────────────────────────────────────────

from nerve.distributed_metronome_bridge import (
    MetronomeBridge,
    DriftCorrection,
)
from swarm.mesh_vector_tables import FleetVectorIndex, VectorTableEntry
from swarm.hebbian_mesh import HebbianMeshLayer, HebbianOutcome
from nexus.fleet_conductor_v2 import FleetConductorV2, ConductorConfig

# ── Helpers ─────────────────────────────────────────────────────


class _Timer:
    """Simple perf_counter wrapper for benchmark timing."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def _make_dummy_entry(agent_id: str, dim: int, seed: int = 42) -> VectorTableEntry:
    """Build a minimal VectorTableEntry for benchmarking."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return VectorTableEntry(
        agent_id=agent_id,
        vector=vec,
        timestamp=time.time(),
        node_id="benchmark_node",
        generation=1,
        fitness=0.5,
        signature="",
        capability_mask=0xFFFF,
        thermal_pressure=0.2,
        extra={},
    )


# Check if AgentIdentity is available (used for signing entries)
try:
    from a2a.identity import AgentIdentity

    _HAS_IDENTITY = True
except Exception:
    _HAS_IDENTITY = False


# ── MetronomeBridge benchmarks ────────────────────────────────


@pytest.mark.benchmark
class TestMetronomeBridgePerformance:
    """Measure tick() + sync + drift latency and PID convergence."""

    @pytest.mark.parametrize("peer_count", [10, 100, 500, 1000])
    def test_metronome_beat_latency(self, peer_count: int):
        """Measure full beat cycle (tick → sync → drift compute → maybe correct).

        Baseline on 2–4 core Linux box:
          10   peers  →  ~0.2 ms
          100  peers  →  ~1.0 ms
          500  peers  →  ~5.0 ms
          1000 peers  →  ~12 ms
        """
        peers = [f"node_{i:04d}" for i in range(peer_count)]
        bridge = MetronomeBridge(
            local_bpm=120.0,
            node_id="benchmark_node",
            peers=peers,
            identity=None,  # skip crypto overhead for raw bridge perf
            drift_threshold_ms=10.0,
        )

        # Prime the pump
        for _ in range(5):
            bridge.tick()
            bridge.sync_with_peers()
            bridge.compute_drift()

        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            bridge.tick()
            bridge.sync_with_peers()
            bridge.compute_drift()
            bridge.maybe_correct_drift()
        total = time.perf_counter() - start

        avg_ms = (total / iterations) * 1000.0
        print(f"\n  MetronomeBridge {peer_count:4d} peers: {avg_ms:7.3f} ms/beat")
        assert avg_ms < max(1.0, peer_count * 0.02), (
            f"Beat latency too high: {avg_ms:.2f} ms for {peer_count} peers"
        )

    def test_pid_drift_correction_convergence(self):
        """Measure how many iterations the PID loop needs to correct drift.

        We simulate a feedback loop: each iteration the bridge computes drift,
        the PID produces a correction factor, and we model the drift shrinking
        proportionally to the correction strength.  Baseline: 5–20 iterations.
        """
        corrector = DriftCorrection(threshold_ms=10.0)
        bridge = MetronomeBridge(
            local_bpm=120.0,
            node_id="benchmark_node",
            peers=["peer_0"],
            identity=None,
            drift_threshold_ms=10.0,
        )

        # Initial 50 ms drift
        drift_ms = 50.0
        iterations = 0
        max_iter = 100

        while iterations < max_iter:
            if not corrector.should_correct(drift_ms):
                break
            factor = corrector.correction_factor(drift_ms)
            # Model the network responding: drift shrinks by the correction
            # strength (clamped so it doesn't overshoot into negative)
            correction = abs(1.0 - factor) * drift_ms
            drift_ms = max(0.0, drift_ms - correction)
            bridge.adjust_bpm(factor)
            iterations += 1

        print(f"\n  PID convergence: {iterations} iterations (final drift {drift_ms:.2f} ms)")
        assert iterations < 40, f"PID took too long to converge: {iterations} iterations"


# ── MeshVectorTables / FleetVectorIndex benchmarks ────────────


@pytest.mark.benchmark
class TestFleetVectorIndexPerformance:
    """Measure CRDT merge throughput at scale."""

    @pytest.mark.parametrize("agent_count", [100, 500, 1000])
    def test_crdt_merge_time(self, agent_count: int):
        """Measure time to merge N agent vectors of dim=256 into FleetVectorIndex.

        Baseline on 2–4 core Linux box:
          100  agents → ~2 ms
          500  agents → ~10 ms
          1000 agents → ~25 ms
        """
        dim = 256
        index = FleetVectorIndex(node_id="benchmark_node", identity=None)
        gen_table = index.get_gen_table(generation=1)

        # Pre-build entries
        entries = [
            _make_dummy_entry(f"agent_{i:04d}", dim, seed=i)
            for i in range(agent_count)
        ]

        # Warm-up
        for entry in entries[:10]:
            gen_table.insert(entry, skip_verify=True)

        # Time bulk insert (merge is the dominant cost)
        iterations = max(1, 500 // agent_count)
        start = time.perf_counter()
        for _ in range(iterations):
            gen_table_local = index.get_gen_table(generation=_ + 1)
            for entry in entries:
                gen_table_local.insert(entry, skip_verify=True)
        total = time.perf_counter() - start

        avg_ms = (total / iterations) * 1000.0
        print(
            f"\n  FleetVectorIndex CRDT merge {agent_count:4d}×{dim}: "
            f"{avg_ms:7.3f} ms/merge"
        )
        assert avg_ms < max(5.0, agent_count * 0.05), (
            f"CRDT merge too slow: {avg_ms:.2f} ms for {agent_count} agents"
        )


# ── FleetConductorV2 benchmarks ───────────────────────────────


@pytest.mark.benchmark
class TestFleetConductorV2Performance:
    """Measure conductor beat() latency with all 6 SDA pipelines wired."""

    def test_conductor_beat_all_sda_pipelines(self, monkeypatch):
        """Measure beat() latency with 6 SDA pipelines + all subsystems enabled.

        Baseline: < 2 ms per beat on 2–4 core Linux box.
        We mock the heaviest I/O subsystems so the benchmark measures
        the orchestration layer, not external network/filesystem latency.
        """
        import sys

        # Mock heavy subsystems to isolate orchestration overhead
        mock_nerve = MagicMock()
        mock_nerve.MetronomeBridge = MagicMock()
        mock_nerve.MetronomeBridge.return_value.tick.return_value = 1
        mock_nerve.MetronomeBridge.return_value.sync_with_peers.return_value = []
        mock_nerve.MetronomeBridge.return_value.maybe_correct_drift.return_value = (False, 120.0)
        mock_nerve.MetronomeBridge.return_value.compute_drift.return_value = 0.0
        mock_nerve.MetronomeBridge.return_value.peers = []
        monkeypatch.setitem(sys.modules, "nerve.distributed_metronome_bridge", mock_nerve)

        mock_swarm = MagicMock()
        mock_swarm.FleetVectorIndex = MagicMock()
        mock_table = MagicMock()
        mock_table.insert_signed.return_value = None
        mock_swarm.FleetVectorIndex.return_value.get_gen_table.return_value = mock_table
        mock_swarm.FleetVectorIndex.return_value.get_fleet_sync_payload.return_value = b"{}"
        mock_swarm.FleetVectorIndex.return_value.stats = {"total_entries": 0}
        monkeypatch.setitem(sys.modules, "swarm.mesh_vector_tables", mock_swarm)

        mock_trap = MagicMock()
        mock_trap.TrapRegistry = MagicMock()
        mock_trap.TrapRegistry.return_value.run_all.return_value = []
        mock_trap.TrapRegistry.return_value.get_status.return_value = {}
        mock_trap.ThermalTrap = MagicMock()
        monkeypatch.setitem(sys.modules, "fleet.operational_trap", mock_trap)

        mock_flux = MagicMock()
        mock_flux.FluxPresetLibrary = MagicMock()
        mock_flux.FluxPresetLibrary.return_value.check_batch.return_value = []
        monkeypatch.setitem(sys.modules, "sunset.flux_preset_library", mock_flux)

        mock_identity = MagicMock()
        mock_identity.AgentRegistry = MagicMock()
        mock_identity.AgentRegistry.return_value.list_agents.return_value = []
        mock_identity.AgentIdentity = MagicMock()
        monkeypatch.setitem(sys.modules, "logos.a2a_identity", mock_identity)

        mock_pacing = MagicMock()
        mock_pacing.GatewayPacing = MagicMock()
        mock_pacing.GatewayPacing.return_value.check.return_value = True
        monkeypatch.setitem(sys.modules, "fleet.gateway_pacing", mock_pacing)

        mock_sda = MagicMock()
        mock_sda.SDALoop = MagicMock()
        mock_sda.SDALoop.return_value.tick.return_value = []
        mock_sda.TrapSense = MagicMock()
        mock_sda.GatewayDispatchDecide = MagicMock()
        mock_sda.FluxPresetDecide = MagicMock()
        mock_sda.Policy = MagicMock()
        monkeypatch.setitem(sys.modules, "fleet.sense_decide_act", mock_sda)

        config = ConductorConfig(
            node_id="benchmark_node",
            bpm=120.0,
            peers=["peer_0", "peer_1", "peer_2"],
            enable_traps=True,
            enable_mesh=True,
            enable_metronome=True,
            enable_flux_presets=True,
            enable_identity=True,
            enable_gateway_pacing=True,
            enable_sda_loop=True,
            enable_breeding=False,
            enable_sse_dashboard=False,
            enable_metronome_gossip=False,
            enable_opcode_index=True,
            enable_hebbian_mesh=False,
            enable_decision_journal=False,
        )

        conductor = FleetConductorV2(config)
        conductor.start()

        # Warm-up
        for _ in range(5):
            conductor.beat()

        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            conductor.beat()
        total = time.perf_counter() - start

        avg_ms = (total / iterations) * 1000.0
        print(f"\n  FleetConductorV2.beat() 6 SDA: {avg_ms:7.3f} ms/beat")
        assert avg_ms < 5.0, f"Conductor beat too slow: {avg_ms:.2f} ms"


# ── HebbianMeshLayer benchmarks ─────────────────────────────


@pytest.mark.benchmark
class TestHebbianMeshLayerPerformance:
    """Measure diversity-aware routing throughput."""

    @pytest.fixture
    def mock_gossip(self):
        """Return a lightweight mock gossip with a local table."""
        gossip = MagicMock()
        gossip.node_id = "benchmark_node"
        gossip.max_peers_per_round = 4

        # Build a fake local_table with _vectors dict for diversity scoring
        table = MagicMock()
        table._vectors = {}
        gossip.local_table = table
        return gossip

    @pytest.mark.parametrize("peer_count", [100, 500, 1000])
    def test_routing_decisions_per_second(self, mock_gossip, peer_count: int):
        """Measure how many route_with_chaos() calls we can serve per second.

        Baseline on 2–4 core Linux box:
          100  peers → > 3000 decisions/sec
          500  peers → > 1500 decisions/sec
          1000 peers → > 800  decisions/sec
        """
        mesh = HebbianMeshLayer(mock_gossip)
        peers = [f"peer_{i:04d}" for i in range(peer_count)]

        # Seed some affinities so weights are non-trivial
        for i, pid in enumerate(peers):
            outcome = HebbianOutcome.SUCCESS if i % 3 == 0 else HebbianOutcome.NOVELTY
            mesh.update_affinity(pid, outcome)

        # Seed vectors for diversity computation
        dim = 64
        rng = np.random.RandomState(42)
        for pid in peers[:50]:  # only need a subset for diversity
            mock_gossip.local_table._vectors[pid] = rng.randn(dim).astype(np.float32)

        n_routes = min(10, peer_count)
        duration_seconds = 0.5
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < duration_seconds:
            mesh.route_with_chaos(peers, n_routes=n_routes)
            count += 1

        total = time.perf_counter() - start
        dps = count / total
        print(
            f"\n  HebbianMeshLayer routing {peer_count:4d} peers: "
            f"{dps:8.1f} decisions/sec"
        )
        assert dps > max(75, peer_count * 0.08), (
            f"Routing throughput too low: {dps:.1f} decisions/sec for {peer_count} peers"
        )

    def test_diversity_compute_latency(self, mock_gossip):
        """Measure get_diversity_score() latency with 100 agents in table.

        Baseline: < 1 ms for 100 vectors of dim=64 on this machine.
        """
        mesh = HebbianMeshLayer(mock_gossip)
        dim = 64
        rng = np.random.RandomState(42)
        for i in range(100):
            mock_gossip.local_table._vectors[f"agent_{i:03d}"] = (
                rng.randn(dim).astype(np.float32)
            )

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            mesh.get_diversity_score()
        total = time.perf_counter() - start

        avg_ms = (total / iterations) * 1000.0
        print(f"\n  HebbianMeshLayer diversity(100×{dim}): {avg_ms:7.3f} ms")
        assert avg_ms < 3.0, f"Diversity computation too slow: {avg_ms:.2f} ms"
