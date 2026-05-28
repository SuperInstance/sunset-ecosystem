# FleetBFT-QD — Byzantine Fault Tolerant Consensus + Quality Diversity Breeding

## Overview

FleetBFT-QD combines two cutting-edge mathematical foundations into a single fleet subsystem:

1. **Practical Byzantine Fault Tolerance (PBFT)** — Castro & Liskov (1999), extended with HotStuff pipelining concepts (Yin et al, 2019) and semantic confidence weighting (WBFT, 2025).
2. **Quality Diversity (QD)** — MAP-Elites archive (Mouret & Clune, 2015) with CMA-ES self-adaptive emitters (Hansen & Ostermeier, 2001).

The result: **BFT-gated breeding**. Every breeding batch decision is agreed upon by the fleet via Byzantine consensus, and parent selection is informed by a diversity archive that ensures the fleet explores the full behavior space rather than converging prematurely.

## Mathematical Foundations

### PBFT Consensus

For a fleet of N nodes, the protocol tolerates up to f Byzantine (arbitrarily faulty/malicious) nodes where:

```
f = floor((N - 1) / 3)
quorum = 2f + 1
```

**Five phases:**
1. **REQUEST** — Client sends operation to primary.
2. **PRE-PREPARE** — Primary assigns sequence number, broadcasts to all.
3. **PREPARE** — Replicas validate digest and primary identity, broadcast prepare.
4. **COMMIT** — Replicas collect 2f+1 prepares, broadcast commit.
5. **REPLY** — Replicas collect 2f+1 commits, execute, reply to client.

**View Change Protocol:**
- Replicas start a timer when receiving a pre-prepare.
- If the timer expires (leader unresponsive), replicas broadcast VIEW-CHANGE.
- New primary (deterministic rotation: primary = nodes[view % N]) collects 2f+1 view-changes.
- New primary broadcasts NEW-VIEW; all replicas adopt the new view.

### Semantic Confidence Weighting (WBFT)

Traditional BFT assumes binary correctness: a node is either honest or Byzantine. In multi-agent fleets, agents have varying confidence levels based on:

- **Historical accuracy** — EMA-tracked reputation per task type.
- **Task complexity** — Larger payloads increase uncertainty.
- **Capability match** — Task-specific performance history.

Confidence score: `c ∈ [0.1, 1.0]`

Weighted quorum: `Σ(c_i) ≥ quorum_threshold`

This allows high-confidence agents to effectively "speak louder" in consensus, while low-confidence agents need more votes to reach agreement.

### MAP-Elites Archive

The QD archive is an N-dimensional grid where each cell holds the best individual for that behavioral region.

**Behavior descriptor:** A vector `b ∈ ℝⁿ` where each dimension is a behavioral trait (e.g., exploration rate, communication frequency, success rate).

**Grid discretization:**
```
idx_i = clamp(floor((b_i - low_i) / (high_i - low_i) * grid_i), 0, grid_i - 1)
```

**Metrics:**
- **Coverage** = occupied_cells / total_cells (% of behavior space explored)
- **QD-Score** = Σ fitness_i across all occupied cells

### CMA-ES Emitter

Covariance Matrix Adaptation Evolution Strategy:
- Maintains multivariate Gaussian `N(m, σ²C)`
- Samples offspring: `x = m + σ * C^{1/2} * z` where `z ~ N(0, I)`
- Updates mean, covariance, and step-size from elite selection
- Self-adapts exploration/exploitation trade-off

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  FleetBreederConsensus                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   BFT Layer  │      │   QD Layer   │                     │
│  │  (PBFTNode)  │◄────►│ (QDArchive)  │                     │
│  │              │      │              │                     │
│  │  SemanticBFT │      │  CMAESEmitter│                     │
│  │  + WBFT conf │      │  + MAP-Elites│                     │
│  └──────────────┘      └──────────────┘                     │
├─────────────────────────────────────────────────────────────┤
│  Integration:                                               │
│  • wire_to_fleet_conductor() — consensus for state changes  │
│  • wire_to_mesh_gossip()     — CRDT archive propagation      │
│  • wire_to_metronome_bridge()— heartbeat-driven view sync    │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic PBFT Consensus

```python
from swarm.fleet_bft_qd import PBFTNode, FleetBFTNetwork

# Create 4 nodes (tolerates 1 Byzantine fault)
nodes = [PBFTNode(f"n{i}", ["n0","n1","n2","n3"], "secret") for i in range(4)]
net = FleetBFTNetwork(nodes)

# Run consensus on a breeding batch
ok = net.broadcast_request("breed_batch", {
    "parent_ids": ["agent_1", "agent_2"],
    "mutation_rate": 0.3,
})
assert ok  # 2f+1 = 3 nodes agreed
```

### Byzantine Fault Injection

```python
# Mark n3 as Byzantine (arbitrary/malicious)
net.set_byzantine(["n3"])

# Consensus still works (N=4, f=1, 1 Byzantine ≤ f)
ok = net.broadcast_request("breed", {"batch_id": 2})
assert ok

# But 2 Byzantine nodes breaks tolerance
net.set_byzantine(["n2", "n3"])
ok = net.broadcast_request("breed", {"batch_id": 3})
assert not ok
```

### Network Partition Recovery

```python
# Partition the primary
net.set_partitioned(["n0"])

# Run view change to elect new primary
net.run_view_change()  # n1 becomes primary in view 1

# Clear partition, continue with new primary
net.clear_faults()
ok = net.broadcast_request("breed", {"batch_id": 4})
assert ok
```

### Quality Diversity Breeding

```python
from swarm.fleet_bft_qd import FleetBreederConsensus
import numpy as np

fbc = FleetBreederConsensus(
    node_id="n0",
    all_nodes=["n0", "n1", "n2", "n3"],
    secret_key="fleet-secret",
    archive_dims=(10, 10),           # 2D behavior space
    behavior_bounds=[(0.0, 1.0), (0.0, 1.0)],
)

# Evaluate offspring and add to archive
child = {"id": "offspring_42", "parent": "agent_1"}
behavior = np.array([0.7, 0.3])  # High exploration, low communication
fitness = 0.85

added = fbc.evaluate_offspring(child, fitness, behavior)
if added:
    print("Improved archive cell!")

print(fbc.archive.stats)
# {'n_occupied': 1, 'coverage': 0.01, 'qd_score': 0.85, ...}
```

### Semantic Confidence

```python
from swarm.fleet_bft_qd import SemanticBFTNode

node = SemanticBFTNode("n0", ["n0", "n1", "n2", "n3"], "secret")

# Update reputation based on historical accuracy
node.update_reputation("n1", "breed", success=True)
node.update_reputation("n2", "breed", success=False)

# Confidence for a new task
conf = node.compute_confidence("breed", {"parents": ["a", "b"]})
print(f"Confidence: {conf:.2f}")  # ~0.9 for simple payload, good history
```

## API Reference

### PBFTNode

| Method | Description |
|--------|-------------|
| `handle_request(op, payload)` | Client initiates request (primary only) |
| `handle_pre_prepare(msg)` | Replica validates and prepares |
| `handle_prepare(msg)` | Replica commits on quorum prepares |
| `handle_commit(msg)` | Replica executes on quorum commits |
| `start_view_change()` | Initiate leader recovery |
| `handle_view_change(msg)` | Process view-change from peer |
| `handle_new_view(msg)` | Adopt new view |
| `get_status()` | Node status dict |

### SemanticBFTNode (extends PBFTNode)

| Method | Description |
|--------|-------------|
| `compute_confidence(task, payload)` | Compute semantic confidence score |
| `update_reputation(node, task, success)` | EMA reputation update |
| `weighted_quorum_reached(msgs)` | Check weighted vote threshold |

### QDArchive

| Method | Description |
|--------|-------------|
| `add(descriptor, individual, fitness)` | Insert into archive |
| `get_random_elite()` | Sample random occupied cell |
| `get_all_elites()` | All (individual, fitness) pairs |
| `coverage` | % grid explored |
| `qd_score` | Sum fitness across archive |

### CMAESEmitter

| Method | Description |
|--------|-------------|
| `sample(n)` | Sample n individuals |
| `update(elites)` | Adapt distribution from elites |

### FleetBreederConsensus

| Method | Description |
|--------|-------------|
| `propose_breeding_batch(candidates, size)` | BFT-gated batch proposal |
| `execute_breeding(payload)` | Run committed batch |
| `evaluate_offspring(child, fitness, behavior)` | Archive offspring |
| `get_sync_payload()` | CRDT-compatible sync data |
| `get_status()` | Full status dict |

### FleetBFTNetwork

| Method | Description |
|--------|-------------|
| `broadcast_request(op, payload)` | End-to-end consensus round |
| `run_view_change()` | Simulate view change across nodes |
| `set_byzantine(ids)` | Inject Byzantine faults |
| `set_partitioned(ids)` | Simulate network partitions |
| `clear_faults()` | Remove all faults |

## Integration with Fleet Infrastructure

### HolonomyConsensus Upgrade

The existing `nexus.distributed_consensus.HolonomyConsensus` provides simple vote-counting. FleetBFT-QD upgrades it to full PBFT with:

- Cryptographic digests on all proposals
- Pre-prepare / prepare / commit phases
- View change protocol for leader recovery
- Semantic confidence weighting

### MetronomeBridge

The metronome heartbeat drives view synchronization:
- Each beat, replicas check leader liveness
- Timeout → automatic view change
- New primary synchronized to metronome phase

### MeshVectorGossip

QD archive updates propagate via CRDT gossip:
- `FleetBreederConsensus.get_sync_payload()` produces merge-friendly deltas
- `apply_sync_payload()` merges peer archive stats
- Full grid merge delegated to `mesh_vector_tables.py` vector merging

### FleetConductorV2

BFT consensus guards conductor state changes:
- Room grid resize decisions
- Thermal threshold updates
- Subsystem enable/disable votes

## Research References

1. **PBFT**: Castro, M. & Liskov, B. (1999). "Practical Byzantine Fault Tolerance". OSDI.
2. **HotStuff**: Yin, M. et al (2019). "HotStuff: BFT Consensus in the Lens of Blockchain". arXiv:1803.05069.
3. **WBFT**: (2025). "Weighted Byzantine Fault Tolerance Consensus Driven Trusted Multiple Large Language Models Network". arXiv:2505.05103.
4. **MAP-Elites**: Mouret, J.-B. & Clune, J. (2015). "Illuminating search spaces by mapping elites". arXiv:1504.04909.
5. **CMA-ES**: Hansen, N. & Ostermeier, A. (2001). "Completely Derandomized Self-Adaptation in Evolution Strategies". Evolutionary Computation.
6. **CMA-MAE**: Bryant, M. et al (2024). "CMA-MAE: Covariance Matrix Adaptation for MAP-Elites". arXiv:2407.00190.

## Test Summary

72 tests covering:
- PBFT phase correctness (12 tests)
- Byzantine fault tolerance: 0, 1, 2 faults across 4, 7, 10 nodes (6 tests)
- View change protocol: leader crash, timeout, recovery (4 tests)
- Network partitions: split brain, rejoin (3 tests)
- Semantic confidence: reputation, weighted quorum (6 tests)
- Quorum certificates: validation, signatures (3 tests)
- QD Archive: insertion, coverage, qd_score, 3D grids (9 tests)
- CMA-ES: sampling, update convergence, sigma adaptation (5 tests)
- FleetBreederConsensus: proposal, execution, evaluation, sync (7 tests)
- FleetBFTNetwork: end-to-end, faults, partitions (8 tests)
- Edge cases: empty payloads, zero bounds, large dimensions (9 tests)

Run: `pytest tests/test_fleet_bft_qd.py -v`
