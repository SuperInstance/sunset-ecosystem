# FleetDiversity — Pyversity-Powered Diversity Selection

## Overview

FleetDiversity wraps **[Pringled/pyversity](https://github.com/Pringled/pyversity)** (491⭐) to bring research-backed diversification algorithms to the Cocapn Fleet breeding pipeline.

**What it does:** Instead of selecting parents purely by fitness (which leads to premature convergence), FleetDiversity uses probabilistic repulsion (DPP), maximal marginal relevance (MMR), and other strategies to ensure the fleet explores the full behavior space.

**Why it matters:** The original MAP-Elites archive in FleetBFT-QD stores diverse individuals, but *parent selection* still defaulted to tournament selection. FleetDiversity upgrades every selection decision with mathematical rigor.

## Research Foundations

| Strategy | Paper | What It Does | Time Complexity |
|----------|-------|-------------|-----------------|
| **DPP** (default) | Kulesza & Taskar (2012), Chen et al (2018) | Probabilistic repulsion — diverse yet relevant | O(k·n·d + n·k²) |
| **MMR** | Carbonell & Goldstein (1998) | Relevance minus similarity penalty | O(k·n·d) |
| **MSD** | Borodin, Lee & Ye (2012) | Maximum variety, may sacrifice relevance | O(k·n·d) |
| **COVER** | Puthiya Parambath et al (2016) | Facility-location topic coverage | O(k·n²) |
| **SSD** | Huang et al (2021) | Sequence-aware novelty for feeds | O(k·n·d) |

**DPP is the benchmark winner:** +26-44% ILAD, +54-86% ILMD, +1.8-3.1% relevance over baselines. Use `diversity=0.5-0.8`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  FleetBreederConsensus                      │
│                     (BFT-gated breeding)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  QDArchive   │      │FleetDiversity│                     │
│  │ (MAP-Elites) │◄────►│  Selector    │                     │
│  │              │      │              │                     │
│  │  stores all  │      │  picks       │                     │
│  │  elites      │      │  diverse     │                     │
│  └──────────────┘      │  parents     │                     │
│                        └──────────────┘                     │
│                              │                              │
│                              ▼                              │
│                        ┌──────────────┐                     │
│                        │  pyversity   │                     │
│                        │  (DPP/MMR)   │                     │
│                        └──────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Parent Selection

```python
from swarm.fleet_diversity import (
    FleetDiversitySelector,
    DiversityStrategy,
    PopulationItem,
)
import numpy as np

selector = FleetDiversitySelector(
    strategy=DiversityStrategy.DPP,  # probabilistic repulsion — default
    diversity=0.6,  # 0.0 = pure fitness, 1.0 = pure diversity
    default_k=10,
)

# Create a population of agents
population = [
    PopulationItem(
        id=f"agent_{i}",
        embedding=np.random.randn(256),
        fitness=np.random.rand(),
    )
    for i in range(100)
]

# Select 10 diverse parents
parents = selector.select_parents(population, k=10)
print(
    f"Selected {len(parents)} parents with mean fitness {np.mean([p.fitness for p in parents]):.2f}"
)
```

### Strategy Comparison

```python
# DPP: best overall (default)
dpp_parents = selector.select_parents(population, k=10, strategy=DiversityStrategy.DPP)

# MMR: relevance-first with diversity penalty
mmr_parents = selector.select_parents(population, k=10, strategy=DiversityStrategy.MMR)

# COVER: topic coverage for archive sync
cover_parents = selector.select_parents(
    population, k=10, strategy=DiversityStrategy.COVER
)

# MSD: maximum variety (may sacrifice some relevance)
msd_parents = selector.select_parents(population, k=10, strategy=DiversityStrategy.MSD)
```

### QDArchive Elite Diversification

```python
from swarm.fleet_bft_qd import FleetBreederConsensus

fbc = FleetBreederConsensus(
    node_id="n0",
    all_nodes=["n0", "n1", "n2", "n3"],
    secret_key="fleet-secret",
)

# Wire diversity selector to archive
selector = FleetDiversitySelector.from_breeder_consensus(fbc)

# Select diverse elites from archive for cross-node breeding pool
diverse_elites = selector.wire_to_qd_archive(fbc.archive, k=8)
print(f"Selected {len(diverse_elites)} diverse elites for sync")
```

### Nearest-Neighbor Re-ranking

```python
# Find nearest neighbors, then diversify
query_embedding = np.random.randn(256)
candidates = population[:50]  # some candidate pool

# Re-rank with DPP to avoid redundant results
top_diverse = selector.rerank_nearest_neighbors(
    candidates, query_embedding, k=10, diversity=0.7
)
```

### Diversity Metrics

```python
stats = selector.compute_diversity_stats(population, selected_indices=[0, 5, 10])
print(f"ILAD (intra-list avg distance): {stats.ilad:.3f}")
print(f"ILMD (intra-list min distance): {stats.ilmd:.3f}")
print(f"Mean pairwise distance: {stats.mean_pairwise_distance:.3f}")
```

## API Reference

### FleetDiversitySelector

| Method | Description |
|--------|-------------|
| `select_parents(pop, k, strategy, diversity)` | Select `k` diverse parents from population |
| `diversify_archive_elites(elites, embeddings, k, strategy)` | Diverse subset of archive elites |
| `rerank_nearest_neighbors(candidates, query, k, diversity)` | Diversity-aware NN re-ranking |
| `compute_diversity_stats(pop, selected_indices)` | ILAD, ILMD, pairwise distances |
| `wire_to_qd_archive(archive, k, strategy)` | Select diverse elites from QDArchive |
| `from_breeder_consensus(consensus, ...)` | Factory wired to FleetBreederConsensus |
| `get_history()` / `clear_history()` | Selection audit trail |

### PopulationItem

```python
@dataclass
class PopulationItem:
    id: str
    embedding: np.ndarray
    fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### DiversityStats

```python
@dataclass
class DiversityStats:
    n_items: int
    mean_fitness: float
    mean_pairwise_distance: float
    ilad: float  # Intra-List Average Distance
    ilmd: float  # Intra-List Minimum Distance
    selected_indices: List[int]
    selected_fitness_mean: float
    selected_diversity_mean: float
```

### DiversityStrategy

- `DPP` — Determinantal Point Process (probabilistic repulsion) **[default]**
- `MMR` — Maximal Marginal Relevance
- `MSD` — Max Sum of Distances
- `COVER` — Facility-Location coverage
- `SSD` — Sliding Spectrum Decomposition (sequence-aware)

## Fallback Mode

If `pyversity` is not installed, FleetDiversity automatically falls back to pure-NumPy implementations of all strategies. The fallback is slightly slower but mathematically identical. Install pyversity for speed:

```bash
pip install pyversity
```

## Integration with Fleet Infrastructure

### FleetBreederConsensus

Diversity selection happens **before** BFT proposal:
1. `FleetDiversitySelector.select_parents()` → diverse parent set
2. `FleetBreederConsensus.propose_breeding_batch()` → BFT consensus
3. All honest nodes execute the same diverse batch

### QDArchive

`wire_to_qd_archive()` uses grid indices as embedding proxies for diversification. When cross-node breeding pools sync, COVER strategy ensures topic coverage across the behavior space.

### FleetVectorIndex

`rerank_nearest_neighbors()` integrates with vector index lookups to avoid returning near-duplicate agents.

## Test Summary

43 tests covering:
- All 5 strategies: DPP, MMR, MSD, COVER, SSD (5 test classes)
- Archive integration: QDArchive elite diversification (3 tests)
- NN re-ranking: query-based diversity (3 tests)
- Diversity stats: ILAD, ILMD, pairwise distances (4 tests)
- Fallback: pure-NumPy without pyversity (5 tests)
- Edge cases: empty, single item, k > n, zero embeddings, negative fitness (6 tests)
- History tracking: audit trail (3 tests)
- Parameter override: per-call strategy/diversity/k (3 tests)
- Breeder integration: BFT proposal pattern, archive update (3 tests)
- Scale: 100 candidates sub-second, 256-dim embeddings, 3D archive (3 tests)

Run: `pytest tests/test_fleet_diversity.py -v`

## References

- **pyversity**: https://github.com/Pringled/pyversity (MIT, 491⭐)
- **DPP**: Kulesza, A. & Taskar, B. (2012). "Determinantal Point Processes for Machine Learning"
- **DPP fast greedy**: Chen, L. et al (2018). "Fast greedy MAP inference for determinantal point process"
- **MMR**: Carbonell, J. & Goldstein, J. (1998). "The use of MMR, diversity-based reranking"
- **MSD**: Borodin, A. et al (2012). "Max-sum diversification, monotone submodular functions"
- **COVER**: Puthiya Parambath, S.A. et al (2016). "A coverage-based approach to recommendation diversity"
- **SSD**: Huang, Y. et al (2021). "Sliding Spectrum Decomposition for Diversified Recommendation"
