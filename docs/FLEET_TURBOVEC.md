# FleetTurboVec — Rust-Accelerated Vector Index

## Overview

FleetTurboVec wraps **[RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)** (TurboQuant algorithm, Rust + Python bindings) to provide a high-performance vector index backend for FleetVectorIndex.

**What it does:** Replaces the pure-NumPy brute-force search in FleetVectorIndex with a Rust-backed quantized index that uses **16× less memory** and searches **faster than FAISS** on ARM.

**Why it matters:** When the fleet scales to 10,000+ agents, the pure-NumPy vector table becomes a bottleneck. FleetTurboVec brings "get to the metal" performance without changing the fleet's API.

## Key Advantages

| Metric | Pure-NumPy | FleetTurboVec (4-bit) | FleetTurboVec (2-bit) |
|--------|-----------|----------------------|----------------------|
| Memory (10M docs, 256-dim) | 31 GB | 7.8 GB | 3.9 GB |
| Training step | Required | **None** | **None** |
| Online ingest | Slow | **Fast** | **Fast** |
| ARM performance | Baseline | **Faster than FAISS** | **Faster than FAISS** |
| Filtered search | Post-filter | **Query-time mask/allowlist** | **Query-time mask/allowlist** |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  FleetVectorIndex                           │
│              (cross-node breeding pools)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────────┐                 │
│  │  Pure-NumPy  │      │  FleetTurboVec   │                 │
│  │  Fallback    │  OR  │  (Rust Backend)  │                 │
│  │              │      │                  │                 │
│  │  brute-force │      │  TurboQuant      │                 │
│  │  O(n·d)      │      │  O(n·d·bit/32)   │                 │
│  └──────────────┘      └──────────────────┘                 │
│                              │                              │
│                              ▼                              │
│                        ┌──────────────┐                     │
│                        │  turbovec    │                     │
│                        │  (PyO3)      │                     │
│                        │              │                     │
│                        │  IdMapIndex  │                     │
│                        │  uint64 IDs  │                     │
│                        │  .tvim files │                     │
│                        └──────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Search

```python
from swarm.fleet_turbovec import FleetTurboVecIndex, TurboVecConfig, TurboVecEntry
import numpy as np

# Create index with 4-bit quantization (2-bit and 8-bit also available)
index = FleetTurboVecIndex(
    TurboVecConfig(dim=256, bit_width=4, diversity_rerank=True)
)

# Ingest agent embeddings
entries = [
    TurboVecEntry(
        agent_id=f"Oracle1::agent_{i}",
        vector=np.random.randn(256).astype(np.float32),
        fitness=0.85,
        generation=3,
        node_id="Oracle1",
    )
    for i in range(1000)
]
index.add_entries(entries)

# Warm up caches (one-time init cost)
index.prepare()

# Search
query = np.random.randn(256).astype(np.float32)
results = index.search(query, k=10)
for r in results:
    print(f"{r.rank}: {r.agent_id} (score={r.score:.3f})")
```

### Filtered Search

```python
# Only search agents from node "Oracle1" with fitness > 0.8
def my_filter(entry: TurboVecEntry) -> bool:
    return entry.node_id == "Oracle1" and entry.fitness > 0.8

results = index.search(query, k=5, filter_fn=my_filter)
```

### Diversity Re-ranking

```python
# After NN search, apply DPP to avoid redundant results
index = FleetTurboVecIndex(
    TurboVecConfig(
        dim=256,
        bit_width=4,
        diversity_rerank=True,
        diversity_k=20,      # Rerank top-20 NN candidates
        diversity_strategy="dpp",
        diversity_lambda=0.7,
    )
)

results = index.search(query, k=5, diversity_rerank=True)
```

### Integration with FleetVectorIndex

```python
from swarm.mesh_vector_tables import FleetVectorIndex
from swarm.fleet_turbovec import FleetTurboVecIndex, TurboVecConfig

# Existing fleet index
fvi = FleetVectorIndex(node_id="n0", identity=agent_identity)
# ... populate with entries ...

# Migrate to TurboVec backend
tv = FleetTurboVecIndex.from_fleet_vector_index(
    fvi,
    TurboVecConfig(dim=256, bit_width=4)
)

# Export back to fleet entries when needed
fleet_entries = tv.to_fleet_entries()
```

### Save and Load

```python
# Save snapshot
index.save("/path/to/checkpoint/")

# Load snapshot
loaded = FleetTurboVecIndex.load("/path/to/checkpoint/")
print(f"Restored {len(loaded)} entries")
```

## API Reference

### FleetTurboVecIndex

| Method | Description |
|--------|-------------|
| `add_entries(entries)` | Batch ingest TurboVecEntry objects |
| `search(query, k, filter_fn, diversity_rerank)` | NN search with optional filtering and DPP rerank |
| `remove(agent_id)` | O(1) deletion by agent ID |
| `save(path)` | Serialize to `.tvim` + `metadata.json` |
| `load(path)` | Restore from serialized snapshot |
| `prepare()` | Warm up SIMD caches |
| `from_fleet_vector_index(fvi, config)` | Migrate from FleetVectorIndex |
| `to_fleet_entries()` | Export as VectorTableEntry list |

### TurboVecConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | `None` | Vector dimension (None = lazy from first add) |
| `bit_width` | `4` | Quantization: 2, 4, or 8 bits per dimension |
| `diversity_rerank` | `True` | Apply DPP/MMR after NN search |
| `diversity_k` | `10` | Number of coarse candidates for reranking |
| `diversity_strategy` | `"dpp"` | `dpp`, `mmr`, `msd`, `cover`, `ssd` |
| `diversity_lambda` | `0.5` | Diversity-relevance tradeoff |

### TurboVecEntry

```python
@dataclass
class TurboVecEntry:
    agent_id: str
    vector: np.ndarray        # float32
    fitness: float = 0.0
    generation: int = 0
    node_id: str = ""
    metadata: dict = field(default_factory=dict)
```

## Building turbovec

FleetTurboVec requires the `turbovec` Python package, which contains Rust extensions. Build from source:

```bash
git clone https://github.com/RyanCodrai/turbovec.git
cd turbovec/turbovec-python
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

**Memory note:** The Rust build is memory-intensive. If compilation is killed (OOM), use single-job compilation:

```bash
CARGO_BUILD_JOBS=1 maturin develop --release
```

Once built, `import turbovec` should work in any Python environment.

## Fallback Mode

If `turbovec` is not installed, FleetTurboVec automatically falls back to pure-NumPy brute-force search. The fallback is:
- Slower for large populations (>1000 agents)
- Identical API
- No `.tvim` serialization (uses `.npy` + `.json` instead)

## Integration Points

| System | Integration |
|--------|-------------|
| **FleetVectorIndex** | `from_fleet_vector_index()` / `to_fleet_entries()` |
| **FleetDiversitySelector** | `search(..., diversity_rerank=True)` calls DPP/MMR |
| **MeshVectorGossip** | `.tvim` snapshots for cross-node sync |
| **BreederDaemonV2** | Fast NN lookup for `get_breedable_pool()` |

## Test Summary

28 tests covering:
- Backend initialization (turbovec vs numpy fallback)
- Add: batch ingest, lazy dim, ID mapping
- Search: basic NN, filtered search, diversity rerank, k > n, empty index
- Remove: deletion, fallback rebuild
- Save/Load: metadata round-trip
- Prepare: cache warm-up
- FleetVectorIndex integration: migration to/from
- Edge cases: single entry, zero vectors, 256-dim, 1000-entry batch
- Config: default and custom parameters

Run: `pytest tests/test_fleet_turbovec.py -v`

## References

- **turbovec**: https://github.com/RyanCodrai/turbovec
- **TurboQuant**: Google Research, "Data-Oblivious Quantization for Vector Search"
- **PyO3**: https://pyo3.rs — Rust/Python bindings
