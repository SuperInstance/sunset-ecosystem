# FleetKorok — Hybrid Text Search Adapter

## Overview

FleetKorok wraps **[Pringled/korok](https://github.com/Pringled/korok)** (dense vector + BM25 sparse + cross-encoder reranking) to provide text-based tile and document retrieval for the fleet.

**What it does:** Hybrid search over agent tiles, documentation, and memos — combining semantic meaning (dense embeddings) with keyword matching (BM25) and optional cross-encoder reranking for precision.

**Why it matters:** When the fleet accumulates thousands of tiles across dozens of rooms, finding the right context for a breeding decision requires more than brute-force vector search. FleetKorok adds **text intelligence** to the fleet's retrieval stack.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FleetKorokIndex                          │
│                    (hybrid search)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Dense      │    │   Sparse     │    │  Reranker   │   │
│  │  (model2vec) │    │  (BM25)      │    │ (CrossEnc)  │   │
│  │              │    │              │    │             │   │
│  │  semantic    │    │  keyword     │    │  precision  │   │
│  │  meaning     │    │  exact match │    │  refinement │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             ▼                               │
│                    fused score = α·dense + (1-α)·sparse     │
│                             │                               │
│                             ▼                               │
│                    FleetKorokResult[]                        │
│                    (doc_id, text, score, metadata)          │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Search

```python
from fleet.fleet_korok import FleetKorokIndex, FleetKorokConfig, FleetKorokEntry

index = FleetKorokIndex(
    FleetKorokConfig(
        alpha=0.6,           # 60% dense, 40% sparse
        use_bm25=True,
        use_dense=True,
        use_reranker=False,
    )
)

entries = [
    FleetKorokEntry(
        doc_id="tile_42",
        text="Breeding daemon spawned 3 offspring with DPP diversity",
        metadata={"room": "breeding", "generation": 3},
    ),
    FleetKorokEntry(
        doc_id="tile_43",
        text="MetronomeBridge synchronized 12 nodes with 500ms drift",
        metadata={"room": "sync", "generation": 3},
    ),
]
index.add_entries(entries)
index.build()

results = index.search("diversity breeding", k=5)
for r in results:
    print(f"{r.doc_id}: {r.text} (score={r.score:.3f})")
```

### From Tile List

```python
tiles = [
    {"tile_id": "t1", "text": "Sunset ecosystem architecture overview"},
    {"tile_id": "t2", "text": "FLUX VM opcode alignment audit"},
]
index = FleetKorokIndex.from_tile_list(tiles)
results = index.search("FLUX audit", k=2)
```

### With Dense Encoder

```python
from model2vec import StaticModel

encoder = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
index = FleetKorokIndex(
    FleetKorokConfig(
        encoder=encoder,
        use_dense=True,
        use_bm25=True,
        alpha=0.7,
    )
)
```

### With Cross-Encoder Reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
index = FleetKorokIndex(
    FleetKorokConfig(
        use_dense=True,
        use_bm25=True,
        use_reranker=True,
        reranker=reranker,
        k_reranker=30,  # candidates for reranking
    )
)
```

## API Reference

### FleetKorokIndex

| Method | Description |
|--------|-------------|
| `add_entries(entries)` | Add FleetKorokEntry documents |
| `build()` | Build the hybrid index |
| `search(query, k)` | Search for top-k results |
| `remove(doc_id)` | Delete a document |
| `clear()` | Remove all documents |
| `get_entry(doc_id)` | Retrieve a specific entry |
| `from_tile_list(tiles, text_extractor)` | Build index from tile dicts |
| `to_dict()` | Serialize metadata |

### FleetKorokConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | Dense/sparse balance (0=sparse, 1=dense) |
| `k_reranker` | `30` | Candidates for cross-encoder reranking |
| `use_bm25` | `True` | Enable sparse keyword search |
| `use_dense` | `True` | Enable dense vector search |
| `use_reranker` | `False` | Enable cross-encoder reranking |
| `encoder` | `None` | model2vec StaticModel or similar |
| `reranker` | `None` | sentence-transformers CrossEncoder |
| `distance_metric` | `"cosine"` | Distance metric for dense search |
| `stopwords` | `"en"` | Stopwords for BM25 tokenization |

### FleetKorokEntry

| Field | Description |
|-------|-------------|
| `doc_id` | Unique document identifier |
| `text` | Searchable text content |
| `metadata` | Arbitrary dict (room, agent_id, generation, etc.) |
| `vector` | Optional pre-computed dense vector |

### FleetKorokResult

| Field | Description |
|-------|-------------|
| `doc_id` | Matched document ID |
| `text` | Full text of matched document |
| `score` | Relevance score (higher = better) |
| `metadata` | Passthrough from entry |
| `dense_score` | Dense component score (if available) |
| `sparse_score` | Sparse component score (if available) |

## Fallback Mode

When `korok` is not installed, FleetKorok falls back to pure-Python keyword matching:
- Splits query into words
- Counts word occurrences in each document
- Returns top-k by count

The fallback is:
- Fast for small corpora (<1000 docs)
- No semantic understanding (keyword only)
- Zero dependencies

## Integration Points

| System | Integration |
|--------|-------------|
| **Mem0** | FleetKorok indexes Mem0 memories for text retrieval |
| **Tile Lifecycle** | `from_tile_list()` ingests active tiles for search |
| **SenseDecideAct** | `sense()` phase queries FleetKorok for relevant context |
| **FleetVectorIndex** | Dense vectors from FleetKorok feed into mesh breeding pools |

## Test Summary

17 tests covering:
- Backend detection (korok vs fallback)
- add_entries + build
- Search: keyword match, no match, empty index, result objects
- Remove, clear, len
- from_tile_list with custom extractor
- to_dict serialization
- Config: defaults and custom parameters

Run: `pytest tests/test_fleet_korok.py -v`

## Fork Status

**Upstream:** https://github.com/Pringled/korok
**Our fork:** Not yet created (Python adapter approach used)
**Reason:** korok is a pip-installable library; wrapping it with fleet-specific doc_id mapping and metadata passthrough is cleaner than forking.

## References

- **korok**: https://github.com/Pringled/korok
- **model2vec**: https://github.com/MinishLab/model2vec
- **bm25s**: https://github.com/xhluca/bm25s
- **sentence-transformers**: https://www.sbert.net
