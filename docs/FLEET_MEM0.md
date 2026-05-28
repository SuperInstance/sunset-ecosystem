# FleetMem0 — Semantic Memory Adapter

## Overview

FleetMem0 wraps **[mem0ai/mem0](https://github.com/mem0ai/mem0)** (v2 algorithm) to provide per-agent semantic memory with vector + BM25 + entity retrieval.

**What it does:** Replaces file-based memory (`memory/YYYY-MM-DD.md` + `MEMORY.md`) with a searchable, semantic memory layer. Agents can add memories, search them by meaning, and retrieve relevant context before making decisions.

**Why it matters:** When the fleet scales to hundreds of agents, flat file memory becomes unmanageable. FleetMem0 gives us:
- **Semantic search** — "What was I working on 3 days ago?" becomes answerable
- **Entity linking** — memories connected by entities, not just timestamps
- **Temporal reasoning** — time-aware retrieval for current state vs past events
- **Cross-agent memory sharing** — via mesh gossip of memory graphs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SenseDecideAct                            │
│                      (SDA Loop)                              │
├─────────────────────────────────────────────────────────────┤
│                     FleetMem0Memory                          │
│                          │                                   │
│            ┌─────────────┴─────────────┐                   │
│            │                           │                   │
│      mem0.Memory                 Fallback Python            │
│      (vector+BM25+entity)        keyword search             │
│            │                           │                   │
│            ▼                           ▼                   │
│   ┌────────────────┐          ┌────────────────┐            │
│   │  Vector Store  │          │  In-memory list │            │
│   │  (Qdrant/Chroma│          │  of dicts        │            │
│   │  /FAISS/...)   │          │                  │            │
│   └────────────────┘          └────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Memory

```python
from fleet.fleet_mem0 import FleetMem0Memory, FleetMem0Config

mem = FleetMem0Memory(
    FleetMem0Config(
        agent_id="kimi1",
        vector_store="qdrant",
        llm_provider="ollama",
        embedding_model="nomic-embed-text",
    )
)

# Add memories
mem.add("The sunset ecosystem now has 19 modules and 484 tests.")
mem.add("FLUX VM uses 60 opcodes but Python uses zero.", metadata={"topic": "FLUX"})

# Search
results = mem.search("How many modules do we have?", k=3)
for r in results:
    print(f"{r.score:.2f}: {r.content}")
```

### Per-Agent Memory Profiles

```python
from fleet.fleet_mem0 import FleetMem0Memory
from swarm.agent_identity import AgentIdentity

identity = AgentIdentity(agent_id="Oracle1", role="auditor")
mem = FleetMem0Memory.from_agent_identity(identity)

mem.add("Oracle1 prefers Dieter Rams meets Moebius aesthetic")
mem.add("Oracle1 audited 12 modules last night")
```

### Integration with SenseDecideAct

```python
from fleet.sense_decide_act import SDALoop
from fleet.fleet_mem0 import FleetMem0Memory

loop = SDALoop()
mem = FleetMem0Memory(FleetMem0Config(agent_id="kimi1"))

# Sense: query relevant memories before deciding
context = mem.search("What should I work on next?", k=5)
memory_context = "\n".join(r.content for r in context)

# Decide: use memory context in the decision
action = loop.decide("Next task", context=memory_context)
```

## API Reference

### FleetMem0Memory

| Method | Description |
|--------|-------------|
| `add(content, metadata)` | Store a memory. Returns memory_id or None. |
| `search(query, k)` | Semantic search for top-k memories. |
| `get_all()` | Retrieve all stored memories. |
| `delete(memory_id)` | Remove a memory by ID. |
| `history(memory_id)` | Get edit history of a memory. |
| `from_agent_identity(identity)` | Create memory for an agent. |
| `to_dict()` | Serialize status. |

### FleetMem0Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vector_store` | `"qdrant"` | Vector store backend |
| `vector_store_path` | `"~/.openclaw/mem0_vectors"` | Storage path |
| `llm_provider` | `"ollama"` | LLM for extraction |
| `embedding_provider` | `"ollama"` | Embedding model provider |
| `embedding_model` | `"nomic-embed-text"` | Embedding model name |
| `user_id` | `"fleet_default"` | User scope |
| `agent_id` | `"kimi1"` | Agent scope |
| `version` | `"v2"` | Mem0 algorithm version |

### FleetMemoryEntry

| Field | Description |
|-------|-------------|
| `content` | Memory text |
| `memory_id` | Unique ID |
| `metadata` | Arbitrary dict |
| `score` | Relevance score from search |

## Fallback Mode

When `mem0ai` is not installed or initialization fails, FleetMem0 falls back to pure-Python keyword search:
- Splits query into words
- Counts occurrences in stored memories
- Returns top-k by count

The fallback is stateless (no persistence) and has no semantic understanding.

## Vector Store Options

| Store | Best For | Setup |
|-------|----------|-------|
| **Qdrant** | Production, scale | `pip install qdrant-client` |
| **Chroma** | Local dev, simplicity | `pip install chromadb` |
| **FAISS** | Research, speed | `pip install faiss-cpu` |
| **PGVector** | Existing Postgres | `pip install pgvector` |
| **Weaviate** | Cloud-native | `pip install weaviate-client` |

Default is Qdrant for file-based local storage.

## Integration Points

| System | Integration |
|--------|-------------|
| **AgentIdentity** | `from_agent_identity()` → per-agent memory profile |
| **SenseDecideAct** | `sense()` queries Mem0 for relevant context |
| **MeshVectorGossip** | Memory graphs shared across nodes via CRDT merge |
| **FleetKorok** | Mem0 memories indexed for hybrid text search |
| **Heartbeat** | Automatic memory consolidation during idle checks |

## Test Summary

15 tests covering:
- Initialization (default + custom config)
- add: stores content with metadata
- search: empty index, fallback keyword match, no match
- get_all: retrieves all memories
- delete: handles nonexistent IDs
- history: returns list
- from_agent_identity: creates per-agent memory
- to_dict: serialization
- Config: defaults and custom parameters

Run: `pytest tests/test_fleet_mem0.py -v`

## Fork Status

**Upstream:** https://github.com/mem0ai/mem0
**Our fork:** `/root/.openclaw/workspace/forks/mem0-fork`
**Status:** Fork created, not yet modified. Adapter pattern used instead.
**Plan:** If we need deep changes (local model extraction, fleet-specific entity types), we'll modify the fork.

## References

- **mem0**: https://github.com/mem0ai/mem0
- **Mem0 v2 paper**: April 2026 — single-pass ADD-only extraction
- **Benchmarks**: LoCoMo 91.6, LongMemEval 94.8, BEAM 64.1 (1M tokens)
- **Qdrant**: https://qdrant.tech
- **Ollama**: https://ollama.com
