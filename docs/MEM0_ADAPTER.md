# Mem0 Adapter

Cross-agent memory sharing with **CRDT-style merge**, **gossip protocol integration**, and **per-agent profile isolation**.

Inspired by Mem0's cross-session memory layer, rebuilt for fleet-scale agent collaboration.

---

## Quick Start

```python
from fleet.mem0_adapter import Mem0Memory, AgentProfile

memory = Mem0Memory()
profile = memory.get_profile("agent_1")
profile.remember("User prefers dark mode")
print(profile.recall("dark mode"))  # ["User prefers dark mode"]

# Share across agents
memory.share_memory("agent_1", "agent_2", "dark mode preference")

# Build gossip payload
payload = memory.build_sync_payload("agent_1")
# → List of MemoryEntry dicts with timestamps and HMAC signatures
```

---

## Architecture

```
Mem0Memory
├── get_profile(agent_id)  → AgentProfile
│   ├── remember(content, tags=[])
│   ├── recall(query, k=5) → List[str]
│   └── entries  → List[MemoryEntry]
├── share_memory(sender, recipient, content)
├── receive_gossip(payload)  → CRDT merge
├── build_sync_payload(agent_id)
└── attach_to_gossip(mesh_gossip)  → auto-register handler
```

---

## MemoryEntry

Each memory entry is signed and timestamped:

```python
MemoryEntry(
    content: str,
    timestamp: float,
    source: str,
    tags: List[str],
    hmac: str,            # HMAC-SHA256(content + timestamp + source)
)
```

Entries are immutable. Updates create new entries; CRDT merge picks the newer timestamp.

---

## CRDT Merge Rules

1. **New entry**: not in local store → add
2. **Existing entry**: compare timestamps → newer wins
3. **HMAC verification**: invalid signature → reject
4. **Deduplication**: same (content, source) within 60s → skip

---

## AgentProfile

Per-agent isolated memory container:

- `remember(content, tags=[], ttl=3600.0)` — store with optional TTL
- `recall(query, k=5)` — simple substring match (fleet-style; no embeddings)
- `forget(query)` — mark entries as forgotten
- `entries` — all stored entries
- `stats()` — count, tags, oldest, newest

---

## Gossip Integration

```python
from swarm.mesh_vector_gossip import MeshVectorGossip
from fleet.mem0_adapter import Mem0Memory

gossip = MeshVectorGossip()
memory = Mem0Memory()
memory.attach_to_gossip(gossip)

# Now any incoming gossip payload with "mem0_sync" type
# is automatically routed to memory.receive_gossip()
```

The handler:
1. Receives `MeshVectorGossip.Message`
2. Extracts `payload["entries"]`
3. Runs CRDT merge per agent
4. Logs merged / rejected counts

---

## Cross-Agent Sharing

```python
# Direct point-to-point
memory.share_memory("agent_a", "agent_b", "server_host = 147.224.38.131")

# Fleet-wide broadcast
payload = memory.build_sync_payload("agent_a")
gossip.broadcast("mem0_sync", payload, ttl=300)
```

---

## HMAC Signing

Mem0Memory generates a per-instance key via `os.urandom(32)` (or loads from env).
All entries are signed to prevent tampering in transit.

```python
key = os.urandom(32)
sig = hmac.new(key, f"{content}:{ts}:{source}".encode(), hashlib.sha256).hexdigest()
```

---

## Testing

Run: `python3 -m pytest tests/test_mem0_adapter.py -v`

31 tests covering:
- Profile creation and retrieval
- Remember / recall / forget lifecycle
- TTL expiration
- HMAC signing and verification
- CRDT merge (new wins, newer wins, tamper reject)
- Cross-agent sharing
- Gossip handler registration
- Stats and roundtrip

---

## Comparison with Original Mem0

| Feature | Mem0 (original) | Fleet Mem0Adapter |
|--------|-----------------|-------------------|
| Storage | SQLite / Qdrant | In-memory dict |
| Embeddings | OpenAI / custom | None (substring) |
| TTL | Yes | Yes |
| Cross-session | Yes | Yes |
| Cross-agent | No | **Yes (CRDT)** |
| Gossip sync | No | **Yes (mesh)** |
| HMAC signing | No | **Yes** |
