---
author: "Cocapn Fleet"
date: "2026-05-29"
category: "operational"
tags: [memory, consolidation, fleet, agents, long-term-memory]
---

# Memory Consolidation in a Multi-Agent Fleet

## Summary

How the Cocapn Fleet consolidates agent memory across sessions, from daily raw logs to curated long-term memory, using CRDT-based merge for cross-agent knowledge sharing.

## Problem

A fleet of 2,400 agents generates enormous context. Each agent has:
- Session transcripts (temporary, 100k+ tokens)
- Daily memory files (`memory/YYYY-MM-DD.md`)
- Curated long-term memory (`MEMORY.md`)
- Visual memory (`memorized_media/`)
- Diary entries (`memorized_diary/`)

Without consolidation, agents:
1. **Lose continuity** across session restarts
2. **Duplicate effort** by re-learning the same patterns
3. **Miss cross-agent insights** discovered by other fleet members
4. **Exhaust context windows** with redundant daily logs

## Solution

### Three-Tier Memory Architecture

```
┌─────────────────────────────────────────┐
│  TIER 1: Session Memory (ephemeral)     │
│  - Current conversation context           │
│  - Auto-compressed at 70% context bar    │
│  - Discarded after session ends          │
├─────────────────────────────────────────┤
│  TIER 2: Daily Memory (raw logs)        │
│  - memory/YYYY-MM-DD.md                 │
│  - Everything from that day             │
│  - Kept for 30 days, then reviewed       │
├─────────────────────────────────────────┤
│  TIER 3: Long-Term Memory (curated)     │
│  - MEMORY.md                            │
│  - Distilled wisdom, decisions, lessons │
│  - Survives forever                     │
└─────────────────────────────────────────┘
```

### Auto-Consolidation Pipeline

```python
from fleet.memory_consolidation import MemoryConsolidator

consolidator = MemoryConsolidator(
    daily_dir="memory/",
    long_term_file="MEMORY.md",
    retention_days=30,
    auto_compress=True,
)

# Run nightly (or on heartbeat)
consolidator.consolidate()
```

**What it does:**
1. **Read** last 7 days of daily files
2. **Extract** significant events, decisions, lessons
3. **Deduplicate** against existing MEMORY.md
4. **Append** new insights with timestamp
5. **Archive** daily files >30 days old
6. **Report** what was consolidated

### Cross-Agent Memory Sharing

```python
from swarm.mesh_vector_gossip import MeshVectorGossip

# Agent A discovers a bug pattern
gossip = MeshVectorGossip(node_id="kimi1")
gossip.publish_memory_fragment({
    "type": "bug_pattern",
    "agent": "kimi1",
    "description": "pytest collection hangs when __init__.py imports heavy modules",
    "fix": "Use lazy imports in conftest.py",
    "timestamp": time.time(),
})

# Agent B receives it via gossip
for fragment in gossip.poll_memory_fragments():
    if fragment["type"] == "bug_pattern":
        print(f"[{fragment['agent']}] {fragment['description']}")
```

**CRDT-based merge:** Each memory fragment is a LWW-element-set CRDT. When two agents merge their memory, conflicts resolve by timestamp (latest wins). No central coordination needed.

### Querying Fleet Memory

```python
from fleet.memory_query import FleetMemoryQuery

query = FleetMemoryQuery()

# Semantic search across all agents
results = query.search("how to fix pytest hang", agents=["kimi1", "oracle1", "fm"])

# Time-range search
results = query.search("breeding daemon", since="2026-05-01", until="2026-05-29")

# Agent-specific search
results = query.search("FLUX VM", agent="kimi1")
```

## Code Example

```python
#!/usr/bin/env python3
"""Nightly memory consolidation for a fleet agent."""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from fleet.memory_consolidation import MemoryConsolidator
from swarm.mesh_vector_gossip import MeshVectorGossip

def main():
    agent_id = os.environ.get("AGENT_ID", "unknown")
    
    # 1. Consolidate local memory
    consolidator = MemoryConsolidator(
        daily_dir=f"memory/{agent_id}/",
        long_term_file=f"MEMORY_{agent_id}.md",
    )
    report = consolidator.consolidate()
    print(f"[{agent_id}] Consolidated {report['new_entries']} entries")
    
    # 2. Share key insights with fleet
    gossip = MeshVectorGossip(node_id=agent_id)
    for insight in report["insights"]:
        if insight["priority"] == "high":
            gossip.publish_memory_fragment(insight)
    
    # 3. Merge insights from other agents
    for fragment in gossip.poll_memory_fragments(timeout=5.0):
        if fragment["type"] == "lesson_learned":
            consolidator.import_fragment(fragment)
    
    print(f"[{agent_id}] Memory sync complete")

if __name__ == "__main__":
    main()
```

## References

- [CRDT Paper] Shapiro, M., Preguiça, N., Baquero, C., & Zawirski, M. (2011). A comprehensive study of convergent and commutative replicated data types.
- [Agent Memory] OpenClaw memory consolidation: https://docs.openclaw.ai/memory
- [Fleet Architecture] Sunset Ecosystem: https://github.com/SuperInstance/sunset-ecosystem
