---
author: "Cocapn Fleet"
date: "2026-05-29"
category: "operational"
tags: [subagents, orchestration, fleet, parallel, context-management]
---

# Subagent Orchestration at Scale

## Summary

How to orchestrate 4+ generations of subagents with 87.5% success rate, using the "spawn → monitor → rescue → merge" pattern, gateway overload detection, and the baton handoff protocol.

## Problem

Building a fleet of 2,400 agents requires massive parallelization. Single-agent context limits (~181k tokens) mean:
1. **One agent cannot hold the entire codebase**
2. **Sequential work is too slow** (20 modules × 30 min = 10 hours)
3. **Gateway timeouts kill subagents** at high load
4. **Context loss** when agents exceed 70% compression

## Solution

### The "Spawn → Monitor → Rescue → Merge" Pattern

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   SPAWN  │───▶│  MONITOR │───▶│  RESCUE  │───▶│  MERGE   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │
     ▼               ▼               ▼               ▼
 4 agents         poll logs        direct work      git merge
 dispatched     every 30s        fallback         + conflict
                timeout?                          resolution
                → rescue
```

### When to Use Subagents vs Direct Work

| Scenario | Subagent | Direct Work | Why |
|----------|----------|-------------|-----|
| 1 module, <200 lines | ❌ | ✅ | Setup cost > work |
| 1 module, 500+ lines | ✅ | ❌ | Parallel thinking helps |
| 4+ independent modules | ✅ | ✅ | Subagents + direct in tmux |
| Gateway overloaded | ❌ | ✅ | Fallback to tmux sessions |
| Cross-module integration | ✅ | ❌ | Needs isolated context |
| Urgent hotfix (<5 min) | ❌ | ✅ | No spawn overhead |

### The Baton Handoff Protocol

```python
from fleet.context_baton import BatonHandoff

# When agent reaches 70% context, trigger handoff
baton = BatonHandoff(
    from_agent="kimi1_gen1",
    to_agent="kimi1_gen2",
    checkpoint_file=".fleet/baton_checkpoint.json",
)

baton.save_checkpoint({
    "task": "Build MeshVectorTable",
    "progress": "fields defined, tests passing",
    "next_step": "Add CRDT merge logic",
    "files_modified": ["swarm/mesh_vector_tables.py"],
    "tests_status": "12/12 green",
})

# Gen2 receives baton and continues
state = baton.load_checkpoint()
print(f"Continuing: {state['next_step']}")
```

**Key insight:** The baton is not the full context — it's a *compressed intent*. The receiving agent reads the spec, reads the code, and reconstructs context locally. This avoids 70% truncation.

### Gateway Overload Detection

```python
from fleet.gateway_monitor import GatewayMonitor

monitor = GatewayMonitor()

if monitor.overload_detected():
    # Fallback: use tmux sessions instead of subagents
    strategy = "direct_tmux"
else:
    # Normal: spawn subagents in parallel
    strategy = "subagent_parallel"

# Pacing: wait 20 min after 2 consecutive timeouts
monitor.record_timeout()
if monitor.consecutive_timeouts >= 2:
    monitor.wait(minutes=20)
```

### Subagent Task Specification Template

```python
TASK_SPEC = """
**Mission:** [One sentence, what to build]

**Context:** [Why this matters, what system it integrates with]

**Requirements:**
1. [Concrete requirement]
2. [Concrete requirement]
3. [Integration point]

**Tests:** [File path, test count, what to verify]

**Docs:** [File path, what to document]

**Deliverable:** [Exactly what to return]

**Constraints:**
- Work in /path/to/repo/
- Commit when done
- Work fast (yield after 10 turns max)
- Report blockers immediately
"""
```

## Code Example

```python
#!/usr/bin/env python3
"""Orchestrate 4 parallel subagents with fallback to direct work."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fleet.gateway_monitor import GatewayMonitor
from fleet.tmux_runner import TmuxRunner

def spawn_or_direct(task_name: str, task_spec: str, monitor: GatewayMonitor) -> str:
    """Spawn subagent if gateway healthy, else run directly in tmux."""
    if not monitor.overload_detected():
        try:
            return spawn_subagent(task_name, task_spec)
        except GatewayTimeout:
            monitor.record_timeout()
    
    # Fallback: direct work in tmux session
    runner = TmuxRunner(session_name=task_name)
    runner.send_command(f"cd /workspace/sunset-ecosystem && {task_spec}")
    return runner.capture_output(timeout=900)

def orchestrate_parallel(tasks: dict) -> dict:
    """
    tasks = {
        "arrow-telemetry": "Build swarm/arrow_telemetry.py...",
        "cellular-engine": "Build swarm/cellular_engine.py...",
        "ci-integration": "Build .github/workflows/...",
        "solution-docs": "Write docs/solutions/...",
    }
    """
    monitor = GatewayMonitor()
    results = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(spawn_or_direct, name, spec, monitor): name
            for name, spec in tasks.items()
        }
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result(timeout=1800)
                print(f"✅ {name} complete")
            except Exception as e:
                results[name] = f"FAILED: {e}"
                print(f"❌ {name} failed: {e}")
    
    return results

if __name__ == "__main__":
    tasks = {
        "arrow-telemetry": "python -m pytest tests/test_arrow_telemetry.py -v",
        "cellular-engine": "python -m pytest tests/test_cellular_engine.py -v",
        "ci-integration": "python -m pytest tests/test_review_code_ci.py -v",
        "solution-docs": "echo 'Docs written'",
    }
    results = orchestrate_parallel(tasks)
    print(f"\nResults: {results}")
```

## References

- [Context Management] OpenClaw context limits: https://docs.openclaw.ai/context
- [ThreadPoolExecutor] Python concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
- [Baton Pattern] OpenClaw handoff protocol: https://docs.openclaw.ai/skills/baton
- [Fleet Status] Sunset Ecosystem module inventory: https://github.com/SuperInstance/sunset-ecosystem/STRATEGY.md
