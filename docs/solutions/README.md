---
author: "Cocapn Fleet"
date: "2026-05-29"
category: "index"
tags: [solutions, documentation, fleet, patterns]
---

# Fleet Solution Documents

Structured solution documents capturing the Cocapn Fleet's operational patterns, design decisions, and engineering practices.

## Index

| Document | Category | Summary |
|----------|----------|---------|
| [Integration Testing](testing/integration_testing.md) | Testing | How to test module integration in a multi-agent fleet with parallel test execution, dependency injection, and CI/CD integration. |
| [Test-Driven Architecture](architecture/test_driven_architecture.md) | Architecture | Designing fleet modules from test specifications first, using the "spec → test → impl → integrate" cycle. |
| [Debugging Strategies](debugging/debugging_strategies.md) | Debugging | How to debug distributed fleet systems: pytest hangs, subagent timeouts, gateway overload, and context truncation. |
| [Memory Consolidation](memory_consolidation.md) | Operational | How the fleet consolidates agent memory across sessions using three-tier architecture and CRDT-based cross-agent sharing. |
| [Subagent Orchestration](subagent_orchestration.md) | Operational | How to orchestrate 4+ generations of subagents with 87.5% success rate using the "spawn → monitor → rescue → merge" pattern. |
| [Operational Trap](operational_trap.md) | Operational | How the OperationalTrap module detects fleet health degradation and triggers circuit breakers, cooldowns, and alerts. |

## Format

All solution docs follow a standard structure:

```yaml
---
author: "Cocapn Fleet"
date: "YYYY-MM-DD"
category: "testing|architecture|debugging|operational"
tags: [relevant, tags]
---

# Title

## Summary
One paragraph describing the problem and solution.

## Problem
What goes wrong and why it matters.

## Solution
How we solve it, with diagrams and examples.

## Code Example
Runnable Python code demonstrating the pattern.

## References
Papers, docs, and internal files for further reading.
```

## Contributing

When you discover a new fleet pattern:
1. Create a new file under `docs/solutions/`
2. Use the standard YAML frontmatter + structure
3. Include a runnable code example
4. Update this index
5. Commit to `main`
