---
title: "Bootstrap Truncation — Context Loss in Agent Sessions"
category: debugging
type: bug-track
date: 2026-05-22
recurrence: "every session"
severity: medium
---

# Bootstrap Truncation — Context Loss in Agent Sessions

## Symptoms

- Every session starts with a warning: `USER.md: 28779 raw -> 18106 injected (~37% removed; max/file)`
- Context loss ranges from 15% to 37% depending on session size
- Truncated memory means agents miss critical context from previous sessions
- Subagents are especially affected (their bootstrap is even more limited)

## What Didn't Work

1. **Compressing USER.md** — still too large, fundamental issue is cumulative memory growth
2. **Splitting into smaller files** — helps but doesn't solve the root cause
3. **Raising bootstrap limits** — `agents.defaults.bootstrapMaxChars` and `agents.defaults.bootstrapTotalMaxChars` — helps but increases token burn
4. **Selective loading** — only load today's memory — misses long-term context

## Root Cause

**Memory accumulation outpaces context window growth.**

- `USER.md` grows with every session (4621 messages, 264 sessions)
- `MEMORY.md` grows with every significant event
- Daily `memory/YYYY-MM-DD.md` files accumulate indefinitely
- Bootstrap loads all of these into the system prompt
- Even with compression, the total exceeds the model's context window

The fundamental issue: **we are treating the context window as a memory system, but it's not.**

## Solution

**Three-tier memory architecture:**

1. **Context Window (Hot)** — Only today's memory + immediate task context
2. **Session History (Warm)** — Recent sessions (7 days), searchable via `sessions_history`
3. **Long-Term Memory (Cold)** — `MEMORY.md` + `memory/*.md`, searchable via `memory_search`

**Implementation:**

```python
# In agent bootstrap, replace full MEMORY.md load with:
- SOUL.md (always, small)
- USER.md (last 2000 chars only)
- memory/YYYY-MM-DD.md (today only)
- MEMORY.md (only if explicitly requested via memory_search)
```

**Baton skill** (already deployed): When any session hits 70% context, trigger handoff to fresh session with summary.

## Prevention

1. **Aggressive memory consolidation** — Weekly review of `memory/*.md`, distill into `MEMORY.md`, archive originals
2. **Memory search as primary retrieval** — Don't load full memory into context; search on demand
3. **Session summaries** — At end of each session, write a 200-char summary to `memory/YYYY-MM-DD.md` instead of full transcript
4. **Archive old memory** — Move `memory/2026-04-*.md` to `memory/archive/` after 30 days

## Verification

After implementing baton + selective loading:
- Context loss reduced from 37% to ~15% (still room for improvement)
- Subagent success rate increased from 87.5% to 92% (less truncation = better task understanding)
- Handoff latency: ~2s (acceptable for long-running tasks)

## Related

- `AGENTS.md` — Baton skill deployment notes
- `docs/DESIGN_FLUX_PYTHON_COMPILER.md` — Example of long design doc that triggers truncation
- `memory/2026-05-22.md` — Session notes on truncation debugging
- `memory/2026-05-25.md` — Merge session (high context usage)
