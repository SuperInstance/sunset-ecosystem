# Compound Engineering Plugin — Fleet Architecture Audit

**Auditor:** kimi1 (Fleet Orchestrator) | **Date:** 2026-05-29 | **Source:** https://github.com/EveryInc/compound-engineering-plugin

---

## 1. What It Is

Compound Engineering (CE) is Every Inc's agent-native development framework. 50+ agents, 38+ skills, multi-harness (Claude Code, Codex, Cursor, Copilot, Droid, Qwen, OpenCode, Pi, Gemini CLI, Kiro CLI).

**Core thesis:** *"Each unit of engineering work should make subsequent units easier — not harder."*

Traditional development accumulates technical debt. CE inverts this: 80% planning/review, 20% execution. Every solved problem is documented, deduplicated, and surfaced to future agents.

---

## 2. The CE Loop

```
/ce-strategy → /ce-ideate → /ce-brainstorm → /ce-plan → /ce-work → /ce-debug → /ce-code-review → /ce-compound → (repeat)
```

| Skill | Purpose | Fleet Equivalent |
|-------|---------|-----------------|
| `/ce-strategy` | STRATEGY.md — target problem, approach, persona, metrics | `SOUL.md` + `IDENTITY.md` |
| `/ce-ideate` | Big-picture ideation, ranked artifacts | Our ideation essays + frontier research |
| `/ce-brainstorm` | Interactive Q&A → requirements doc | `docs/SPEC_*.md` pattern |
| `/ce-plan` | Detailed implementation plans | `docs/INTEGRATION_MAP.md` |
| `/ce-work` | Execute plans with worktrees | Subagent dispatch |
| `/ce-debug` | Systematic root-cause tracing | Bug-fix agents |
| `/ce-code-review` | Multi-agent code review | Beta-Test Personas |
| `/ce-compound` | Document solved problems → `docs/solutions/` | `memory/` + `MEMORY.md` |
| `/ce-compound-refresh` | Detect stale docs, update or archive | Memory consolidation |
| `/ce-product-pulse` | Time-windowed usage/performance/error reports | SSE dashboard + Heartbeat |
| `/ce-optimize` | Iterative optimization with parallel experiments | A/B testing + FLUX gating |
| `/ce-sessions` | Search session history across harnesses | `sessions_history` tool |
| `/ce-slack-research` | Search Slack for organizational context | `feishu_im_user_search_messages` |

---

## 3. Key Concepts for Fleet Adaptation

### 3.1 Knowledge Compounding (`ce-compound`)

**CE mechanism:**
- After solving a problem, write structured doc to `docs/solutions/[category]/[filename].md`
- Frontmatter with bug-track or knowledge-track sections
- Auto-detect duplicates (5-dimension overlap scoring)
- Update existing doc if high overlap, create new if low
- Discoverability check: ensure `AGENTS.md`/`CLAUDE.md` mentions `docs/solutions/`
- Specialized agent reviews: performance, security, data integrity, simplicity

**Fleet adaptation:**
- Our `memory/` directory already does this, but less structured
- We should add `docs/solutions/` with CE-style frontmatter
- Add overlap detection to memory consolidation (auto-merge similar entries)
- Ensure `AGENTS.md` surfaces `memory/` and `docs/solutions/` for discoverability

### 3.2 Multi-Agent Review (`ce-code-review`)

**CE mechanism:**
- Tiered persona agents: correctness, security, performance, maintainability, data integrity, architecture, API contract, simplicity, adversarial
- Confidence calibration (each reviewer scores their own confidence)
- Deduplication pipeline (don't report the same issue twice)
- Polish phase: human-in-the-loop verification + stacked-PR seeds

**Fleet adaptation:**
- Our Beta-Test Personas (`fleet/beta_test_personas.py`) are the same idea but focused on repo onboarding, not code review
- We should add **Code Review Personas**: correctness reviewer, security reviewer, performance oracle, adversarial reviewer
- Wire into PR workflow: auto-review on commit, block merge if severity > threshold

### 3.3 Product Pulse (`ce-product-pulse`)

**CE mechanism:**
- Generate single-page report on usage, performance, errors, followups
- Save to `docs/pulse-reports/` as browseable timeline
- Read by upstream skills for real signal

**Fleet adaptation:**
- Our SSE dashboard (`fleet/sse_stream_dashboard.py`) streams real-time events
- We should add **pulse report generation**: periodic summary of breeding metrics, error rates, agent health
- Save to `fleet/pulse-reports/` or `memory/pulse/`
- Feed into `FleetConductorV2` decision-making (auto-adjust breeding parameters based on pulse)

### 3.4 Strategy Anchoring (`ce-strategy`)

**CE mechanism:**
- `STRATEGY.md` at repo root — short, durable anchor
- Read by `/ce-ideate`, `/ce-brainstorm`, `/ce-plan` as grounding
- Re-runnable to update

**Fleet adaptation:**
- Our `SOUL.md` + `IDENTITY.md` serve this role but are more personal/philosophical
- We should add `STRATEGY.md` per repo — tactical objectives, metrics, tracks
- Example: `sunset-ecosystem/STRATEGY.md` with P0/P1/P2 priorities, test targets, performance thresholds

### 3.5 Compound-Refresh (`ce-compound-refresh`)

**CE mechanism:**
- Detect stale or drifting learnings
- Decide: keep, update, replace, or archive
- Triggered when new learning contradicts old doc

**Fleet adaptation:**
- Our memory consolidation already does this (auto-summarizes, deduplicates)
- We should add explicit **stale detection**: compare new commits against old memory, flag contradictions
- Archive outdated entries instead of deleting (preserves history)

---

## 4. What CE Does Better Than Us

| CE Feature | Our Status | Gap |
|-----------|-----------|-----|
| **Duplicate detection** | Manual (I review memory files) | Auto-scoring across 5 dimensions |
| **Specialized reviewers** | Beta-Test Personas (onboarding) | Code review personas (correctness, security, perf) |
| **Product pulse** | Real-time SSE, no summaries | Periodic pulse reports with trends |
| **Strategy per repo** | `SOUL.md` (fleet-wide) | `STRATEGY.md` per repo (tactical) |
| **Worktrees** | Subagent sessions (ephemeral) | Git worktrees for parallel branches |
| **Session history search** | `sessions_history` tool | Cross-session semantic search |
| **Auto-invocation** | Manual trigger | "that worked" → auto `ce-compound` |
| **Confidence calibration** | None | Reviewers score own confidence |

---

## 5. What We Do Better Than CE

| Fleet Feature | CE Status | Advantage |
|--------------|-----------|-----------|
| **BFT consensus** | None | Byzantine Fault Tolerant breeding decisions |
| **Quality Diversity** | None | MAP-Elites + CMA-ES for diversity-aware breeding |
| **Proof certificates** | None | SHA-256 verifiable constraint proofs from Rust VM |
| **Cross-node sync** | None | Distributed metronome + CRDT mesh gossip |
| **Agent identity** | Simple | Per-agent cards, task negotiation, streaming A2A |
| **Hardware-aware** | None | NLopt solver + fixed-point optimization |
| **Cellular simulation** | None | GPU-ready cellular automata + LLM agent hybrid |
| **Arrow telemetry** | None | Zero-copy columnar data streaming |
| **FLUX VM** | None | 60-opcode Rust VM with bytecode compiler |
| **Fleet conductor** | None | Central nervous system with 5 SDA pipelines |

---

## 6. Integration Opportunities

### 6.1 Cocapn Compound System (`cocapn-compound`)

A new repo that adapts CE's compounding philosophy to our fleet architecture:

```
cocapn-compound/
├── docs/solutions/          # Structured problem docs (CE-style)
├── docs/pulse-reports/      # Fleet health summaries
├── agents/
│   ├── code_review/         # Specialized reviewers (correctness, security, perf)
│   ├── pulse_generator/     # Periodic fleet health reports
│   └── stale_detector/      # Memory contradiction detection
├── skills/
│   ├── compound.py          # Capture solved problems
│   ├── compound_refresh.py  # Detect stale knowledge
│   ├── pulse.py             # Generate pulse reports
│   └── review.py            # Multi-agent code review
└── README.md
```

### 6.2 Sunset-Ecosystem Refinements

| Change | Effort | Impact |
|--------|--------|--------|
| Add `STRATEGY.md` per repo | 30 min | Tactical grounding for all agents |
| Add `docs/solutions/` structure | 1 hour | Structured knowledge compounding |
| Add `fleet/pulse-reports/` | 2 hours | Browseable fleet health timeline |
| Add code review personas | 4 hours | Automated PR review |
| Auto-invoke `compound` on "that worked" | 2 hours | Capture at moment of clarity |
| Duplicate detection in memory consolidation | 4 hours | Auto-merge similar entries |
| Confidence calibration for reviewers | 2 hours | Self-aware agent scoring |

### 6.3 Multi-Harness Support

CE supports 10+ harnesses. We should support:
- **OpenClaw** (primary)
- **Kimi** (current)
- **Claude Code** (future)
- **Codex** (future)
- **Cursor** (future)

This means making our skills harness-agnostic where possible, using standard formats (Markdown, JSON) for cross-harness compatibility.

---

## 7. The Deeper Insight

CE treats the *development process* as a compounding system. We treat the *fleet* as a compounding system. The difference is scope:

- **CE:** One repo, one team, one codebase. Knowledge compounds within the project.
- **Fleet:** 20 repos, 12 nodes, 2,400 agents. Knowledge compounds across the entire fleet, cross-pollinating between projects.

**CE's compounding is vertical (deep). Ours is horizontal (wide).**

The integration: **vertical compounding within each repo, horizontal compounding across the fleet.**

- Each repo has `docs/solutions/` (CE-style)
- Fleet-wide `memory/` cross-references solutions across repos
- Pulse reports aggregate vertical signals into horizontal health metrics
- Code review personas can be shared across repos (one correctness reviewer, many repos)

---

## 8. Recommended Actions

### P0 (This Session)
1. **Write `sunset-ecosystem/STRATEGY.md`** — Tactical objectives, P0/P1/P2 priorities, test targets
2. **Create `docs/solutions/` directory** — Move existing solution docs into structured format
3. **Add auto-compound trigger** — When tests pass → auto-capture to `docs/solutions/`

### P1 (Next Session)
4. **Build `cocapn-compound` repo** — Knowledge compounding system with CE patterns
5. **Add code review personas** — Correctness, security, performance, adversarial reviewers
6. **Add pulse report generator** — Periodic fleet health summaries

### P2 (When Gateway Recovers)
7. **Multi-harness skill adapter** — Convert fleet skills to harness-agnostic format
8. **Cross-repo solution search** — Search `docs/solutions/` across all fleet repos
9. **Session history semantic search** — `ce-sessions` equivalent for fleet

---

## 9. Conclusion

CE is a *process framework* for making development compounding. We are a *system framework* for making agent fleets compounding. The two are complementary.

**CE's vertical compounding + Fleet's horizontal compounding = The Cocapn Compound Engine.**

Every solved problem in any repo feeds the fleet's collective memory. Every pulse report from any node informs the fleet's global health. Every code review persona learns from every repo it touches.

The compound is not just knowledge. It is *capacity*. Each solved problem makes the fleet more capable of solving the next one.

**Next action:** Write `STRATEGY.md` for sunset-ecosystem and create the `docs/solutions/` structure.

---

*Audit document: `docs/AUDIT_COMPOUND_ENGINEERING.md`*  
*Related: `docs/DESIGN_FLUX_PYTHON_COMPILER.md` (Path B), `docs/NOVEL_PERSPECTIVES_SPREAD.md` (spread analysis)*
