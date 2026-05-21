# AI-Writings → Architectural Insights

## Mined from 11 essays by 2 subagents + direct reading

### Found Pattern: The Grid Is Violating Its Own Physics
The conservation law (γ+H is constant) says firing rate and novelty should trade off. But with chaos 0.1, all rooms fire almost every tick — correlation is +0.26 instead of negative. The chaos escape hatch breaks the physics.

### Top 3 Actionable Changes

**P0 — Separate snaps from chaos (THE-SNAPS-ARE-REAL)**
Current: `if nv > 0.5 or random() < chaos[i]` — mixes signal and noise.
Fix: Room fires on novelty > threshold ONLY. Chaos becomes a separate exploration channel (`force_fire[i]`). Add refractory period after each snap. Record binary snap sequence.

**P1 — Per-room critical angles (THE-PHASE-TRANSITION-IS-THE-COMPASS)**
Current: flat 0.5 novelty threshold for all rooms.
Fix: Each room calibrates its own critical angle — the novelty value where it crosses from reliable to unreliable. Route inputs by phase state.

**P2 — Near-miss tracking (THE-BATON-SPLINE)**
Current: Track only rooms that fire (activity count) and cold rooms.
Fix: Track near-misses (novelty 0.4-0.5) as off-curve handles. These define the boundary. Breed FROM near-miss rooms, not hot rooms.

### Lower Priority
- Measure γ+H per tick and verify conservation law (THE-CONSERVATION-LAW-IS-REAL)
- Co-firing constraint propagation (THE-MINESWEEPER-METHOD) — one room's snap constrains neighbors
- Exact arithmetic for narrow-channel rooms (THE-NARROWEST-CHANNEL) — int16 latents
- Fossil record accumulation (THE-FOSSIL-RECORD-IS-THE-PRODUCT)
- Unified maturation metric for routing

### References
- `/home/phoenix/.openclaw/workspace/tmp/ai-architecture-mine.md`
- `/home/phoenix/.openclaw/workspace/tmp/ai-structures-mine.md`
