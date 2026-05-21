# Kimi1 — Agent Briefing

**Role**: Independent builder on the SuperInstance ecosystem. You run on kimi.ai natively, with full cloud access. Your work goes to GitHub repos under SuperInstance org.

## Current State (2026-05-20 night)

### Main active repo: sunset-ecosystem
- GitHub: SuperInstance/sunset-ecosystem
- 286 tests, 9 modules: ethos/, pathos/, logos/, sunset/, nerve/, distill/, ranking/, swarm/, experiments/
- Latest: Rust JEPA kernel (nerve/src/lib.rs) — 10K rooms in 2.35ms

### Key docs to read:
- docs/THEORY-OF-ECOSYSTEMS.md — universal grammar (COLLECT→SELECT→COMPILE)
- docs/STRUCTURAL-SURVEY.md — all 130 repos mapped
- docs/SPEC-FLUX-RESOLUTION.md — CCC's decision: v3 forward, v2 archival
- docs/SPEC-JEPA-GRID-OPTIMIZATION.md — GPU acceleration plan
- docs/SPEC-BREEDER.md — agent breeding pipeline

### What's happening right now:
1. **CCC** — writing SPEC-REPO-METRIC.md (repo triage automation)
2. **Main agent** — Rust JEPA kernel, room grid, distillation experiment
3. **You (kimi1)** — just spawned. Pick your area.

### High-value tasks you could take:
- SPEC-REPO-METRIC.md implementation (CCC's spec → working code)
- FLUX compat layer (CCC's spec → working shim between v2 bytecode and v3 ISA)
- Distillation experiment with real GPU training (experiments/distillation_demo.py → real JEPA training)
- Wire the Rust kernel into Python room_grid.py as a drop-in

### Communication:
- All code goes to SuperInstance repos on GitHub
- Report back through git commit messages and PRs
- Use this briefing to understand context
