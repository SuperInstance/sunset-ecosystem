---
name: fleet-ecosystem
description: 'Cocapn Fleet science integration for Claw. Provides breeding, consensus, mesh gossip, FLUX VM gating, vector tables, metronome sync, and SSE dashboard. Use when: (1) spawning breeding cycles, (2) checking FLUX constraints, (3) querying mesh vector tables, (4) launching the fleet dashboard, (5) checking fleet-wide status. Requires the fleet bridge daemon (claw_fleet_bridge.py) to be running.'
metadata:
  {
    "openclaw": { "emoji": "🦞" }
  }
---

# Fleet Ecosystem Skill

The Cocapn Fleet's science modules — now available as Claw skills.

## Prerequisites

Start the fleet bridge daemon:

```bash
python3 claw_fleet_bridge.py --port 8850
```

Or let the Fleet Services Extension auto-start it.

## Commands

### `@fleet status`

Get fleet-wide status from FleetConductorV2.

```bash
curl http://localhost:8850/status | python3 -m json.tool
```

### `@fleet breed`

Run a breeding cycle with a FLUX preset.

```bash
curl -X POST http://localhost:8850/breed \
  -H "Content-Type: application/json" \
  -d '{"n_winners": 5, "preset": "diversity"}'
```

Presets: `diversity`, `performance`, `safety`, `exploration`, `convergence`.

### `@fleet flux check`

Check if a candidate agent passes FLUX VM constraints.

```bash
curl -X POST http://localhost:8850/flux/check \
  -H "Content-Type: application/json" \
  -d '{"candidate": {"name": "test_agent", "version": "1.0"}}'
```

### `@fleet flux presets`

List available FLUX constraint presets.

```bash
curl http://localhost:8850/flux/presets | python3 -m json.tool
```

### `@fleet mesh insert`

Insert a vector into the mesh table.

```bash
curl -X POST http://localhost:8850/mesh/insert \
  -H "Content-Type: application/json" \
  -d '{
    "table_id": "esch_population",
    "vector": [0.1, 0.2, 0.3],
    "fitness": 0.85,
    "extra": {"generation": 42}
  }'
```

### `@fleet mesh query`

Query the mesh table by minimum fitness.

```bash
curl -X POST http://localhost:8850/mesh/query \
  -H "Content-Type: application/json" \
  -d '{"min_fitness": 0.8}'
```

### `@fleet dashboard`

Launch the SSE stream dashboard.

```bash
curl http://localhost:8849/dashboard  # or / for auto-redirect
```

### `@fleet bridge health`

Check bridge daemon health.

```bash
curl http://localhost:8850/health
```

## Integration with Claw Tools

The fleet bridge exposes fleet modules as HTTP endpoints. Claw skills can call them via the `bash` or `web_fetch` tools. The bridge lazy-imports fleet modules, so startup is fast even if sunset-ecosystem is large.

## Architecture

```
Claw Gateway
    └── Fleet Ecosystem Skill (SKILL.md)
        └── HTTP calls → FleetBridgeServer (claw_fleet_bridge.py)
            ├── /status        → FleetConductorV2
            ├── /breed         → BreederDaemonV2 + FluxPresetLibrary
            ├── /flux/check    → FluxVMGater
            ├── /flux/presets  → FluxPresetLibrary
            ├── /mesh/insert   → MeshVectorTable
            ├── /mesh/query    → FleetVectorIndex
            └── /health        → health check
```

## References

- `docs/CLAW_INTEGRATION_PLAN.md` — full integration architecture
- `docs/EXOTICA_NLOPT_RESEARCH_BRIEF.md` — FLUX optimization mapping
- `docs/FLEET_BFT_QD.md` — BFT consensus details
