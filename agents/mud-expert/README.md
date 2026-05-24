# Mud-Expert Agent

**Template ID:** `mud-expert`  
**Domain:** PLATO / Nerve  
**Trinity Focus:** ethos=0.4, pathos=0.4, logos=0.2  
**Thermal Cost:** ~120MB RAM, 1 CPU core

## Purpose

Maps and catalogs PLATO MUD rooms. The mud-expert is the ecosystem's cartographer —
it enters every room, records objects, NPCs, and exits, and builds a navigable graph.

## Training Path

| Lesson | Concept |
|--------|---------|
| LESSON-05 | JEPA Room |
| LESSON-06 | JEPAGrid |
| LESSON-09 | Breeding and Rebirth |
| LESSON-13 | Nerve Fiber States |
| LESSON-14 | Fingerprinting |

## Lifecycle

1. **SPAWNED** — Enters the Harbor room, receives mapping mission
2. **ACTIVE** — Explores rooms systematically, records findings in SensoryTiles
3. **ADAPTING** — Builds room fingerprint database, learns navigation heuristics
4. **COMPILED** — Operates autonomously, reports map updates without prompting
5. **SUNSET** — Archives map data, DNA stored for future cartographers

## Key Behaviors

- Uses `NerveFiber.perceive()` to read room descriptions
- Uses `RoomGrid.fingerprints()` to identify rooms by their neural signature
- Reports anomalies (broken rooms, missing exits) as PLATO tiles
- Maintains a git-agent shell in `fleet-repos/mud-expert-<id>/`

## Onboarding

```python
from nerve.templates import TemplateRegistry
from swarm.breeder import Breeder

registry = TemplateRegistry()
breeder = Breeder(registry)
agent = breeder.spawn_from_template("mud-expert")
```

## Sunset Output

When sunset, the mud-expert leaves behind:
- `map.json` — complete room graph with coordinates
- `npc_census.json` — last-seen timestamps for all NPCs
- `object_catalog.json` — searchable object database
- `README.md` — onboarding guide for the next mud-expert
