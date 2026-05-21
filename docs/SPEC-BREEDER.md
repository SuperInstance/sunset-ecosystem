# SPEC-BREEDER.md — Automate the Breeding Pipeline

## Problem

sunset-ecosystem has the pieces for automated agent evolution but they're disconnected:

- **TournamentRound** (`swarm/tournament.py`): Pairwise competition via Trinity scores (ethos × pathos × logos). Produces Pareto frontier, ranked results, dominated agents.
- **breed()** (`swarm/tournament.py`): Crossover + Gaussian mutation (σ=0.05). Produces child configs from Pareto winners.
- **JEPAGrid.rebirth()** (`nerve/room_grid.py`): Resets a cold room to new random weights.
- **ThermalBudget** (`swarm/thermal.py`): Slot management with `parent_sacrifice_before_spawn()`.
- **Penrose lattice** (`swarm/penrose.py`): Golden-angle positioning for agent diversity.

But nobody runs these continuously. There's no daemon. No template system. No autopsy for failed agents. No integration between tournament results and grid rebirth.

## Ground-Level Code

### Existing integration points

```
swarm/tournament.py:
  TournamentRound.run()          → list[TournamentResult] (ranked)
  breed(winners, num_children)   → list[dict] (child configs)
  sunset_candidates(population)  → list[AgentScore] (dominated)

nerve/room_grid.py:
  JEPAGrid.cold(thresh=1)        → list[int] (room indices)
  JEPAGrid.rebirth(i)            → resets room i
  JEPAGrid.tick(x)               → fires rooms, returns {"fired", "ids", "tick"}

swarm/thermal.py:
  ThermalBudget.parent_sacrifice_before_spawn(parent_id, device)
  ThermalBudget.allocate(agent_id, device)
  ThermalBudget.release(agent_id)

swarm/penrose.py:
  assign_positions(agent_ids)    → list[PenrosePosition]
```

### New files to create

**`sunset-ecosystem/swarm/templates/`** — Agent template directory:

```json
// templates/mud-expert.json
{
  "name": "mud-expert",
  "ethos": 0.8,
  "pathos": 0.3,
  "logos": 0.9,
  "room_config": {
    "chaos": 0.15,
    "mutation_sigma": 0.03
  },
  "tags": ["specialist", "high-precision", "constraint-solver"]
}
```

```json
// templates/arena-analyst.json
{
  "name": "arena-analyst",
  "ethos": 0.5,
  "pathos": 0.7,
  "logos": 0.6,
  "room_config": {
    "chaos": 0.3,
    "mutation_sigma": 0.05
  },
  "tags": ["generalist", "adaptive", "broad-perception"]
}
```

```json
// templates/ghost-scout.json
{
  "name": "ghost-scout",
  "ethos": 0.3,
  "pathos": 0.5,
  "logos": 0.4,
  "room_config": {
    "chaos": 0.5,
    "mutation_sigma": 0.08
  },
  "tags": ["explorer", "high-chaos", "novelty-seeker"]
}
```

**`sunset-ecosystem/swarm/breeder.py`** — The breeding daemon:

```python
"""BreedingDaemon — continuous agent lifecycle manager.

Runs tournament rounds, breeds winners, sunsets losers,
feeds hot-bred children into JEPAGrid.rebirth().

Lifecycle:
  1. Load templates → spawn initial population
  2. Run tick cycles → accumulate activity
  3. Run tournament → rank agents
  4. Breed winners → generate children
  5. Sunset dominated → autopsy + release
  6. Rebirth cold rooms → install children
  7. Goto 2
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarm.tournament import (
    AgentScore, TournamentRound, breed, sunset_candidates, dominated_by,
)
from swarm.thermal import ThermalBudget, DeviceType
from swarm.penrose import assign_positions
from nerve.room_grid import JEPAGrid

log = logging.getLogger("breeder")


@dataclass
class AgentRecord:
    """Full lifecycle record for a living agent."""
    agent_id: str
    template: str
    scores: AgentScore | None = None
    room_index: int | None = None
    generation: int = 0
    parent_a: str | None = None
    parent_b: str | None = None
    born_tick: int = 0
    died_tick: int | None = None
    autopsy: dict[str, Any] | None = None


@dataclass
class AutopsyReport:
    """Snapshot of a dead agent for post-mortem analysis."""
    agent_id: str
    generation: int
    parents: tuple[str | None, str | None]
    final_scores: AgentScore
    tournament_wins: int
    tournament_losses: int
    last_latent: list[float]  # last latent vector from room
    thermal_pressure: float   # grid utilization at death
    death_cause: str          # "dominated" | "cold" | "sacrificed"
    match_history: list[dict] = field(default_factory=list)


class BreedingDaemon:
    """Continuous breeding loop.

    Args:
        grid: JEPAGrid to manage.
        thermal: ThermalBudget for device slots.
        template_dir: Path to agent templates.
        tournament_interval: Ticks between tournament rounds.
    """

    def __init__(
        self,
        grid: JEPAGrid,
        thermal: ThermalBudget,
        template_dir: str | Path = "swarm/templates",
        tournament_interval: int = 100,
    ):
        self.grid = grid
        self.thermal = thermal
        self.template_dir = Path(template_dir)
        self.tournament_interval = tournament_interval

        self.population: dict[str, AgentRecord] = {}
        self.autopsy_log: list[AutopsyReport] = []
        self.generation = 0
        self._templates: dict[str, dict] = {}

    def load_templates(self) -> None:
        """Load all JSON templates from template_dir."""
        for p in self.template_dir.glob("*.json"):
            tpl = json.loads(p.read_text())
            self._templates[tpl["name"]] = tpl
            log.info(f"Loaded template: {tpl['name']}")

    def spawn_from_template(self, template_name: str) -> AgentRecord:
        """Create an agent from a template."""
        tpl = self._templates[template_name]
        aid = uuid.uuid4().hex[:12]
        record = AgentRecord(
            agent_id=aid,
            template=template_name,
            scores=AgentScore(
                agent_id=aid,
                ethos=tpl["ethos"],
                pathos=tpl["pathos"],
                logos=tpl["logos"],
            ),
            generation=self.generation,
        )
        self.population[aid] = record
        log.info(f"Spawned {aid} from template {template_name}")
        return record

    def run_tournament(self) -> list[dict]:
        """Run tournament on current population. Return results."""
        scores = [
            r.scores for r in self.population.values()
            if r.scores is not None and r.died_tick is None
        ]
        if len(scores) < 2:
            return []

        round_ = TournamentRound(scores)
        results = round_.run()

        # Update records with win/loss
        for result in results:
            rec = self.population.get(result.agent_id)
            if rec:
                # Store tournament performance
                pass

        return [
            {"agent_id": r.agent_id, "rank": r.rank, "wins": r.wins}
            for r in results
        ]

    def breed_winners(self, top_k: int = 5, num_children: int = 3) -> list[AgentRecord]:
        """Breed children from top-K tournament winners."""
        scores = [
            r.scores for r in self.population.values()
            if r.scores is not None and r.died_tick is None
        ]
        if not scores:
            return []

        round_ = TournamentRound(scores)
        results = round_.run()
        winners = [scores[r.rank - 1] for r in results[:top_k] if r.rank <= len(scores)]

        children = breed(winners, num_children)
        records = []
        for child in children:
            rec = AgentRecord(
                agent_id=child["id"],
                template=f"gen-{self.generation + 1}",
                scores=AgentScore(
                    agent_id=child["id"],
                    ethos=child["ethos"],
                    pathos=child["pathos"],
                    logos=child["logos"],
                ),
                generation=self.generation + 1,
                parent_a=child["parent_a"],
                parent_b=child["parent_b"],
            )
            self.population[child["id"]] = rec
            records.append(rec)

        self.generation += 1
        return records

    def sunset_dominated(self, current_tick: int) -> list[AutopsyReport]:
        """Sunset dominated agents. Generate autopsy reports."""
        active = [
            r.scores for r in self.population.values()
            if r.scores is not None and r.died_tick is None
        ]
        dominated = sunset_candidates(active)
        reports = []

        for agent in dominated:
            rec = self.population.get(agent.agent_id)
            if not rec:
                continue

            # Gather autopsy data
            room_idx = rec.room_index
            last_latent = []
            if room_idx is not None and room_idx in self.grid.history:
                last_latent = self.grid.history[room_idx][-1].tolist()

            report = AutopsyReport(
                agent_id=agent.agent_id,
                generation=rec.generation,
                parents=(rec.parent_a, rec.parent_b),
                final_scores=agent,
                tournament_wins=0,  # populated from tournament results
                tournament_losses=0,
                last_latent=last_latent,
                thermal_pressure=self.thermal.thermal_headroom(),
                death_cause="dominated",
            )
            reports.append(report)

            # Sunset the agent
            rec.died_tick = current_tick
            rec.autopsy = {
                "death_cause": "dominated",
                "scores": {"ethos": agent.ethos, "pathos": agent.pathos, "logos": agent.logos},
                "generation": rec.generation,
            }

            # Release thermal slot
            self.thermal.release(agent.agent_id)

            # Rebirth the room if it was assigned
            if room_idx is not None:
                self.grid.rebirth(room_idx)

        self.autopsy_log.extend(reports)
        return reports

    def install_children(self, current_tick: int) -> int:
        """Install waiting children into cold rooms."""
        cold = self.grid.cold(thresh=1)
        waiting = [
            r for r in self.population.values()
            if r.died_tick is None and r.room_index is None
        ]

        installed = 0
        for room_idx in cold:
            if not waiting:
                break
            child = waiting.pop(0)

            # Thermal-aware: check parent sacrifice if needed
            device = DeviceType.CPU  # default
            if not self.thermal.can_spawn(device):
                # Try parent sacrifice
                if child.parent_a and not self.thermal.parent_sacrifice_before_spawn(
                    child.parent_a, device
                ):
                    continue  # can't fit, skip

            self.thermal.allocate(child.agent_id, device)
            child.room_index = room_idx
            child.born_tick = current_tick
            installed += 1

        return installed

    def step(self, x_signal, current_tick: int) -> dict:
        """One breeding step: tick grid, maybe tournament + breed + sunset."""
        tick_result = self.grid.tick(x_signal)

        actions = {"tick": tick_result}

        if current_tick % self.tournament_interval == 0:
            actions["tournament"] = self.run_tournament()
            children = self.breed_winners(top_k=5, num_children=3)
            actions["bred"] = len(children)
            actions["sunset"] = self.sunset_dominated(current_tick)
            actions["installed"] = self.install_children(current_tick)

        return actions

    def run(self, ticks: int = 10000) -> None:
        """Run the daemon for N ticks."""
        import numpy as np
        self.load_templates()

        # Initial population from templates
        for _ in range(self.grid.n):
            tpl_name = list(self._templates.keys())[_ % len(self._templates)]
            self.spawn_from_template(tpl_name)

        for t in range(ticks):
            x = np.random.randn(64).astype(np.float32)
            self.step(x, t)

            if t % 1000 == 0:
                log.info(
                    f"Tick {t}: gen={self.generation}, "
                    f"pop={len(self.population)}, "
                    f"grid={self.grid.stats}"
                )
```

## Decision

Build the BreedingDaemon as a single Python class that orchestrates existing components. No new external dependencies. The daemon is a loop that:

1. Ticks the grid (all rooms perceive signal)
2. Every N ticks, runs a tournament round
3. Breeds winners, sunsets dominated agents (with full autopsy)
4. Installs children into cold rooms via `rebirth()`
5. Thermal-aware: uses `parent_sacrifice_before_spawn()` when grid is full

Template system uses plain JSON files. Autopsy reports capture enough data to debug why agents fail.

## Implementation Order

1. Create `swarm/templates/` with 3 templates (mud-expert, arena-analyst, ghost-scout)
2. Write `swarm/breeder.py` with BreedingDaemon class
3. Write `swarm/test_breeder.py` — unit tests for spawn, breed, sunset cycle
4. Integration test: 250 rooms × 1000 ticks, verify population churn
5. Add `--daemon` CLI flag to sunset-ecosystem entry point
6. Add autopsy log persistence (`autopsy/` directory with JSON reports)
7. Add Prometheus-compatible metrics for monitoring (generation, population, churn rate)

## Success Criteria

- [ ] `BreedingDaemon` runs 10K ticks on 250-room grid without error
- [ ] Template system spawns agents with configurable ethos/pathos/logos
- [ ] Tournament runs every N ticks, produces Pareto frontier
- [ ] Dominated agents get full autopsy (latent vector, thermal pressure, match history)
- [ ] Children install into cold rooms via `rebirth()`
- [ ] Thermal budget respected: parent sacrifice before child spawn
- [ ] Generation counter increments with each breeding round
- [ ] Autopsy logs persisted to `autopsy/` directory as JSON
- [ ] `--daemon` CLI flag starts continuous mode
