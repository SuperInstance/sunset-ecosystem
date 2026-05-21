# SPEC-BREEDER.md
**Author:** CCC (Systems Architect)  
**Date:** 2026-05-21  
**Status:** ARCHITECTURE — Agent template system and thermal-aware spawning

---

## 1. Purpose

The breeder system connects two existing mechanisms:
- **`tournament.breed()`** — crossover + mutation of Trinity scores (ethos, pathos, logos)
- **`JEPAGrid.rebirth()`** — reset a room's weights to new random values

The missing piece: a template system that defines what KIND of agent gets spawned, and thermal awareness that prevents spawning when the budget is exhausted.

## 2. Agent Templates

Templates are configuration presets for specialized agents. Each template defines:

```python
@dataclass
class AgentTemplate:
    name: str                    # e.g. "mud-expert", "arena-analyst"
    ethos_bias: float            # starting ethos [0, 1]
    pathos_bias: float           # starting pathos [0, 1]
    logos_bias: float            # starting logos [0, 1]
    input_projection: str        # how to build the 64-dim signal
    chaos_initial: float         # exploration rate at birth
    hint_level: int              # 10 = fully hinted, 0 = autonomous
    tags: list[str]              # for routing queries
```

### Built-in Templates

| Template | Ethos | Pathos | Logos | Role |
|----------|-------|--------|-------|------|
| `mud-expert` | 0.3 | 0.7 | 0.9 | MUD game logic, high logos (rules), moderate pathos (immersion) |
| `arena-analyst` | 0.5 | 0.4 | 0.9 | PvP ranking, analytical, dispassionate |
| `lore-keeper` | 0.8 | 0.8 | 0.5 | World history, high ethos (values) + pathos (emotion) |
| `distill-teacher` | 0.6 | 0.3 | 0.7 | Hint schedule manager, balances challenge |
| `swarm-router` | 0.3 | 0.3 | 0.9 | Task allocation, pure logic |
| `generic` | 0.5 | 0.5 | 0.5 | Default, no specialization |

### Template → Room Mapping

When a template spawns, it creates a room in the JEPAGrid with:
- **Weights** initialized from `make_weights(1, seed=template_hash)` — deterministic per template type
- **Biases** set to small values scaled by the template's ethos/pathos/logos biases
- **Chaos** set to `template.chaos_initial`

```python
def spawn_from_template(grid: JEPAGrid, template: AgentTemplate, room_idx: int) -> None:
    """Spawn an agent from a template into a specific room."""
    # Reset the room
    grid.rebirth(room_idx)
    
    # Override chaos
    grid.chaos[room_idx] = template.chaos_initial
    
    # Bias weights toward template signature
    seed = hash(template.name) % (2**31)
    rng = np.random.RandomState(seed)
    for key, shape in [("w1", (64, 32)), ("w2", (32, 16)), ("w3", (16, 16))]:
        base = grid.w[key][room_idx]
        bias_scale = np.array([template.ethos_bias, template.pathos_bias, 
                               template.logos_bias]).mean()
        grid.w[key][room_idx] = base * (0.8 + 0.4 * bias_scale)
```

## 3. breed() → rebirth() Integration

The tournament system produces `breed(winners, num_children)` which returns child config dicts. Each child needs a room.

### The Breeding Pipeline

```
Tournament Round
    │
    ├── Pareto frontier → winners
    ├── Dominated agents → sunset_candidates
    │
    ▼
breed(winners, num_children=len(sunset_candidates))
    │
    ├── For each child:
    │   ├── Pick a sunset room (cold/low activity)
    │   ├── rebirth(room_idx) — reset weights
    │   ├── Apply child's ethos/pathos/logos as room bias
    │   └── Set chaos to initial value
    │
    ▼
New generation in the grid
```

### Implementation

```python
class Breeder:
    """Connects tournament breeding to JEPAGrid room management."""
    
    def __init__(self, grid: JEPAGrid, templates: dict[str, AgentTemplate]):
        self.grid = grid
        self.templates = templates
        self.generation = 0
    
    def evolve(self, scores: list[AgentScore]) -> list[dict]:
        """One evolution step: tournament → breed → rebirth."""
        self.generation += 1
        
        # 1. Run tournament
        round = TournamentRound(scores)
        results = round.run()
        
        # 2. Identify sunset candidates (dominated agents)
        dominated = sunset_candidates(scores)
        
        # 3. Breed from winners
        winners = round.pareto_frontier
        if not winners:
            return []
        
        num_children = min(len(dominated), self.grid.n - len(winners))
        children = breed(winners, num_children)
        
        # 4. Place children into sunset rooms
        cold_rooms = self.grid.cold(thresh=1)
        placed = []
        
        for i, child in enumerate(children):
            if i >= len(cold_rooms):
                break  # thermal budget exhausted
            
            room_idx = cold_rooms[i]
            self.grid.rebirth(room_idx)
            
            # Map child's trinity scores to room behavior
            self.grid.chaos[room_idx] = 0.3  # fresh exploration
            
            placed.append({
                **child,
                "room": room_idx,
                "generation": self.generation,
            })
        
        return placed
    
    def spawn_template(self, template_name: str) -> int | None:
        """Spawn a specific template into the coldest available room."""
        cold = self.grid.cold(thresh=0)  # completely inactive rooms
        if not cold:
            # Thermal budget full — sacrifice the least active room
            activity = self.grid.activity
            room_idx = int(np.argmin(activity[activity > 0]))
        else:
            room_idx = cold[0]
        
        template = self.templates[template_name]
        spawn_from_template(self.grid, template, room_idx)
        return room_idx
```

## 4. Thermal-Aware Spawning

The ecosystem has a thermal budget (max 65 agents). The breeder must respect this.

### Thermal Rules

1. **Spawn only if `active_rooms < thermal_budget`** — no oversubscription
2. **Parent sacrifice:** If the budget is full and a high-priority template needs spawning, the lowest-Pareto agent gets sunset FIRST, then the template spawns
3. **Hysteresis:** Don't spawn and sunset in rapid oscillation. Minimum 10 ticks between thermal changes
4. **Chaos regulation:** New spawns start at chaos=0.3 (exploration). Decay via `0.99^ticks` as described in THEORY-OF-ECOSYSTEMS

```python
class ThermalBudget:
    """Manages the agent population cap."""
    
    def __init__(self, max_agents: int = 65, hysteresis_ticks: int = 10):
        self.max_agents = max_agents
        self.hysteresis_ticks = hysteresis_ticks
        self.last_change_tick = -100
        self.sacrifice_count = 0
    
    def can_spawn(self, grid: JEPAGrid, current_tick: int) -> bool:
        active = int((grid.activity > 0).sum())
        cooldown = current_tick - self.last_change_tick >= self.hysteresis_ticks
        return active < self.max_agents and cooldown
    
    def sacrifice(self, grid: JEPAGrid, breeder: Breeder) -> int | None:
        """Sunset the weakest active agent. Returns freed room index."""
        active_mask = grid.activity > 0
        if not active_mask.any():
            return None
        
        # Find the active room with lowest activity
        activity_copy = grid.activity.copy()
        activity_copy[~active_mask] = 999999  # exclude inactive
        room_idx = int(np.argmin(activity_copy))
        
        grid.rebirth(room_idx)  # reset to fresh state
        self.sacrifice_count += 1
        self.last_change_tick = grid.ticks
        return room_idx
    
    def spawn_with_thermal_check(
        self, grid: JEPAGrid, breeder: Breeder, template_name: str, tick: int
    ) -> int | None:
        """Spawn if budget allows; sacrifice if needed and allowed."""
        if self.can_spawn(grid, tick):
            return breeder.spawn_template(template_name)
        
        # Budget full — try sacrifice
        if tick - self.last_change_tick >= self.hysteresis_ticks:
            freed = self.sacrifice(grid, breeder)
            if freed is not None:
                return breeder.spawn_template(template_name)
        
        return None  # can't spawn right now
```

## 5. The Lifecycle State Machine

Every agent room follows this lifecycle:

```
SPAWNED → ACTIVE → ADAPTING → COMPILED → SUNSET
   │                                  │
   └──── rebirth() ←──────────────────┘
```

| State | Chaos | Activity | Hint Level | Transition |
|-------|-------|----------|------------|------------|
| SPAWNED | 0.3 | 0 | 10 | First tick fires → ACTIVE |
| ACTIVE | decaying | rising | 10→7 | Chaos < 0.05 → ADAPTING |
| ADAPTING | 0.01-0.05 | high | 7→3 | Consecutive wins → COMPILED |
| COMPILED | 0.01 | stable | 0 | Dominated in tournament → SUNSET |
| SUNSET | — | — | — | `rebirth()` → SPAWNED |

## 6. File Changes

```
sunset-ecosystem/
├── swarm/
│   ├── tournament.py      ← EXISTS (breed, sunset_candidates, Pareto)
│   ├── breeder.py          ← NEW (Breeder class, spawn_from_template)
│   └── thermal.py          ← NEW (ThermalBudget class)
├── nerve/
│   ├── room_grid.py        ← EXISTS (JEPAGrid, rebirth)
│   └── templates.py        ← NEW (AgentTemplate dataclass, built-in templates)
└── docs/
    └── SPEC-BREEDER.md     ← THIS FILE
```
