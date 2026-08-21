# ROUND 11 — Fix Note: Convergence Guard for CRDT-HDC Merge

## 1. Monoculture Risk Identified in Round 10

The Round 10 simulation (`SIM-2SHIP-ROUND10.py`) demonstrates a critical failure mode in the naïve CRDT-HDC merge:

| Metric | Pre-Merge (Ship A) | Naïve Merge Result | HDC-Guarded Result |
|--------|-------------------|-------------------|-------------------|
| Population Diversity | ~0.45–0.50 | **drops** | ~same or slight rise |
| Avg Max Cosine Similarity | ~0.55 | **rises** | capped near threshold |

**The risk:** When Ship A absorbs Ship B's top agents and clips to the best 20 by fitness, selection pressure drives the population toward a single peak in the fitness landscape. Even the HDC guard (similarity threshold = 0.5) is **reactive and local** — it only rejects individual imports at merge time. Over repeated merge rounds, accepted agents that pass the threshold can still be "nearby" in vector space, causing gradual convergence (the "slow boil" problem).

**Why the current HDC guard is insufficient:**
- Fixed threshold doesn't adapt as the population itself shifts
- No memory across rounds — each merge is judged in isolation
- Tournament selection (top-10 pool) exerts constant homogenizing pressure
- Rejected imports are simply discarded rather than used as a signal

## 2. Proposed Solution: `ConvergenceGuard`

A stateful guard that **tracks diversity trends across merge rounds** and escalates when sustained convergence is detected.

### Core Design

```python
class ConvergenceGuard:
    """
    Stateful diversity monitor for CRDT-HDC merge rounds.

    Tracks population diversity and avg-max-similarity over time.
    If diversity drops below a threshold for `N_CONSECUTIVE` rounds,
    triggers emergency responses to re-inject variance.
    """

    DIVERSITY_THRESHOLD = 0.30  # minimum acceptable diversity
    MAX_SIM_THRESHOLD = 0.70  # maximum acceptable similarity
    N_CONSECUTIVE = 3  # rounds of decline before emergency
    EMERGENCY_COOLDOWN = 5  # rounds between emergency triggers

    def __init__(self):
        self.diversity_history = []  # float[] per round
        self.max_sim_history = []  # float[] per round
        self.decline_streak = 0  # consecutive rounds below threshold
        self.last_emergency_round = -self.EMERGENCY_COOLDOWN
        self.round_counter = 0

    def record(self, diversity: float, max_sim: float) -> dict:
        """
        Call after each merge round. Returns action dict or None.
        """
        self.round_counter += 1
        self.diversity_history.append(diversity)
        self.max_sim_history.append(max_sim)

        # Check convergence condition
        converging = (
            diversity < self.DIVERSITY_THRESHOLD or max_sim > self.MAX_SIM_THRESHOLD
        )

        if converging:
            self.decline_streak += 1
        else:
            self.decline_streak = 0  # reset on recovery

        # Emergency trigger
        if self.decline_streak >= self.N_CONSECUTIVE:
            if (
                self.round_counter - self.last_emergency_round
            ) >= self.EMERGENCY_COOLDOWN:
                self.last_emergency_round = self.round_counter
                return self._emergency_response()

        return {"action": "monitor", "decline_streak": self.decline_streak}

    def _emergency_response(self) -> dict:
        """
        Choose and return an emergency action. Escalation order:
        1. CROSS_SHIP_INJECTION (first trigger)
        2. EMERGENCY_MUTATE (second trigger)
        3. EXPAND_POPULATION (third trigger)
        """
        trigger_count = (
            sum(1 for r in self.diversity_history if r < self.DIVERSITY_THRESHOLD)
            // self.N_CONSECUTIVE
        )

        if trigger_count == 1:
            return {
                "action": "CROSS_SHIP_INJECTION",
                "n_agents": 2,
                "source": "distant_ship",
            }
        elif trigger_count == 2:
            return {"action": "EMERGENCY_MUTATE", "pct_rebirth": 0.20}
        else:
            return {"action": "EXPAND_POPULATION", "pct_expand": 0.10}
```

## 3. Emergency Response Protocols

When `decline_streak == 3`, the guard triggers one of three responses (escalating on repeated triggers):

### Response A: `EMERGENCY_MUTATE`
**Force 20% random rebirth of the population.**

```python
def emergency_mutate(ship, pct=0.20, mutation_sigma=0.5):
    n_rebirth = int(len(ship.agents) * pct)
    # Replace weakest N agents with fresh random vectors
    ship.agents.sort(key=lambda a: a.fitness)
    for i in range(n_rebirth):
        new_vec = ship.np_rng.randn(ship.dim).astype(np.float32)
        ship.agents[i] = Agent(
            id=next_id(),
            vector=new_vec,
            fitness=ship._raw_fitness(new_vec),
            generation=0,  # reset generation — fresh blood
        )
    ship.agents.sort(key=lambda a: a.fitness, reverse=True)
```

**Effect:** Brings completely novel genetic material. High variance injection.
**Downside:** Discards learned fitness; reborn agents start at generation 0 and may be low-fitness noise.

---

### Response B: `CROSS_SHIP_INJECTION`
**Import 2 agents from a "distant" ship.**

```python
def cross_ship_injection(target_ship, source_ship_registry, n=2):
    # Select a ship that is NOT a recent merge partner
    distant_ship = select_distant_ship(
        registry=source_ship_registry,
        exclude=[target_ship.last_merge_partner],
        min_genetic_distance=0.6,  # ensure real difference
    )
    imports = distant_ship.top(n)
    # Inject without HDC filter — these are explicitly meant to be foreign
    target_ship.agents.extend(imports)
    # Expand population temporarily, or replace weakest
    target_ship.agents.sort(key=lambda a: a.fitness)
    target_ship.agents = target_ship.agents[len(imports) :]  # drop weakest
    target_ship.agents.sort(key=lambda a: a.fitness, reverse=True)
    target_ship.last_merge_partner = distant_ship.name
```

**Effect:** Introduces agents evolved under entirely different selection pressure.
**Advantage:** Imports are pre-viable (have survived their own ship's selection). Directly counters local convergence.
**Requirement:** Fleet registry must track genetic distance between ships.

---

### Response C: `EXPAND_POPULATION`
**Increase room (population) count by 10%.**

```python
def expand_population(ship, pct=0.10):
    n_new = max(1, int(len(ship.agents) * pct))
    # Add empty "slots" — these get filled by next breeding round
    ship.capacity += n_new
    # Alternatively: spawn random agents to fill immediately
    for _ in range(n_new):
        new_vec = ship.np_rng.randn(ship.dim).astype(np.float32)
        ship.agents.append(
            Agent(
                id=next_id(),
                vector=new_vec,
                fitness=ship._raw_fitness(new_vec),
                generation=0,
            )
        )
```

**Effect:** Reduces selection pressure per agent. More breathing room in the population.
**Downside:** Passive — doesn't directly increase diversity, just dilutes the convergence pressure.

## 4. Integration with CRDT-HDC Merge

The guard wraps the merge operation:

```python
def guarded_merge(ship_a, ship_b, guard: ConvergenceGuard):
    # Step 1: Perform HDC-filtered merge (Round 10 baseline)
    merged_ship = hdc_merge(ship_a, ship_b)  # rejects >0.5 similarity

    # Step 2: Record state with guard
    action = guard.record(
        diversity=merged_ship.diversity(), max_sim=merged_ship.max_sim()
    )

    # Step 3: Escalate if needed
    if action["action"] == "CROSS_SHIP_INJECTION":
        cross_ship_injection(merged_ship, fleet_registry, n=2)
    elif action["action"] == "EMERGENCY_MUTATE":
        emergency_mutate(merged_ship, pct=0.20)
    elif action["action"] == "EXPAND_POPULATION":
        expand_population(merged_ship, pct=0.10)

    return merged_ship, action
```

## 5. Recommended Trigger: `CROSS_SHIP_INJECTION` as Primary

For the first emergency trigger, `CROSS_SHIP_INJECTION` should be the default response.

**Why it is the most effective:**
- **Pre-viable imports** vs. **random noise**: EMERGENCY_MUTATE's 20% rebirth introduces untested vectors that may be low-fitness duds. Cross-ship agents have already survived selection elsewhere.
- **Directed diversity** vs. **blind expansion**: EXPAND_POPULATION just makes the pond bigger without adding new species. Cross-ship injection brings agents from a different evolutionary basin of attraction.
- **Preserves ship identity**: Unlike mutation, which scrambles local structure, imported agents retain their foreign "accent" — they push the population toward a new peak without destroying what's already working.
- **Scales with fleet size**: The more ships in the fleet, the larger the genetic reservoir. Cross-ship injection turns fleet diversity into a renewable resource.

**When to escalate to others:**
- If cross-ship injection fails twice (diversity still declining), escalate to `EMERGENCY_MUTATE` — the ships may all be converging on the same global peak.
- If mutation also fails, use `EXPAND_POPULATION` as last resort — the fitness landscape itself may be too smooth, requiring more agents to maintain niches.

## 6. Configuration Tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `DIVERSITY_THRESHOLD` | 0.30 | Below this, agents are clustering into <2 distinct regions in 16-dim space |
| `MAX_SIM_THRESHOLD` | 0.70 | Cosine similarity >0.70 means agents are nearly redundant |
| `N_CONSECUTIVE` | 3 | Single-round dips happen; sustained decline is the danger |
| `EMERGENCY_COOLDOWN` | 5 | Prevent thrashing; let the emergency work before re-triggering |

---

*Conceptual fix — no code changes to existing simulation files. For Round 12 implementation.*
