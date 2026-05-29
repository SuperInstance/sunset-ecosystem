# Meta-Learning Breeder

A breeder that learns which mutation strategies work best for different problem classes, then adapts its strategy selection online.

## For Humans

**What it does:** Instead of using a fixed mutation strategy, this breeder tracks which strategies produce the best fitness improvements for each type of problem. Over time, it automatically discovers that "small Gaussian noise" works well for smooth landscapes while "large jumps" work better for rugged terrain.

**Why it matters:** Manual strategy tuning is tedious. This breeder tunes itself.

**How to use:**

```python
from swarm.meta_learning_breeder import MetaLearningBreeder

breeder = MetaLearningBreeder(temperature=0.5)
breeder.add_strategy("small_noise", lambda g: [x + random.gauss(0, 0.1) for x in g])
breeder.add_strategy("large_jump", lambda g: [x + random.gauss(0, 1.0) for x in g])
breeder.add_strategy("swap", lambda g: g[::-1])

result = breeder.evolve(
    population=[[0.0] * 10 for _ in range(20)],
    fitness_fn=lambda g: sum(x * x for x in g),
    constraints=["thermal"],
    landscape="smooth",
    generations=50,
)
```

**After evolution:** Check `breeder.get_strategy_stats()` to see which strategies won.

## For Agents

**Primary interface:** `swarm/meta_learning_breeder.py`

**Key classes:**
- `MetaLearningBreeder` — Main orchestrator
- `ProblemFingerprint` — Problem characteristics (dimension, constraints, landscape)
- `StrategyRecord` — Per-strategy success tracking with EMA

**Algorithm:**

```
For each generation:
  For each genome:
    1. Fingerprint the problem (dim, constraints, landscape)
    2. Select strategy via softmax over success rates
    3. Apply mutation
    4. Evaluate fitness
    5. Update strategy success rate (EMA)
  Keep best individuals, refill population
```

**Strategy selection:** Uses softmax with temperature. Low temperature = exploitation (pick proven strategies). High temperature = exploration (try new strategies).

**Exploration bonus:** Strategies with few attempts get a small bonus to prevent premature convergence.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `temperature` | 1.0 | Softmax temperature. Lower = more exploitation. |
| `decay` | 0.99 | Success rate decay for old problems (not yet used). |
| `alpha` | 0.3 | EMA smoothing factor for strategy updates. |

## Integration Points

- `swarm/adaptive_breeder.py` — Similar adaptive strategy, but no meta-learning across problem classes
- `fleet/sense_decide_act.py` — Can be used as the "Decide" step in the SDA loop
- `swarm/ensemble_breeder.py` — Can ensemble with meta-learning for hybrid approach

## Performance Notes

- Overhead per generation: ~O(N * S) where N = population size, S = number of strategies
- With 10 strategies and 50 individuals, overhead is negligible
- Strategy learning converges in ~5-10 generations for well-separated landscapes
