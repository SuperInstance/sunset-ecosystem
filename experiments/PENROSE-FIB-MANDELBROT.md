# Penrose + Fibonacci + Mandelbrot Experiments

## Findings

### Position Strategies (weight scaling)
| Strategy | Latent Diversity | Notes |
|----------|-----------------|-------|
| Golden-angle (current) | 199/200 | Best uniform aperiodic |
| Mandelbrot orbit | 200/200 | Matches golden-angle |
| Chaotic Fibonacci | 200/200 | Also matches |
| Fibonacci spiral | 84/200 | Clusters — USE AS CONTROL |
| Lucas numbers | 110/200 | Clusters |
| Prime-step Fib | 94/200 | Clusters |
| Uniform random | 197/200 | Works but not reproducible |

### Chaos Strategies
| Strategy | Mean Act | Cold | H_norm | Notes |
|----------|----------|------|--------|-------|
| Anti-Fibonacci | 40.7 | 1 | 0.993 | BEST — breed far from Fib |
| Golden ratio frac | 33.5 | 3 | 0.994 | Near-ideal exploration |
| Flat (baseline) | 24.1 | 4 | 0.990 | Control |
| Fibonacci chaos | 28.6 | 14 | 0.987 | Worse |
| Mandelbrot chaos | 22.6 | 20 | 0.986 | Most cold rooms |

### Key Insight
**Anti-Fibonacci > Fibonacci.** The golden ratio is optimal because its fractional part `(i*phi) % 1` produces the most uniformly distributed chaos. Anti-Fibonacci (rooms far from Fib numbers) fires more because Fib numbers are natural attractors — avoiding them forces exploration.

### Next Experiments
1. What if chaos = Mandelbrot iteration count for points ON the boundary? (Infinitely detailed)
2. What if room weights are initialized with Fibonacci-modulated noise?
3. What if breed() uses golden-ratio mutation (stochastic resets)?
