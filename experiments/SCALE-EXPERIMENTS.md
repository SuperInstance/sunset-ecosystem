# Scale Experiments — Phase Change Search

## Methodology
Ran kimi1's tournament breeding sim at 10K, 100K, and 1M agent populations.
Three breeding strategies tested: winners (top 50%), near-misses (0.15-0.35), and lowest 50%.

## Results

### Winners-breed (control) — NO phase change at any scale
- 10K → mean 0.807, CV 0.117
- 100K → mean 0.848, CV 0.123
- 1M → mean 0.849, CV ~0.12
- **Scaling population does NOT increase ceiling. Mean stabilizes at ~0.85.**
- Diversity drops 3x (9869→4161 unique types) over 100 gens.

### Near-miss breeding
- Mean STUCK at 0.25, best caps at 0.55
- **Breeding from the boundary keeps you at the boundary.** Local equilibrium.
- Confirms Baton Spline: off-curve handles define shape but don't pull the curve.

### Reverse breeding (bottom 50%)
- **Death spiral.** Mean crashes to 0.002 in 5 generations.
- Diversity stays HIGH: lock-in to a low-fitness attractor.
- Warning: bad selection functions produce stable-but-terrible populations.

## Key Insight
The conservation law floor is CV ~0.12 at mean ~0.85. You CANNOT push mean higher without accepting lower diversity, and you CANNOT maintain diversity without capping mean. To escape a local equilibrium, you need:
1. An explicit diversity objective (not just fitness)
2. Chaos injection that targets curvature, not amplitude
3. Multi-objective Pareto frontier (like our existing TournamentRound does)

## Reference
Run with: python3 experiments/tournament_sim.py
Or: python3 -c with 50K-1M agents, 30-100 gens.
