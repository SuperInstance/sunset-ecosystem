import random
import math
from simulators.tournament_core import (
    Agent,
    crossover,
    mutate,
    tournament_step,
    _mean,
    _std,
)


def simulate(pop_size=20, generations=50, mutation_rate=0.1, thermal_cap=30, seed=42):
    random.seed(seed)
    pop = [Agent() for _ in range(pop_size)]
    history = []
    for gen in range(generations):
        pop, _ = tournament_step(pop, thermal_cap, mutation_rate)
        best = max(pop, key=lambda a: a.fitness())
        history.append({
            'gen': gen,
            'pop': len(pop),
            'best_f': best.fitness(),
            'best_e': best.ethos,
            'best_p': best.pathos,
            'best_l': best.logos,
            'mean_f': sum(a.fitness() for a in pop) / len(pop),
            'std_e': _std([a.ethos for a in pop]),
            'std_p': _std([a.pathos for a in pop]),
            'std_l': _std([a.logos for a in pop]),
        })
    return history


if __name__ == '__main__':
    hist = simulate()
    # Print CSV-like output
    print("gen,pop,best_f,best_e,best_p,best_l,mean_f,std_e,std_p,std_l")
    for h in hist:
        print(f"{h['gen']},{h['pop']},{h['best_f']:.6f},{h['best_e']:.4f},{h['best_p']:.4f},{h['best_l']:.4f},{h['mean_f']:.6f},{h['std_e']:.4f},{h['std_p']:.4f},{h['std_l']:.4f}")
