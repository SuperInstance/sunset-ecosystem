import random
import math
import csv
from simulators.tournament_core import (
    Agent,
    crossover,
    mutate,
    tournament_step,
    diversity_metric,
)


def simulate(pop_size=20, generations=100, mutation_rate=0.1, thermal_cap=30, strategy='dynamic', seed=42):
    random.seed(seed)
    pop = [Agent() for _ in range(pop_size)]
    history = []
    total_breeding = 0
    gens_to_champion = None
    
    for gen in range(generations):
        cap = thermal_cap
        if strategy == 'dynamic':
            if gen >= 20:
                cap = int(thermal_cap * 1.5)
        
        pop, breed_count = tournament_step(pop, cap, mutation_rate, track_breeding=True)
        total_breeding += breed_count
        
        best = max(pop, key=lambda a: a.fitness())
        best_f = best.fitness()
        
        if gens_to_champion is None and best_f >= 0.9999:
            gens_to_champion = gen + 1
        
        mean_f = sum(a.fitness() for a in pop) / len(pop)
        div = diversity_metric(pop)
        
        history.append({
            'gen': gen,
            'pop': len(pop),
            'best_f': best_f,
            'mean_f': mean_f,
            'diversity': div,
            'cap': cap,
        })
    
    final = history[-1]
    return {
        'gens_to_champion': gens_to_champion if gens_to_champion else generations,
        'final_champion_fitness': final['best_f'],
        'final_mean_fitness': final['mean_f'],
        'final_diversity': final['diversity'],
        'total_breeding': total_breeding,
        'history': history,
    }


if __name__ == '__main__':
    caps = [20, 30, 50, 100, 200]
    strategies = ['fixed', 'dynamic']
    
    results = []
    raw_rows = []
    
    for cap in caps:
        for strategy in strategies:
            print(f"Running: cap={cap}, strategy={strategy} ...")
            res = simulate(thermal_cap=cap, strategy=strategy, generations=100, seed=42)
            results.append({
                'cap': cap,
                'strategy': strategy,
                'gens_to_champion': res['gens_to_champion'],
                'final_champion_fitness': res['final_champion_fitness'],
                'final_mean_fitness': res['final_mean_fitness'],
                'final_diversity': res['final_diversity'],
                'total_breeding': res['total_breeding'],
            })
            for h in res['history']:
                raw_rows.append({
                    'cap': cap,
                    'strategy': strategy,
                    'gen': h['gen'],
                    'pop': h['pop'],
                    'best_f': h['best_f'],
                    'mean_f': h['mean_f'],
                    'diversity': h['diversity'],
                    'cap_used': h['cap'],
                })
    
    # Write CSV
    with open('tournament_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cap', 'strategy', 'gen', 'pop', 'best_f', 'mean_f', 'diversity', 'cap_used'])
        w.writeheader()
        w.writerows(raw_rows)
    
    # Print summary
    print("\n" + "="*80)
    print(f"{'Cap':>5} {'Strategy':>8} {'Gens→1.0':>10} {'Final Mean':>12} {'Diversity':>12} {'Breeding':>10}")
    print("-"*80)
    for r in results:
        print(f"{r['cap']:>5} {r['strategy']:>8} {r['gens_to_champion']:>10} {r['final_mean_fitness']:>12.6f} {r['final_diversity']:>12.6f} {r['total_breeding']:>10}")
    print("="*80)
