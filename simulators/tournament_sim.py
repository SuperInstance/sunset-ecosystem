import random
import math

class Agent:
    def __init__(self, ethos=None, pathos=None, logos=None):
        self.ethos = ethos if ethos is not None else random.random()
        self.pathos = pathos if pathos is not None else random.random()
        self.logos = logos if logos is not None else random.random()
    
    def fitness(self):
        return self.ethos * self.pathos * self.logos
    
    def dominates(self, other):
        return (self.ethos >= other.ethos and self.pathos >= other.pathos and self.logos >= other.logos) and \
               (self.ethos > other.ethos or self.pathos > other.pathos or self.logos > other.logos)
    
    def __repr__(self):
        return f"A(e={self.ethos:.3f},p={self.pathos:.3f},l={self.logos:.3f},f={self.fitness():.4f})"

def crossover(a, b):
    alpha = random.random()
    return Agent(
        ethos=alpha*a.ethos + (1-alpha)*b.ethos,
        pathos=alpha*a.pathos + (1-alpha)*b.pathos,
        logos=alpha*a.logos + (1-alpha)*b.logos
    )

def mutate(agent, rate):
    return Agent(
        ethos=max(0, min(1, agent.ethos + random.gauss(0, rate))),
        pathos=max(0, min(1, agent.pathos + random.gauss(0, rate))),
        logos=max(0, min(1, agent.logos + random.gauss(0, rate)))
    )

def tournament_step(pop, thermal_cap, mutation_rate, generation=0, dynamic_cap=False):
    cap = thermal_cap
    if dynamic_cap and generation > 20:
        cap = 50
    winners = []
    random.shuffle(pop)
    for i in range(0, len(pop)-1, 2):
        a, b = pop[i], pop[i+1]
        if a.dominates(b):
            winners.append(a)
        elif b.dominates(a):
            winners.append(b)
        elif a.fitness() > b.fitness():
            winners.append(a)
        else:
            winners.append(b)
    if len(pop) % 2 == 1:
        winners.append(pop[-1])
    
    # Breed: fill up to near thermal cap
    offspring = []
    while len(winners) + len(offspring) < cap and len(winners) >= 2:
        p1, p2 = random.sample(winners, 2)
        child = mutate(crossover(p1, p2), mutation_rate)
        offspring.append(child)
    
    return winners + offspring

def simulate(pop_size=20, generations=50, mutation_rate=0.1, thermal_cap=30, seed=42, dynamic_cap=False):
    random.seed(seed)
    pop = [Agent() for _ in range(pop_size)]
    history = []
    for gen in range(generations):
        pop = tournament_step(pop, thermal_cap, mutation_rate, generation=gen, dynamic_cap=dynamic_cap)
        best = max(pop, key=lambda a: a.fitness())
        mean_e = sum(a.ethos for a in pop) / len(pop)
        mean_p = sum(a.pathos for a in pop) / len(pop)
        mean_l = sum(a.logos for a in pop) / len(pop)
        history.append({
            'gen': gen,
            'pop': len(pop),
            'best_f': best.fitness(),
            'best_e': best.ethos,
            'best_p': best.pathos,
            'best_l': best.logos,
            'mean_f': sum(a.fitness() for a in pop) / len(pop),
            'std_e': math.sqrt(sum((a.ethos - mean_e)**2 for a in pop) / len(pop)),
            'std_p': math.sqrt(sum((a.pathos - mean_p)**2 for a in pop) / len(pop)),
            'std_l': math.sqrt(sum((a.logos - mean_l)**2 for a in pop) / len(pop)),
            'diversity': math.sqrt(sum(
                ((a.ethos - mean_e)**2 + (a.pathos - mean_p)**2 + (a.logos - mean_l)**2)
                for a in pop
            ) / len(pop)),
        })
    return history

if __name__ == '__main__':
    hist = simulate()
    # Print CSV-like output
    print("gen,pop,best_f,best_e,best_p,best_l,mean_f,std_e,std_p,std_l")
    for h in hist:
        print(f"{h['gen']},{h['pop']},{h['best_f']:.6f},{h['best_e']:.4f},{h['best_p']:.4f},{h['best_l']:.4f},{h['mean_f']:.6f},{h['std_e']:.4f},{h['std_p']:.4f},{h['std_l']:.4f}")
