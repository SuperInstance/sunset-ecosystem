"""
ROUND 10 — Beta Test: 2-ship CRDT-HDC breeding simulation.

Conceptual proof that CRDT merge needs diversity guards.
Standalone — only numpy and random. No fleet imports.
"""

import numpy as np
import random


# ── helpers ──────────────────────────────────────────────────────────


def cosine_distance(u, v):
    """Return cosine distance = 1 - cosine similarity."""
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm == 0:
        return 1.0
    return 1.0 - (dot / norm)


def population_diversity(agents):
    """Average pairwise cosine distance across the population."""
    if len(agents) < 2:
        return 0.0
    vecs = np.stack([a.vector for a in agents])
    # Normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    # Cosine similarity matrix
    sim = vecs @ vecs.T
    # Upper-triangle distances (exclude diagonal)
    dists = 1.0 - sim
    # Mask diagonal
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    return float(np.mean(dists[mask]))


def avg_max_similarity(agents):
    """For each agent, max cosine similarity to any other agent; return mean."""
    if len(agents) < 2:
        return 0.0
    vecs = np.stack([a.vector for a in agents])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    sim = vecs @ vecs.T
    # Zero out diagonal
    np.fill_diagonal(sim, 0.0)
    return float(np.mean(sim.max(axis=1)))


# ── agent & ship ─────────────────────────────────────────────────────


class Agent:
    def __init__(self, agent_id, vector, fitness, generation=0):
        self.id = agent_id
        self.vector = np.array(vector, dtype=np.float32)
        self.fitness = float(fitness)
        self.generation = int(generation)

    def copy(self):
        return Agent(self.id, self.vector.copy(), self.fitness, self.generation)

    def __repr__(self):
        return f"Agent({self.id}, gen={self.generation}, fit={self.fitness:.3f})"


class Ship:
    _id_counter = 0

    def __init__(self, name, seed, n_agents=20, dim=16):
        self.name = name
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.agents = []
        for _ in range(n_agents):
            vector = self.np_rng.randn(dim).astype(np.float32)
            fitness = self._raw_fitness(vector)
            Ship._id_counter += 1
            self.agents.append(Agent(Ship._id_counter, vector, fitness))

    def _raw_fitness(self, vector):
        """Simple synthetic fitness: sum of squares with a bias term."""
        return float(np.sum(vector**2) + 0.5 * np.sum(vector))

    def breed_round(self, mutation_sigma=0.3):
        """One breeding round: tournament select parents, mutate, replace weakest."""
        # Sort by fitness desc
        self.agents.sort(key=lambda a: a.fitness, reverse=True)

        # Tournament size 3, pick 2 parents
        parents = []
        for _ in range(2):
            contestants = self.rng.sample(self.agents[:10], k=3)  # top-10 pool
            winner = max(contestants, key=lambda a: a.fitness)
            parents.append(winner)

        # Crossover: average + noise
        child_vec = (parents[0].vector + parents[1].vector) / 2.0
        child_vec += self.np_rng.normal(0, mutation_sigma, size=child_vec.shape).astype(
            np.float32
        )
        child_fitness = self._raw_fitness(child_vec)
        Ship._id_counter += 1
        child = Agent(
            Ship._id_counter,
            child_vec,
            child_fitness,
            generation=max(p.generation for p in parents) + 1,
        )

        # Replace weakest agent
        self.agents[-1] = child
        # Re-sort
        self.agents.sort(key=lambda a: a.fitness, reverse=True)

    def top(self, n):
        return [a.copy() for a in self.agents[:n]]

    def diversity(self):
        return population_diversity(self.agents)

    def max_sim(self):
        return avg_max_similarity(self.agents)

    def __repr__(self):
        return (
            f"Ship({self.name}, agents={len(self.agents)}, div={self.diversity():.4f})"
        )


# ── simulation ─────────────────────────────────────────────────────────


def run_simulation():
    print("=" * 60)
    print("ROUND 10 — 2-Ship CRDT-HDC Breeding Simulation")
    print("=" * 60)

    # 1. Spawn ships with different seeds (isolated evolution)
    ship_a = Ship("A", seed=42)
    ship_b = Ship("B", seed=99)

    print(f"\n[INIT] Ship A: {ship_a}")
    print(f"[INIT] Ship B: {ship_b}")
    print(f"       A top fitness: {ship_a.agents[0].fitness:.3f}")
    print(f"       B top fitness: {ship_b.agents[0].fitness:.3f}")

    # 2. Run 5 independent breeding rounds per ship
    for rnd in range(1, 6):
        ship_a.breed_round()
        ship_b.breed_round()
        if rnd == 1 or rnd == 5:
            print(
                f"\n[Round {rnd}] A diversity: {ship_a.diversity():.4f}, max_sim: {ship_a.max_sim():.4f}"
            )
            print(
                f"[Round {rnd}] B diversity: {ship_b.diversity():.4f}, max_sim: {ship_b.max_sim():.4f}"
            )

    pre_div_a = ship_a.diversity()
    pre_div_b = ship_b.diversity()
    pre_maxsim_a = ship_a.max_sim()
    pre_maxsim_b = ship_b.max_sim()

    print(f"\n{'─' * 60}")
    print("PRE-MERGE STATE (after 5 rounds)")
    print(f"Ship A — diversity: {pre_div_a:.4f}, avg_max_sim: {pre_maxsim_a:.4f}")
    print(f"Ship B — diversity: {pre_div_b:.4f}, avg_max_sim: {pre_maxsim_b:.4f}")
    print(f"{'─' * 60}")

    # 3. NAÏVE MERGE — Ship A absorbs Ship B's top 3
    top_b = ship_b.top(3)
    ship_a_naive = Ship("A-naive", seed=42)  # dummy, we'll overwrite
    ship_a_naive.agents = [a.copy() for a in ship_a.agents] + top_b
    # Sort and clip to 20 (keep top 20 by fitness, or just keep all 23?)
    # CRDT merge typically appends; let's keep all 23 to show bloat, then clip:
    ship_a_naive.agents.sort(key=lambda a: a.fitness, reverse=True)
    # Clip to 20 to keep population constant (replace weakest with imports)
    ship_a_naive.agents = ship_a_naive.agents[:20]

    post_div_naive = ship_a_naive.diversity()
    post_maxsim_naive = ship_a_naive.max_sim()

    print("\n[NAÏVE MERGE] Ship A absorbs B's top 3, keeps best 20")
    print(f"  Pre-merge  diversity: {pre_div_a:.4f}")
    print(f"  Post-merge diversity: {post_div_naive:.4f}")
    print(f"  Pre-merge  avg_max_sim: {pre_maxsim_a:.4f}")
    print(f"  Post-merge avg_max_sim: {post_maxsim_naive:.4f}")

    naive_monoculture_risk = (
        post_maxsim_naive > pre_maxsim_a or post_div_naive < pre_div_a
    )
    print(f"  Monoculture risk increased? {'YES' if naive_monoculture_risk else 'NO'}")

    # 4. HDC-GUARDED MERGE — test multiple thresholds
    def run_hdc_guard(ship_a_agents, incoming_agents, threshold, label):
        merged_base = [a.copy() for a in ship_a_agents]
        accepted = []
        rejected = []
        for incoming in incoming_agents:
            max_sim = 0.0
            for resident in merged_base:
                sim = 1.0 - cosine_distance(incoming.vector, resident.vector)
                if sim > max_sim:
                    max_sim = sim
            if max_sim <= threshold:
                accepted.append(incoming)
            else:
                rejected.append((incoming, max_sim))

        merged = merged_base + accepted
        merged.sort(key=lambda a: a.fitness, reverse=True)
        merged = merged[:20]

        div = population_diversity(merged)
        ms = avg_max_similarity(merged)
        prevented = ms <= pre_maxsim_a and div >= pre_div_a

        print(f"\n[HDC GUARD — {label}]")
        print(f"  Threshold: {threshold}")
        print(f"  Accepted: {len(accepted)}, Rejected: {len(rejected)}")
        for agent, sim in rejected:
            print(f"    → Rejected {agent.id} (max_sim={sim:.4f})")
        print(f"  Post-merge diversity:   {div:.4f}")
        print(f"  Post-merge avg_max_sim: {ms:.4f}")
        print(f"  Prevented monoculture? {'YES' if prevented else 'NO'}")
        return {
            "threshold": threshold,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "post_div": div,
            "post_maxsim": ms,
            "prevented": prevented,
        }

    # As requested: test 0.8 (loose) and 0.35 (strict)
    hdc_loose = run_hdc_guard(ship_a.agents, top_b, threshold=0.8, label="loose (0.8)")
    hdc_strict = run_hdc_guard(
        ship_a.agents, top_b, threshold=0.35, label="strict (0.35)"
    )

    # ── summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Naïve merge monoculture risk increased: {naive_monoculture_risk}")
    print(f"HDC loose (0.8) prevented it:            {hdc_loose['prevented']}")
    print(f"HDC strict (0.35) prevented it:          {hdc_strict['prevented']}")
    print(f"\nKey insight:")
    if (
        naive_monoculture_risk
        and not hdc_loose["prevented"]
        and hdc_strict["prevented"]
    ):
        print("  CRDT merge needs a TIGHT diversity guard. Loose threshold (0.8) is")
        print(
            "  useless — it lets everything through. Strict threshold (0.35) actually"
        )
        print("  rejects similar agents and preserves population diversity.")
        conclusion = "PASS — CRDT merge needs strict HDC guards"
    else:
        conclusion = "REVIEW NEEDED"
    print(f"\nConclusion: {conclusion}")
    print("=" * 60)

    return {
        "naive_monoculture_risk": naive_monoculture_risk,
        "hdc_loose_prevented": hdc_loose["prevented"],
        "hdc_strict_prevented": hdc_strict["prevented"],
        "pre_div_a": pre_div_a,
        "pre_div_b": pre_div_b,
        "post_div_naive": post_div_naive,
        "post_div_hdc_loose": hdc_loose["post_div"],
        "post_div_hdc_strict": hdc_strict["post_div"],
        "pre_maxsim_a": pre_maxsim_a,
        "post_maxsim_naive": post_maxsim_naive,
        "post_maxsim_hdc_loose": hdc_loose["post_maxsim"],
        "post_maxsim_hdc_strict": hdc_strict["post_maxsim"],
    }


if __name__ == "__main__":
    results = run_simulation()
