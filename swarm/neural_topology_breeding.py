"""neural_topology_breeding.py — NEAT-style neural architecture search for agent breeding.

Extends the breeding loop with topology mutation operators for neural
network agents. Each agent is a neural network genotype that evolves
both weights and structure.

Mutation operators (from Stanley & Miikkulainen 2002, Real et al. 2019):
1. Add node — split an existing connection, insert a new neuron
2. Add connection — connect two previously unconnected neurons
3. Mutate weight — Gaussian perturbation of connection weights
4. Toggle connection — enable/disable a connection
5. Change activation — mutate the activation function of a neuron

Genome encoding: directed graph with innovation numbers for historical
marking, enabling crossover alignment.

Reference: Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural
Networks through Augmenting Topologies." Evolutionary Computation.
"""
from __future__ import annotations

__all__ = [
    "NeuralGenome",
    "NEATBreeder",
    "add_node_mutation",
    "add_connection_mutation",
    "weight_mutation",
    "toggle_mutation",
    "activation_mutation",
]

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Genome representation ─────────────────────────────────────

@dataclass
class NeuronGene:
    """A single neuron in the genotype."""
    id: int
    neuron_type: str = "hidden"  # "input", "output", "hidden"
    activation: str = "tanh"      # "tanh", "relu", "sigmoid", "linear"
    bias: float = 0.0


@dataclass
class ConnectionGene:
    """A weighted connection between neurons."""
    innovation: int          # global innovation counter ID
    from_id: int
    to_id: int
    weight: float
    enabled: bool = True


@dataclass
class NeuralGenome:
    """Complete neural network genotype."""
    neurons: dict[int, NeuronGene] = field(default_factory=dict)
    connections: dict[int, ConnectionGene] = field(default_factory=dict)
    fitness: float = 0.0
    species_id: int | None = None

    # Genome metadata
    num_inputs: int = 0
    num_outputs: int = 0
    next_neuron_id: int = 0
    next_innovation: int = 0

    def copy(self) -> "NeuralGenome":
        return copy.deepcopy(self)

    def add_neuron(self, neuron_type: str = "hidden", activation: str = "tanh", bias: float = 0.0) -> NeuronGene:
        nid = self.next_neuron_id
        self.next_neuron_id += 1
        n = NeuronGene(id=nid, neuron_type=neuron_type, activation=activation, bias=bias)
        self.neurons[nid] = n
        return n

    def add_connection(self, from_id: int, to_id: int, weight: float) -> ConnectionGene | None:
        # Don't add if connection already exists
        for c in self.connections.values():
            if c.from_id == from_id and c.to_id == to_id:
                return None
        # Don't add recurrent loops (simplified: no self-loops)
        if from_id == to_id:
            return None
        inv = self.next_innovation
        self.next_innovation += 1
        c = ConnectionGene(innovation=inv, from_id=from_id, to_id=to_id, weight=weight)
        self.connections[inv] = c
        return c

    def to_adjacency_list(self) -> dict[int, list[tuple[int, float, bool]]]:
        """Adjacency list: neuron_id -> [(to_id, weight, enabled)]."""
        adj: dict[int, list[tuple[int, float, bool]]] = {n: [] for n in self.neurons}
        for c in self.connections.values():
            if c.from_id in adj:
                adj[c.from_id].append((c.to_id, c.weight, c.enabled))
        return adj

    def compute_depth(self) -> dict[int, int]:
        """Topological depth of each neuron (input=0)."""
        depths: dict[int, int] = {}
        # Mark inputs
        for nid, n in self.neurons.items():
            if n.neuron_type == "input":
                depths[nid] = 0

        # BFS from inputs
        changed = True
        while changed:
            changed = False
            for c in self.connections.values():
                if not c.enabled:
                    continue
                if c.from_id in depths:
                    new_depth = depths[c.from_id] + 1
                    if c.to_id not in depths or new_depth > depths[c.to_id]:
                        depths[c.to_id] = new_depth
                        changed = True
        return depths

    def feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """Simple feedforward evaluation (assumes feedforward topology)."""
        values: dict[int, float] = {}
        depths = self.compute_depth()
        max_depth = max(depths.values()) if depths else 0

        # Set input values
        input_ids = [n.id for n in self.neurons.values() if n.neuron_type == "input"]
        for i, nid in enumerate(input_ids):
            values[nid] = float(inputs[i]) if i < len(inputs) else 0.0

        # Compute layer by layer
        for d in range(1, max_depth + 1):
            layer_neurons = [n for n in self.neurons.values() if depths.get(n.id) == d]
            for n in layer_neurons:
                total = n.bias
                for c in self.connections.values():
                    if c.to_id == n.id and c.enabled and c.from_id in values:
                        total += values[c.from_id] * c.weight
                values[n.id] = _activate(total, n.activation)

        # Collect outputs
        output_ids = sorted([n.id for n in self.neurons.values() if n.neuron_type == "output"])
        return np.array([values.get(oid, 0.0) for oid in output_ids])


def _activate(x: float, fn: str) -> float:
    if fn == "tanh":
        return float(np.tanh(x))
    elif fn == "relu":
        return max(0.0, x)
    elif fn == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    elif fn == "linear":
        return x
    else:
        return float(np.tanh(x))


# ── Mutation operators ──────────────────────────────────────

_ACTIVATIONS = ["tanh", "relu", "sigmoid", "linear"]


def add_node_mutation(
    genome: NeuralGenome,
    rng: np.random.Generator | None = None,
) -> NeuralGenome:
    """Mutation 1: Split an existing connection, insert a new neuron.

    If connection A→B with weight w exists:
    - Disable A→B
    - Add neuron H
    - Add A→H with weight 1.0
    - Add H→B with weight w
    """
    rng = rng or np.random.default_rng()
    enabled_conns = [c for c in genome.connections.values() if c.enabled]
    if not enabled_conns:
        return genome

    c = rng.choice(enabled_conns)
    c.enabled = False

    # New neuron
    h = genome.add_neuron(neuron_type="hidden", activation="tanh", bias=0.0)

    # New connections
    genome.add_connection(c.from_id, h.id, weight=1.0)
    genome.add_connection(h.id, c.to_id, weight=c.weight)

    return genome


def add_connection_mutation(
    genome: NeuralGenome,
    rng: np.random.Generator | None = None,
) -> NeuralGenome:
    """Mutation 2: Add a new connection between two unconnected neurons.

    Only connects lower-depth to higher-depth (feedforward constraint).
    """
    rng = rng or np.random.default_rng()
    depths = genome.compute_depth()

    candidates: list[tuple[int, int]] = []
    for n1 in genome.neurons.values():
        for n2 in genome.neurons.values():
            if n1.id == n2.id:
                continue
            d1 = depths.get(n1.id, 0)
            d2 = depths.get(n2.id, 0)
            if d1 >= d2:
                continue  # feedforward: only forward connections
            # Check not already connected
            exists = any(
                c.from_id == n1.id and c.to_id == n2.id
                for c in genome.connections.values()
            )
            if not exists:
                candidates.append((n1.id, n2.id))

    if not candidates:
        return genome

    from_id, to_id = candidates[rng.integers(len(candidates))]
    weight = rng.normal(0.0, 1.0)
    genome.add_connection(from_id, to_id, weight)
    return genome


def weight_mutation(
    genome: NeuralGenome,
    perturbation_rate: float = 0.8,
    perturbation_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> NeuralGenome:
    """Mutation 3: Gaussian perturbation of connection weights."""
    rng = rng or np.random.default_rng()
    for c in genome.connections.values():
        if not c.enabled:
            continue
        if rng.random() < perturbation_rate:
            c.weight += rng.normal(0.0, perturbation_std)
        else:
            # Full replacement
            c.weight = rng.normal(0.0, 1.0)
    return genome


def toggle_mutation(
    genome: NeuralGenome,
    toggle_rate: float = 0.05,
    rng: np.random.Generator | None = None,
) -> NeuralGenome:
    """Mutation 4: Randomly enable/disable connections."""
    rng = rng or np.random.default_rng()
    for c in genome.connections.values():
        if rng.random() < toggle_rate:
            c.enabled = not c.enabled
    return genome


def activation_mutation(
    genome: NeuralGenome,
    mutation_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> NeuralGenome:
    """Mutation 5: Change activation function of hidden neurons."""
    rng = rng or np.random.default_rng()
    for n in genome.neurons.values():
        if n.neuron_type == "hidden" and rng.random() < mutation_rate:
            n.activation = rng.choice(_ACTIVATIONS)
    return genome


# ── NEAT Breeder ──────────────────────────────────────────────

class NEATBreeder:
    """NEAT-style topology-and-weight breeder for neural network agents.

    Maintains a population of NeuralGenomes, speciates them by topology
    similarity, and applies mutation operators with per-species fitness sharing.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        population_size: int = 100,
        compatibility_threshold: float = 3.0,
        c1: float = 1.0,  # excess gene weight
        c2: float = 1.0,  # disjoint gene weight
        c3: float = 0.4,  # weight difference weight
        rng: np.random.Generator | None = None,
    ) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.compatibility_threshold = compatibility_threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.rng = rng or np.random.default_rng()
        self._population: list[NeuralGenome] = []
        self._species: dict[int, list[NeuralGenome]] = {}
        self._next_species_id = 0
        self._innovation_counter = 0
        self._generation = 0
        self._initialize_population()

    def _initialize_population(self) -> None:
        """Create initial population with minimal topology."""
        for _ in range(self.population_size):
            g = NeuralGenome()
            g.num_inputs = self.num_inputs
            g.num_outputs = self.num_outputs

            # Input neurons
            for i in range(self.num_inputs):
                n = g.add_neuron(neuron_type="input", activation="linear")
                n.id = i  # fixed IDs for inputs
            g.next_neuron_id = self.num_inputs

            # Output neurons
            for i in range(self.num_outputs):
                n = g.add_neuron(neuron_type="output", activation="tanh")
                n.id = self.num_inputs + i
            g.next_neuron_id = self.num_inputs + self.num_outputs

            # Initial connections (fully connected)
            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    g.add_connection(i, self.num_inputs + j, weight=self.rng.normal(0.0, 1.0))

            self._population.append(g)

    # ── compatibility distance ────────────────────────

    def compatibility_distance(self, a: NeuralGenome, b: NeuralGenome) -> float:
        """NEAT compatibility distance: topology + weight difference."""
        innovations_a = {c.innovation: c for c in a.connections.values()}
        innovations_b = {c.innovation: c for c in b.connections.values()}

        all_innovations = sorted(set(innovations_a.keys()) | set(innovations_b.keys()))
        if not all_innovations:
            return 0.0

        max_innov = max(all_innovations)
        excess = 0
        disjoint = 0
        weight_diff = 0.0
        matching = 0

        for inv in all_innovations:
            in_a = inv in innovations_a
            in_b = inv in innovations_b
            if in_a and in_b:
                matching += 1
                weight_diff += abs(innovations_a[inv].weight - innovations_b[inv].weight)
            elif inv > max_innov * 0.8:
                excess += 1
            else:
                disjoint += 1

        n = max(len(a.connections), len(b.connections))
        n = max(n, 1)

        distance = (self.c1 * excess + self.c2 * disjoint) / n
        if matching > 0:
            distance += self.c3 * (weight_diff / matching)

        return distance

    # ── speciation ────────────────────────────────────

    def _speciate(self) -> None:
        """Assign genomes to species based on compatibility distance."""
        # Clear species
        for sp in self._species.values():
            sp.clear()

        for genome in self._population:
            assigned = False
            for sid, members in self._species.items():
                if members:
                    rep = members[0]
                    if self.compatibility_distance(genome, rep) < self.compatibility_threshold:
                        members.append(genome)
                        genome.species_id = sid
                        assigned = True
                        break
            if not assigned:
                sid = self._next_species_id
                self._next_species_id += 1
                self._species[sid] = [genome]
                genome.species_id = sid

    # ── breeding ──────────────────────────────────────

    def _breed_species(self, members: list[NeuralGenome]) -> list[NeuralGenome]:
        """Create offspring for a species."""
        if not members:
            return []

        # Sort by fitness descending
        sorted_members = sorted(members, key=lambda g: g.fitness, reverse=True)
        elite_count = max(1, len(sorted_members) // 5)
        offspring: list[NeuralGenome] = []

        # Preserve elites
        for i in range(elite_count):
            offspring.append(sorted_members[i].copy())

        # Generate rest via mutation/crossover
        while len(offspring) < len(members):
            parent = self.rng.choice(sorted_members[:max(2, len(sorted_members) // 2)])
            child = parent.copy()

            # Apply mutations
            if self.rng.random() < 0.03:
                child = add_node_mutation(child, self.rng)
            if self.rng.random() < 0.05:
                child = add_connection_mutation(child, self.rng)
            child = weight_mutation(child, rng=self.rng)
            if self.rng.random() < 0.01:
                child = toggle_mutation(child, rng=self.rng)
            if self.rng.random() < 0.1:
                child = activation_mutation(child, rng=self.rng)

            offspring.append(child)

        return offspring

    def evolve(self, fitness_fn: Any) -> list[NeuralGenome]:
        """One generation: evaluate, speciate, breed.

        fitness_fn: callable that takes a NeuralGenome and returns a float
        """
        # Evaluate fitness
        for genome in self._population:
            genome.fitness = float(fitness_fn(genome))

        # Speciate
        self._speciate()

        # Fitness sharing within species
        for sid, members in self._species.items():
            if members:
                shared = sum(g.fitness for g in members) / len(members)
                for g in members:
                    g.fitness = shared

        # Breed each species
        new_population: list[NeuralGenome] = []
        for members in self._species.values():
            new_population.extend(self._breed_species(members))

        # Trim to population size
        if len(new_population) > self.population_size:
            new_population = new_population[:self.population_size]
        elif len(new_population) < self.population_size:
            # Fill with random mutations of best genome
            best = max(self._population, key=lambda g: g.fitness)
            while len(new_population) < self.population_size:
                child = best.copy()
                child = weight_mutation(child, rng=self.rng)
                new_population.append(child)

        self._population = new_population
        self._generation += 1
        return self._population

    @property
    def best_genome(self) -> NeuralGenome:
        return max(self._population, key=lambda g: g.fitness)

    @property
    def generation(self) -> int:
        return self._generation

    def report(self) -> dict[str, Any]:
        fitnesses = [g.fitness for g in self._population]
        return {
            "generation": self._generation,
            "population": len(self._population),
            "species": len(self._species),
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "mean_fitness": np.mean(fitnesses) if fitnesses else 0.0,
            "mean_connections": np.mean([len(g.connections) for g in self._population]),
            "mean_neurons": np.mean([len(g.neurons) for g in self._population]),
        }
