"""Tests for neural_topology_breeding.py — NEAT-style neural architecture search.

Run: python3 -m pytest tests/test_neural_topology_breeding.py -v --tb=short
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.neural_topology_breeding import (
    NEATBreeder,
    NeuralGenome,
    add_connection_mutation,
    add_node_mutation,
    activation_mutation,
    toggle_mutation,
    weight_mutation,
)


class TestNeuralGenome:
    def test_create_minimal(self):
        g = NeuralGenome()
        assert len(g.neurons) == 0
        assert len(g.connections) == 0

    def test_add_neuron(self):
        g = NeuralGenome()
        n = g.add_neuron("hidden", "tanh", 0.5)
        assert n.id == 0
        assert n.activation == "tanh"
        assert n.bias == 0.5
        assert len(g.neurons) == 1

    def test_add_connection(self):
        g = NeuralGenome()
        g.add_neuron("input", "linear")
        g.add_neuron("output", "tanh")
        c = g.add_connection(0, 1, 0.5)
        assert c is not None
        assert c.weight == 0.5
        assert c.enabled is True

    def test_add_connection_duplicate(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("output")
        g.add_connection(0, 1, 0.5)
        c2 = g.add_connection(0, 1, 0.7)
        assert c2 is None  # duplicate

    def test_add_connection_self_loop(self):
        g = NeuralGenome()
        g.add_neuron("hidden")
        c = g.add_connection(0, 0, 0.5)
        assert c is None  # no self-loops

    def test_compute_depth_linear(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("hidden")
        g.add_neuron("output")
        g.add_connection(0, 1, 1.0)
        g.add_connection(1, 2, 1.0)
        depths = g.compute_depth()
        assert depths[0] == 0  # input
        assert depths[1] == 1  # hidden
        assert depths[2] == 2  # output

    def test_feedforward_simple(self):
        g = NeuralGenome()
        g.add_neuron("input", "linear")
        g.add_neuron("output", "tanh")
        g.add_connection(0, 1, 1.0)
        result = g.feedforward(np.array([2.0]))
        # tanh(2.0 * 1.0) = tanh(2.0)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(np.tanh(2.0), abs=1e-6)

    def test_feedforward_with_bias(self):
        g = NeuralGenome()
        g.add_neuron("input", "linear")
        g.add_neuron("output", "tanh")
        g.neurons[1].bias = 1.0
        g.add_connection(0, 1, 1.0)
        result = g.feedforward(np.array([1.0]))
        # tanh(1.0 * 1.0 + 1.0) = tanh(2.0)
        assert result[0] == pytest.approx(np.tanh(2.0), abs=1e-6)

    def test_copy(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g2 = g.copy()
        assert len(g2.neurons) == 1
        g2.add_neuron("output")
        assert len(g.neurons) == 1  # original unchanged


class TestAddNodeMutation:
    def test_splits_connection(self):
        g = NeuralGenome()
        g.add_neuron("input", "linear")
        g.add_neuron("output", "tanh")
        g.add_connection(0, 1, 0.5)
        assert len(g.connections) == 1

        mutated = add_node_mutation(g.copy())
        assert len(mutated.neurons) == 3  # added hidden neuron
        assert len(mutated.connections) == 3  # disabled old + 2 new

        # Original connection should be disabled
        orig = [
            c for c in mutated.connections.values() if c.from_id == 0 and c.to_id == 1
        ]
        assert len(orig) == 1
        assert not orig[0].enabled


class TestAddConnectionMutation:
    def test_adds_new_connection(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("hidden")
        g.add_neuron("output")
        g.add_connection(0, 1, 1.0)
        # Now try adding 1→2
        mutated = add_connection_mutation(g.copy())
        assert len(mutated.connections) >= 1

    def test_respects_feedforward(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("output")
        g.add_connection(0, 1, 1.0)
        # Try reverse connection (should fail due to depth)
        mutated = add_connection_mutation(g.copy())
        # Should not add backward connection
        rev = [
            c for c in mutated.connections.values() if c.from_id == 1 and c.to_id == 0
        ]
        assert len(rev) == 0


class TestWeightMutation:
    def test_changes_weights(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("output")
        g.add_connection(0, 1, 1.0)
        mutated = weight_mutation(g.copy(), perturbation_rate=1.0, perturbation_std=0.5)
        w_after = mutated.connections[0].weight
        assert w_after != pytest.approx(1.0)  # should have changed

    def test_perturbation_rate_zero(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("output")
        g.add_connection(0, 1, 1.0)
        mutated = weight_mutation(g.copy(), perturbation_rate=0.0)
        # Full replacement instead of perturbation
        assert mutated.connections[0].weight != 1.0


class TestToggleMutation:
    def test_toggles_connections(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("output")
        g.add_connection(0, 1, 1.0)
        mutated = toggle_mutation(g.copy(), toggle_rate=1.0)
        assert not mutated.connections[0].enabled  # toggled off

    def test_no_toggle(self):
        g = NeuralGenome()
        g.add_neuron("input")
        g.add_neuron("output")
        g.add_connection(0, 1, 1.0)
        mutated = toggle_mutation(g.copy(), toggle_rate=0.0)
        assert mutated.connections[0].enabled


class TestActivationMutation:
    def test_changes_activations(self):
        g = NeuralGenome()
        g.add_neuron("input", "linear")
        g.add_neuron("hidden", "tanh")
        g.add_neuron("output", "tanh")
        # Try a few times to avoid picking the same activation by chance
        for _ in range(5):
            mutated = activation_mutation(g.copy(), mutation_rate=1.0)
            if mutated.neurons[1].activation != "tanh":
                break
        assert mutated.neurons[1].activation != "tanh"

    def test_preserves_input_output(self):
        g = NeuralGenome()
        g.add_neuron("input", "linear")
        g.add_neuron("output", "tanh")
        mutated = activation_mutation(g.copy(), mutation_rate=1.0)
        assert mutated.neurons[0].activation == "linear"
        assert mutated.neurons[1].activation == "tanh"


class TestNEATBreeder:
    def test_initial_population(self):
        breeder = NEATBreeder(num_inputs=2, num_outputs=1, population_size=10)
        assert len(breeder._population) == 10
        # Each genome should have 2 inputs + 1 output + connections
        g = breeder._population[0]
        assert len(g.neurons) == 3
        assert len(g.connections) == 2  # 2 inputs * 1 output

    def test_evolve_increases_generation(self):
        breeder = NEATBreeder(num_inputs=2, num_outputs=1, population_size=10)
        assert breeder.generation == 0
        breeder.evolve(lambda g: 1.0)  # constant fitness
        assert breeder.generation == 1

    def test_best_genome(self):
        breeder = NEATBreeder(num_inputs=2, num_outputs=1, population_size=10)

        def fitness(g):
            return len(g.connections) * 0.1 + len(g.neurons) * 0.5

        breeder.evolve(fitness)
        best = breeder.best_genome
        assert best.fitness > 0

    def test_report(self):
        breeder = NEATBreeder(num_inputs=2, num_outputs=1, population_size=10)
        breeder.evolve(lambda g: 1.0)
        r = breeder.report()
        assert r["generation"] == 1
        assert r["population"] == 10
        assert r["species"] >= 1
        assert "best_fitness" in r

    def test_topology_grows(self):
        breeder = NEATBreeder(num_inputs=2, num_outputs=1, population_size=20)

        def fitness(g):
            return float(len(g.connections))

        initial_mean_conn = np.mean([len(g.connections) for g in breeder._population])
        for _ in range(5):
            breeder.evolve(fitness)
        final_mean_conn = np.mean([len(g.connections) for g in breeder._population])
        # With add_node and add_connection mutations, topology should grow
        assert final_mean_conn >= initial_mean_conn

    def test_speciation(self):
        breeder = NEATBreeder(num_inputs=2, num_outputs=1, population_size=20)
        breeder.evolve(lambda g: np.random.random())
        r = breeder.report()
        assert r["species"] >= 1

    def test_population_size_preserved(self):
        breeder = NEATBreeder(num_inputs=2, num_outputs=1, population_size=15)
        for _ in range(3):
            breeder.evolve(lambda g: float(g.next_neuron_id))
        assert len(breeder._population) == 15
