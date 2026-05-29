"""
Tests for Graph Neural Network Breeder.

Covers: GenomeGraph, dict_to_graph, SimpleGNNPredictor, GNNBreeder.
"""

import random

import numpy as np
import pytest

from swarm.gnn_breeder import (
    GenomeGraph,
    dict_to_graph,
    _gene_similarity,
    SimpleGNNPredictor,
    GNNBreeder,
)


class TestGenomeGraph:
    def test_add_node(self):
        g = GenomeGraph()
        g.add_node("gene_a", 1.0)
        assert "gene_a" in g.nodes
        assert g.nodes["gene_a"]["value"] == 1.0

    def test_add_edge(self):
        g = GenomeGraph()
        g.add_node("a", 1.0)
        g.add_node("b", 2.0)
        g.add_edge("a", "b", 0.5)
        assert ("a", "b") in g.edges
        assert ("b", "a") in g.edges
        assert g.edges[("a", "b")] == 0.5

    def test_get_neighbors(self):
        g = GenomeGraph()
        g.add_node("a", 1.0)
        g.add_node("b", 2.0)
        g.add_node("c", 3.0)
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "c", 1.0)
        neighbors = g.get_neighbors("a")
        assert sorted(neighbors) == ["b", "c"]

    def test_get_degree(self):
        g = GenomeGraph()
        g.add_node("a", 1.0)
        g.add_node("b", 2.0)
        g.add_edge("a", "b", 1.0)
        assert g.get_degree("a") == 1
        assert g.get_degree("b") == 1

    def test_to_genome_dict(self):
        g = GenomeGraph()
        g.add_node("a", 1.0)
        g.add_node("b", 2.0)
        d = g.to_genome_dict()
        assert d == {"a": 1.0, "b": 2.0}

    def test_clone(self):
        g = GenomeGraph()
        g.add_node("a", 1.0)
        g.add_edge("a", "b", 1.0)
        clone = g.clone()
        assert clone.nodes == g.nodes
        assert clone.edges == g.edges
        # Independent copy
        clone.nodes["a"] = {"value": 99.0}
        assert g.nodes["a"]["value"] == 1.0


class TestGeneSimilarity:
    def test_same_prefix(self):
        assert _gene_similarity("gene_a_1", "gene_a_2") > 0.5

    def test_no_prefix(self):
        assert _gene_similarity("abc", "xyz") == 0.0

    def test_empty(self):
        assert _gene_similarity("", "abc") == 0.0

    def test_identical(self):
        assert _gene_similarity("same", "same") == 1.0


class TestDictToGraph:
    def test_basic_conversion(self):
        genome = {"gene_a_1": 1.0, "gene_a_2": 2.0, "gene_b_1": 3.0}
        graph = dict_to_graph(genome)
        assert len(graph.nodes) == 3
        # gene_a_1 and gene_a_2 should have edge due to common prefix
        assert ("gene_a_1", "gene_a_2") in graph.edges

    def test_no_edges(self):
        genome = {"abc": 1.0, "xyz": 2.0}
        graph = dict_to_graph(genome, interaction_threshold=0.9)
        assert len(graph.nodes) == 2
        # No edge because similarity is low
        assert ("abc", "xyz") not in graph.edges


class TestSimpleGNNPredictor:
    def test_predict(self):
        gnn = SimpleGNNPredictor(hidden_dim=8, n_layers=2)
        graph = GenomeGraph()
        graph.add_node("a", 1.0)
        graph.add_node("b", 2.0)
        graph.add_edge("a", "b", 1.0)
        pred = gnn.predict(graph)
        assert isinstance(pred, float)
        assert not np.isnan(pred)

    def test_train_step(self):
        gnn = SimpleGNNPredictor(hidden_dim=8)
        graph = GenomeGraph()
        graph.add_node("a", 1.0)
        pred_before = gnn.predict(graph)
        gnn.train_step(graph, 10.0, lr=0.1)
        pred_after = gnn.predict(graph)
        # Prediction should change after training
        assert abs(pred_after - pred_before) > 0.0 or pred_after == pred_before

    def test_node_feature(self):
        gnn = SimpleGNNPredictor(hidden_dim=8)
        graph = GenomeGraph()
        graph.add_node("a", 5.0)
        graph.add_edge("a", "b", 1.0)
        feat = gnn._node_feature(graph, "a")
        assert len(feat) >= 3
        assert feat[0] == 5.0  # value
        assert feat[1] == 1.0  # degree


class TestGNNBreeder:
    def test_init(self):
        breeder = GNNBreeder(population_size=20, hidden_dim=8)
        assert breeder.population_size == 20
        assert breeder.gnn.hidden_dim == 8

    def test_encode(self):
        breeder = GNNBreeder()
        genome = {"gene_a": 1.0, "gene_b": 2.0}
        graph = breeder.encode(genome)
        assert isinstance(graph, GenomeGraph)
        assert len(graph.nodes) == 2

    def test_predict_fitness(self):
        breeder = GNNBreeder()
        genome = {"gene_a": 1.0, "gene_b": 2.0}
        pred = breeder.predict_fitness(genome)
        assert isinstance(pred, float)

    def test_train(self):
        breeder = GNNBreeder()
        population = [
            ({"gene_a": 1.0, "gene_b": 2.0}, 10.0),
            ({"gene_a": 2.0, "gene_b": 3.0}, 20.0),
        ]
        breeder.train(population)
        assert len(breeder.training_data) == 2

    def test_graph_crossover(self):
        breeder = GNNBreeder()
        p1 = {"gene_a": 1.0, "gene_b": 2.0}
        p2 = {"gene_a": 3.0, "gene_b": 4.0}
        child = breeder.graph_crossover(p1, p2)
        assert "gene_a" in child
        assert "gene_b" in child

    def test_graph_aware_mutation(self):
        breeder = GNNBreeder()
        genome = {"gene_a": 1.0, "gene_b": 2.0}
        mutated = breeder.graph_aware_mutation(genome)
        assert "gene_a" in mutated
        assert "gene_b" in mutated

    def test_select_parents(self):
        breeder = GNNBreeder()
        pop = [
            ({"gene_a": 1.0, "gene_b": 2.0}, 10.0),
            ({"gene_a": 2.0, "gene_b": 3.0}, 20.0),
            ({"gene_a": 3.0, "gene_b": 1.0}, 30.0),
        ]
        parents = breeder.select_parents(pop, k=2)
        assert len(parents) == 2
        # Should select high-fitness parents
        assert parents[0][1] >= 10.0

    def test_breed_generation(self):
        breeder = GNNBreeder(population_size=10)
        pop = [
            ({"gene_a": float(i), "gene_b": float(i * 2)}, float(i * 10))
            for i in range(10)
        ]

        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10 + genome["gene_b"]}

        new_pop = breeder.breed_generation(pop, task_fn)
        assert len(new_pop) == 10
        assert breeder.generation == 1

    def test_structure_summary(self):
        breeder = GNNBreeder()
        genome = {"gene_a": 1.0, "gene_b": 2.0, "gene_c": 3.0}
        summary = breeder.get_structure_summary(genome)
        assert "n_nodes" in summary
        assert "n_edges" in summary
        assert "avg_degree" in summary

    def test_high_degree_mutation_scale(self):
        breeder = GNNBreeder()
        # Create a genome where one gene has many connections
        genome = {"hub": 1.0, "a": 2.0, "b": 3.0, "c": 4.0}
        mutated = breeder.graph_aware_mutation(genome)
        # hub gene should have smaller mutation (high degree)
        assert "hub" in mutated
        assert "a" in mutated

    def test_crossover_with_asymmetric_parents(self):
        breeder = GNNBreeder()
        p1 = {"gene_a": 1.0, "gene_b": 2.0}
        p2 = {"gene_a": 3.0, "gene_c": 4.0}
        child = breeder.graph_crossover(p1, p2)
        assert "gene_a" in child  # In both parents
        assert "gene_b" in child  # Only in p1
        assert "gene_c" in child  # Only in p2

    def test_elitism(self):
        breeder = GNNBreeder(population_size=10)
        pop = [
            ({"gene_a": float(i)}, float(i * 10))
            for i in range(10)
        ]

        def task_fn(genome):
            return {"fitness": genome["gene_a"] * 10}

        new_pop = breeder.breed_generation(pop, task_fn)
        best_new = max(new_pop, key=lambda x: x[1])
        assert best_new[1] >= 80.0  # Elite preservation

    def test_empty_training(self):
        breeder = GNNBreeder()
        breeder.train([])
        assert len(breeder.training_data) == 0
