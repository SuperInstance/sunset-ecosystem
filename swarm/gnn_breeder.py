"""
Graph Neural Network Breeder

Represents genomes as graph structures where:
- Nodes are genes (or gene groups)
- Edges represent interactions (epistasis, correlation, etc.)

Uses a Graph Neural Network (GNN) to predict offspring fitness
from parent graphs, enabling structure-aware breeding.

Key innovations:
- Genome-to-graph encoding
- GNN-based fitness prediction
- Graph crossover: merge parent graphs while preserving topology
- Edge-aware mutation: perturb gene interactions, not just values

References:
- Scarselli et al. (2008) - Graph Neural Network
- Kipf & Welling (2016) - Semi-Supervised Classification with GCNs
- Xu et al. (2018) - How Powerful are Graph Neural Networks?
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class GenomeGraph:
    """A graph representation of a genome."""

    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # edges: (source, target) -> weight
    edges: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # Node features: gene_name -> feature vector
    node_features: Dict[str, np.ndarray] = field(default_factory=dict)

    def add_node(self, name: str, value: float, feature: Optional[np.ndarray] = None):
        self.nodes[name] = {"value": value}
        if feature is not None:
            self.node_features[name] = feature

    def add_edge(self, source: str, target: str, weight: float = 1.0):
        self.edges[(source, target)] = weight
        # Also add reverse edge for undirected graph
        self.edges[(target, source)] = weight

    def get_neighbors(self, node: str) -> List[str]:
        """Get all neighbors of a node."""
        return [t for (s, t) in self.edges if s == node and t != node]

    def get_degree(self, node: str) -> int:
        return len(self.get_neighbors(node))

    def to_genome_dict(self) -> Dict[str, float]:
        """Convert back to simple genome dictionary."""
        return {name: data["value"] for name, data in self.nodes.items()}

    def clone(self) -> "GenomeGraph":
        return GenomeGraph(
            nodes={k: v.copy() for k, v in self.nodes.items()},
            edges=self.edges.copy(),
            node_features={k: v.copy() for k, v in self.node_features.items()},
        )


def dict_to_graph(
    genome: Dict[str, float], interaction_threshold: float = 0.5
) -> GenomeGraph:
    """
    Convert a genome dictionary to a GenomeGraph.
    Creates edges based on gene name similarity and values.
    """
    graph = GenomeGraph()

    # Add all nodes
    for name, value in genome.items():
        graph.add_node(name, value)

    # Create edges based on naming patterns (e.g., "gene_a_1" and "gene_a_2" are related)
    names = list(genome.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i + 1 :]:
            # Compute similarity: common prefix / common function patterns
            similarity = _gene_similarity(name1, name2)
            if similarity > interaction_threshold:
                weight = similarity
                graph.add_edge(name1, name2, weight)

    return graph


def _gene_similarity(name1: str, name2: str) -> float:
    """Compute similarity between two gene names."""
    # Common prefix length
    min_len = min(len(name1), len(name2))
    common_prefix = 0
    for i in range(min_len):
        if name1[i] == name2[i]:
            common_prefix += 1
        else:
            break

    if min_len == 0:
        return 0.0
    return common_prefix / min_len


class SimpleGNNPredictor:
    """
    Simplified Graph Neural Network for fitness prediction.

    Uses message passing: each node's feature is updated by
    aggregating features from neighbors.
    """

    def __init__(self, hidden_dim: int = 8, n_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Learnable weights (random initialization for now)
        self.W_msg = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_readout = np.random.randn(hidden_dim, 1) * 0.01
        self.b_readout = np.zeros(1)

    def _node_feature(self, graph: GenomeGraph, node: str) -> np.ndarray:
        """Get initial feature for a node."""
        if node in graph.node_features:
            return graph.node_features[node]
        # Default feature: [value, degree, random]
        value = graph.nodes[node]["value"]
        degree = graph.get_degree(node)
        return np.array([value, float(degree), random.random()])

    def predict(self, graph: GenomeGraph) -> float:
        """
        Predict fitness from genome graph.
        Returns scalar fitness prediction.
        """
        # Initialize node features
        features = {}
        for node in graph.nodes:
            feat = self._node_feature(graph, node)
            # Pad to hidden_dim
            padded = np.zeros(self.hidden_dim)
            padded[: min(len(feat), self.hidden_dim)] = feat[
                : min(len(feat), self.hidden_dim)
            ]
            features[node] = padded

        # Message passing
        for _ in range(self.n_layers):
            new_features = {}
            for node in graph.nodes:
                # Aggregate from neighbors
                neighbors = graph.get_neighbors(node)
                if not neighbors:
                    new_features[node] = features[node]
                    continue

                agg = np.zeros(self.hidden_dim)
                for neighbor in neighbors:
                    weight = graph.edges.get((node, neighbor), 1.0)
                    agg += weight * features[neighbor]
                agg = agg / max(1, len(neighbors))

                # Update: combine self + aggregated
                combined = features[node] + self.W_msg @ agg
                # Simple nonlinearity: ReLU
                combined = np.maximum(0, combined)
                new_features[node] = combined

            features = new_features

        # Readout: sum all node features, then predict
        total = sum(features.values())
        prediction = self.W_readout.T @ total + self.b_readout
        return float(prediction[0])

    def train_step(self, graph: GenomeGraph, true_fitness: float, lr: float = 0.01):
        """Single training step using gradient descent."""
        pred = self.predict(graph)
        error = pred - true_fitness

        # Simplified gradient update (would be proper backprop in real GNN)
        # Just adjust readout weights
        features = {}
        for node in graph.nodes:
            feat = self._node_feature(graph, node)
            padded = np.zeros(self.hidden_dim)
            padded[: min(len(feat), self.hidden_dim)] = feat[
                : min(len(feat), self.hidden_dim)
            ]
            features[node] = padded

        total = sum(features.values())
        grad = error * total
        self.W_readout -= lr * grad.reshape(self.hidden_dim, 1)
        self.b_readout -= lr * error


class GNNBreeder:
    """
    Breeding daemon that uses GNN fitness prediction.

    1. Encodes genomes as graphs
    2. Trains GNN to predict fitness from graph structure
    3. Uses GNN predictions to guide parent selection
    4. Graph crossover preserves topological relationships
    """

    def __init__(
        self,
        population_size: int = 50,
        hidden_dim: int = 8,
        prediction_weight: float = 0.3,
    ):
        self.population_size = population_size
        self.gnn = SimpleGNNPredictor(hidden_dim=hidden_dim)
        self.prediction_weight = prediction_weight
        self.generation = 0
        self.training_data: List[Tuple[GenomeGraph, float]] = []

    def encode(self, genome: Dict[str, float]) -> GenomeGraph:
        """Encode genome as graph."""
        return dict_to_graph(genome)

    def predict_fitness(self, genome: Dict[str, float]) -> float:
        """Predict fitness using GNN."""
        graph = self.encode(genome)
        return self.gnn.predict(graph)

    def train(self, population: List[Tuple[Dict[str, float], float]]):
        """Train GNN on current population."""
        # Add to training data
        for genome, fitness in population:
            graph = self.encode(genome)
            self.training_data.append((graph, fitness))

        # Train on last 100 examples
        recent = self.training_data[-100:]
        for graph, fitness in recent:
            self.gnn.train_step(graph, fitness)

    def graph_crossover(
        self, parent1: Dict[str, float], parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Crossover that preserves graph structure.

        Strategy: Take subgraphs from each parent and merge,
        preferring to keep edges within subgraphs.
        """
        g1 = self.encode(parent1)
        g2 = self.encode(parent2)

        # Determine which nodes come from which parent
        child_nodes = {}
        for node in set(list(g1.nodes.keys()) + list(g2.nodes.keys())):
            if node in g1.nodes and node in g2.nodes:
                # Node in both: pick from parent with higher predicted fitness
                p1_pred = self.gnn.predict(g1)
                p2_pred = self.gnn.predict(g2)
                if p1_pred > p2_pred:
                    child_nodes[node] = g1.nodes[node]["value"]
                else:
                    child_nodes[node] = g2.nodes[node]["value"]
            elif node in g1.nodes:
                child_nodes[node] = g1.nodes[node]["value"]
            else:
                child_nodes[node] = g2.nodes[node]["value"]

        return child_nodes

    def graph_aware_mutation(self, genome: Dict[str, float]) -> Dict[str, float]:
        """
        Mutation that considers graph structure.

        High-degree nodes (many interactions) get smaller mutations
        (they're more constrained). Low-degree nodes get larger mutations.
        """
        graph = self.encode(genome)
        mutated = genome.copy()

        for node in graph.nodes:
            if node not in mutated:
                continue

            degree = graph.get_degree(node)
            # High-degree nodes are more constrained -> smaller mutation
            if degree > 2:
                scale = 0.05
            elif degree == 1:
                scale = 0.15
            else:
                scale = 0.1

            mutated[node] *= 1 + random.gauss(0, scale)

        return mutated

    def select_parents(
        self, population: List[Tuple[Dict, float]], k: int = 2
    ) -> List[Tuple[Dict, float]]:
        """Select parents using GNN predictions."""
        if len(population) < 2:
            return population[:k]

        # Score = actual_fitness + prediction_weight * gnn_prediction
        scored = []
        for genome, fitness in population:
            pred = self.predict_fitness(genome)
            score = fitness + self.prediction_weight * pred
            scored.append((genome, fitness, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(g, f) for g, f, _ in scored[:k]]

    def breed_generation(
        self,
        population: List[Tuple[Dict[str, float], float]],
        task_fn: Callable[[Dict[str, float]], Any],
    ) -> List[Tuple[Dict[str, float], float]]:
        """Run one generation of GNN-guided breeding."""
        self.generation += 1

        # Train GNN on current population
        self.train(population)

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)

        # Elitism
        n_elite = max(1, len(sorted_pop) // 10)
        new_pop = sorted_pop[:n_elite]

        # Fill rest
        while len(new_pop) < self.population_size:
            parents = self.select_parents(sorted_pop, k=2)
            if len(parents) < 2:
                break

            child = self.graph_crossover(parents[0][0], parents[1][0])
            child = self.graph_aware_mutation(child)

            result = task_fn(child)
            fitness = (
                result.get("fitness", 0.0)
                if isinstance(result, dict)
                else float(result)
            )
            new_pop.append((child, fitness))

        return new_pop

    def get_structure_summary(self, genome: Dict[str, float]) -> Dict:
        """Analyze the graph structure of a genome."""
        graph = self.encode(genome)
        degrees = {node: graph.get_degree(node) for node in graph.nodes}
        avg_degree = np.mean(list(degrees.values())) if degrees else 0
        max_degree = max(degrees.values()) if degrees else 0

        return {
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges) // 2,  # Undirected, so divide by 2
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "hub_nodes": [n for n, d in degrees.items() if d > avg_degree * 2],
        }
