"""
Information-Theoretic Breeder

Uses information theory — entropy, mutual information, and KL divergence —
to guide breeding decisions. Explicitly maximizes population diversity
while driving toward high fitness regions.

Key innovations:
- Entropy-driven parent selection: maximize expected offspring entropy
- Mutual information crossover: transfer information, not just genes
- KL divergence mutation: perturb in directions that increase information gain
- Population entropy as a diversity metric (orthogonal to fitness)

References:
- Shannon (1948) - A Mathematical Theory of Communication
- Cover & Thomas (2006) - Elements of Information Theory
- Polani et al. (2001) - Information theory in sensorimotor loops
- Baluja (1997) - Population-based incremental learning
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class GenomeDistribution:
    """Represents a distribution over a genome's gene values."""

    mean: float
    std: float
    # Histogram representation for discrete genes
    histogram: Optional[Dict[float, int]] = None


def shannon_entropy(values: np.ndarray, bins: int = 10) -> float:
    """
    Compute Shannon entropy of a set of values.
    H(X) = -Σ p(x) log p(x)
    """
    if len(values) == 0:
        return 0.0

    # Use histogram to estimate probabilities
    hist, _ = np.histogram(values, bins=bins, range=(values.min(), values.max()))
    probs = hist / len(values)
    probs = probs[probs > 0]  # Remove zero probabilities

    if len(probs) == 0:
        return 0.0

    return -np.sum(probs * np.log2(probs))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """
    Compute mutual information between two variables.
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0

    # Joint histogram
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    joint_probs = hist_2d / np.sum(hist_2d)
    joint_probs = joint_probs[joint_probs > 0]

    # Marginal histograms
    hist_x, _ = np.histogram(x, bins=bins)
    hist_y, _ = np.histogram(y, bins=bins)
    px = hist_x / len(x)
    py = hist_y / len(y)
    px = px[px > 0]
    py = py[py > 0]

    h_x = -np.sum(px * np.log2(px))
    h_y = -np.sum(py * np.log2(py))
    h_xy = -np.sum(joint_probs * np.log2(joint_probs))

    return h_x + h_y - h_xy


def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
    """
    Compute KL divergence D_KL(P || Q).
    """
    hist_p, _ = np.histogram(p, bins=bins)
    hist_q, _ = np.histogram(q, bins=bins)

    # Normalize
    p_probs = hist_p / np.sum(hist_p)
    q_probs = hist_q / np.sum(hist_q)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_probs = p_probs + epsilon
    q_probs = q_probs + epsilon

    # Renormalize
    p_probs = p_probs / np.sum(p_probs)
    q_probs = q_probs / np.sum(q_probs)

    # KL divergence
    kl = np.sum(p_probs * np.log2(p_probs / q_probs))
    return max(0.0, kl)


def js_divergence(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded).
    JS(P,Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    # Ensure same length for computation
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, bins) + 0.5 * kl_divergence(q, m, bins)


@dataclass
class InformationState:
    """Tracks information-theoretic state of the population."""

    population_entropy: float = 0.0
    fitness_entropy: float = 0.0
    # Per-gene entropies
    gene_entropies: Dict[str, float] = field(default_factory=dict)
    # Mutual information between genes and fitness
    gene_fitness_mi: Dict[str, float] = field(default_factory=dict)
    # Generation count
    generation: int = 0


def compute_population_info_state(
    population: List[Tuple[Dict[str, float], float]], gene_names: List[str]
) -> InformationState:
    """
    Compute information-theoretic state of a population.
    """
    if not population:
        return InformationState()

    # Extract fitness values
    fitnesses = np.array([fitness for _, fitness in population])

    # Population entropy (over all genes combined)
    all_gene_values = []
    for gene_name in gene_names:
        values = np.array([genome.get(gene_name, 0.0) for genome, _ in population])
        all_gene_values.extend(values)

    pop_entropy = shannon_entropy(np.array(all_gene_values))
    fit_entropy = shannon_entropy(fitnesses)

    # Per-gene entropies and MI with fitness
    gene_entropies = {}
    gene_fitness_mi = {}

    for gene_name in gene_names:
        values = np.array([genome.get(gene_name, 0.0) for genome, _ in population])
        gene_entropies[gene_name] = shannon_entropy(values)
        gene_fitness_mi[gene_name] = mutual_information(values, fitnesses)

    return InformationState(
        population_entropy=pop_entropy,
        fitness_entropy=fit_entropy,
        gene_entropies=gene_entropies,
        gene_fitness_mi=gene_fitness_mi,
    )


class InfoTheoreticBreeder:
    """
    Breeding daemon guided by information theory.

    Core principle: breeding should maximize the mutual information
    between the population and the fitness landscape, while maintaining
    sufficient entropy to avoid premature convergence.

    Strategy:
    1. Measure population entropy (diversity check)
    2. Identify genes with high mutual information to fitness (key genes)
    3. Preserve entropy in low-MI genes (diversity reservoir)
    4. Exploit high-MI genes with directed mutations
    5. Crossover maximizes offspring information gain
    """

    def __init__(
        self,
        population_size: int = 50,
        entropy_target: float = 2.0,
        mi_threshold: float = 0.1,
        elitism_ratio: float = 0.1,
    ):
        self.population_size = population_size
        self.entropy_target = entropy_target
        self.mi_threshold = mi_threshold
        self.elitism_ratio = elitism_ratio

        self.info_state: Optional[InformationState] = None
        self.generation = 0

    def analyze_population(
        self, population: List[Tuple[Dict[str, float], float]]
    ) -> InformationState:
        """Analyze population information-theoretic state."""
        if not population:
            return InformationState()

        gene_names = list(population[0][0].keys())
        self.info_state = compute_population_info_state(population, gene_names)
        self.info_state.generation = self.generation
        return self.info_state

    def select_parents(
        self, population: List[Tuple[Dict[str, float], float]], k: int = 2
    ) -> List[Tuple[Dict[str, float], float]]:
        """
        Select parents that maximize expected offspring information.

        Strategy: Choose parents with high fitness AND high mutual
        information divergence between them. This ensures offspring
        inherit diverse information from both parents.
        """
        if len(population) < 2:
            return population[:k]

        # Tournament selection with information bonus
        selected = []
        for _ in range(k):
            tournament_size = min(5, len(population))
            tournament = random.sample(population, tournament_size)

            # Score = fitness + information diversity bonus
            best = None
            best_score = -float("inf")

            for candidate in tournament:
                fitness_score = candidate[1]

                # Information bonus: how much new info does this candidate bring?
                info_bonus = 0.0
                if selected and self.info_state:
                    for existing in selected:
                        info_bonus += self._information_distance(
                            candidate[0], existing[0]
                        )

                score = fitness_score + 0.1 * info_bonus
                if score > best_score:
                    best_score = score
                    best = candidate

            if best:
                selected.append(best)

        return selected

    def _information_distance(
        self, genome1: Dict[str, float], genome2: Dict[str, float]
    ) -> float:
        """Compute information distance between two genomes."""
        # Use common keys only
        common_keys = set(genome1.keys()) & set(genome2.keys())
        if not common_keys:
            return 1.0  # Max distance if no overlap
        g1_vals = np.array([genome1[k] for k in common_keys])
        g2_vals = np.array([genome2[k] for k in common_keys])
        return js_divergence(g1_vals, g2_vals)

    def info_maximizing_crossover(
        self, parent1: Dict[str, float], parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Crossover that maximizes offspring information content.

        For each gene, select the parent whose value has higher
        mutual information with fitness (if known), or use
        weighted random selection based on per-gene entropy.
        """
        child = {}

        for gene_name in parent1:
            if gene_name not in parent2:
                child[gene_name] = parent1[gene_name]
                continue

            # If we have MI data, prefer the gene from the parent
            # with higher fitness (if MI is significant)
            if self.info_state and gene_name in self.info_state.gene_fitness_mi:
                mi = self.info_state.gene_fitness_mi[gene_name]
                if mi > self.mi_threshold:
                    # Higher MI gene: take from fitter parent
                    if parent1.get("_fitness", 0) > parent2.get("_fitness", 0):
                        child[gene_name] = parent1[gene_name]
                    else:
                        child[gene_name] = parent2[gene_name]
                else:
                    # Low MI gene: maximize entropy by random selection
                    child[gene_name] = random.choice(
                        [parent1[gene_name], parent2[gene_name]]
                    )
            else:
                # No MI data: uniform crossover
                child[gene_name] = (
                    parent1[gene_name] if random.random() < 0.5 else parent2[gene_name]
                )

        return child

    def entropy_maintaining_mutation(
        self, genome: Dict[str, float], mutation_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Mutation that maintains population entropy while exploring.

        For high-MI genes: small perturbations (exploit)
        For low-MI genes: larger perturbations (explore / maintain entropy)
        """
        mutated = genome.copy()

        for gene_name in mutated:
            if gene_name.startswith("_"):
                continue

            if random.random() > mutation_rate:
                continue

            if self.info_state and gene_name in self.info_state.gene_fitness_mi:
                mi = self.info_state.gene_fitness_mi[gene_name]

                if mi > self.mi_threshold:
                    # High MI: small perturbation (exploit)
                    scale = 0.05
                else:
                    # Low MI: larger perturbation (explore / maintain diversity)
                    scale = 0.2
            else:
                # No data: medium perturbation
                scale = 0.1

            mutated[gene_name] *= 1 + random.gauss(0, scale)

        return mutated

    def breed_generation(
        self,
        population: List[Tuple[Dict[str, float], float]],
        task_fn: Callable[[Dict[str, float]], Any],
    ) -> List[Tuple[Dict[str, float], float]]:
        """
        Run one generation of information-theoretic breeding.
        """
        self.generation += 1

        # Analyze current population
        self.analyze_population(population)

        # Check entropy status
        if self.info_state and self.info_state.population_entropy < self.entropy_target:
            # Low entropy: boost diversity
            mutation_rate = 0.3
        else:
            mutation_rate = 0.1

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)

        # Elitism
        n_elite = max(1, int(len(sorted_pop) * self.elitism_ratio))
        new_population = sorted_pop[:n_elite]

        # Fill rest with information-theoretic offspring
        while len(new_population) < self.population_size:
            parents = self.select_parents(sorted_pop, k=2)
            if len(parents) < 2:
                break

            # Crossover maximizing information
            child = self.info_maximizing_crossover(parents[0][0], parents[1][0])
            child["_fitness"] = parents[0][
                1
            ]  # Track parent fitness for crossover decisions

            # Mutation maintaining entropy
            child = self.entropy_maintaining_mutation(child, mutation_rate)

            # Evaluate
            result = task_fn(child)
            fitness = (
                result.get("fitness", 0.0)
                if isinstance(result, dict)
                else float(result)
            )

            new_population.append((child, fitness))

        return new_population

    def get_info_summary(self) -> Dict:
        """Return summary of information-theoretic state."""
        if self.info_state is None:
            return {"status": "no_data", "generation": self.generation}

        return {
            "status": "analyzed",
            "generation": self.generation,
            "population_entropy": self.info_state.population_entropy,
            "fitness_entropy": self.info_state.fitness_entropy,
            "key_genes": sorted(
                self.info_state.gene_fitness_mi.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "diverse_genes": sorted(
                self.info_state.gene_entropies.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "entropy_target": self.entropy_target,
            "target_met": self.info_state.population_entropy >= self.entropy_target,
        }
