"""
Causal Discovery Breeder

Uses causal inference (constraint-based PC algorithm + do-calculus) to
discover causal relationships between breeding parameters and fitness outcomes.

Instead of treating the fitness landscape as a black box, this breeder
asks: "What actually causes fitness to improve?" and uses the discovered
causal graph to guide breeding decisions.

Key innovations:
- Causal graph discovery from breeding history
- Do-calculus informed parent selection (intervention-aware)
- Counterfactual evaluation: "What would fitness be if we had mutated gene X?"
- Causal effect estimation for each breeding operator

References:
- Spirtes, Glymour, Scheines (2000) - Causation, Prediction, and Search
- Pearl (2009) - Causality: Models, Reasoning, and Inference
- Peters, Janzing, Schölkopf (2017) - Elements of Causal Inference
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CausalVariable:
    """A variable in the causal model."""

    name: str
    value: float
    type: str = "continuous"  # continuous, discrete, binary


@dataclass
class CausalEdge:
    """A directed edge in the causal graph."""

    source: str
    target: str
    weight: float = 1.0  # Effect strength
    confidence: float = 0.0  # Statistical confidence


@dataclass
class CausalGraph:
    """Discovered causal graph over breeding variables."""

    nodes: Set[str] = field(default_factory=set)
    edges: List[CausalEdge] = field(default_factory=list)
    # Conditional independence tests performed
    ci_tests: int = 0
    # Number of edges removed by CI testing
    edges_removed: int = 0

    def add_edge(
        self, source: str, target: str, weight: float = 1.0, confidence: float = 0.0
    ):
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges.append(CausalEdge(source, target, weight, confidence))

    def parents(self, node: str) -> List[str]:
        """Return all parents of a node in the causal graph."""
        return [e.source for e in self.edges if e.target == node]

    def children(self, node: str) -> List[str]:
        """Return all children of a node."""
        return [e.target for e in self.edges if e.source == node]

    def has_path(self, source: str, target: str) -> bool:
        """Check if there's a directed path from source to target."""
        visited = set()
        stack = [source]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current not in visited:
                visited.add(current)
                stack.extend(self.children(current))
        return False

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order (causal order)."""
        in_degree = {n: 0 for n in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        queue = [n for n in self.nodes if in_degree.get(n, 0) == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def to_dict(self) -> Dict:
        return {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "confidence": e.confidence,
                }
                for e in self.edges
            ],
            "ci_tests": self.ci_tests,
            "edges_removed": self.edges_removed,
        }


class CausalDiscoveryEngine:
    """
    Constraint-based causal discovery using the PC algorithm.
    Discovers causal structure from breeding history data.
    """

    def __init__(self, alpha: float = 0.05, max_cond_vars: int = 3):
        self.alpha = alpha  # Significance level for CI tests
        self.max_cond_vars = max_cond_vars

    def discover(
        self, data: List[Dict[str, float]], variables: List[str]
    ) -> CausalGraph:
        """
        Run PC algorithm on breeding history data.

        Args:
            data: List of observations, each a dict of variable->value
            variables: List of variable names to include in graph

        Returns:
            CausalGraph with discovered edges
        """
        graph = CausalGraph()
        graph.nodes = set(variables)

        # Step 1: Start with fully connected graph
        adjacency = {v: set(variables) - {v} for v in variables}

        ci_tests = 0
        edges_removed = 0

        # Step 2: Iteratively test conditional independences
        for depth in range(self.max_cond_vars + 1):
            for x in variables:
                for y in list(adjacency[x]):
                    if y not in adjacency[x]:
                        continue

                    # Try all conditioning sets of size 'depth'
                    neighbors = list(adjacency[x] - {y})
                    if len(neighbors) < depth:
                        continue

                    from itertools import combinations

                    for cond_set in combinations(neighbors, depth):
                        ci_tests += 1
                        if self._conditional_independence(data, x, y, list(cond_set)):
                            adjacency[x].discard(y)
                            adjacency[y].discard(x)
                            edges_removed += 1
                            break

        # Step 3: Orient edges (simplified: v-structure detection)
        # For each pair X-Y, check if there's a Z where X-Z and Z-Y
        # but X and Y are not adjacent → X → Z ← Y
        for z in variables:
            parents_of_z = []
            for x in variables:
                if x != z and x in adjacency.get(z, set()):
                    parents_of_z.append(x)

            for i, x in enumerate(parents_of_z):
                for y in parents_of_z[i + 1 :]:
                    if y not in adjacency.get(x, set()):
                        # X and Y are not adjacent, but both connect to Z
                        # Orient: X → Z ← Y
                        graph.add_edge(x, z, weight=1.0, confidence=0.7)
                        graph.add_edge(y, z, weight=1.0, confidence=0.7)

        # Add remaining undirected edges with lower confidence
        for x in variables:
            for y in adjacency.get(x, set()):
                if x < y:  # Avoid duplicates
                    # Check if edge already oriented
                    existing = any(
                        (e.source == x and e.target == y)
                        or (e.source == y and e.target == x)
                        for e in graph.edges
                    )
                    if not existing:
                        # Add as bidirectional with low confidence
                        graph.add_edge(x, y, weight=0.5, confidence=0.3)

        graph.ci_tests = ci_tests
        graph.edges_removed = edges_removed
        return graph

    def _conditional_independence(
        self, data: List[Dict[str, float]], x: str, y: str, cond_set: List[str]
    ) -> bool:
        """
        Test conditional independence X ⊥ Y | Z using partial correlation.
        Returns True if independent (p-value > alpha).
        """
        if len(data) < 3:
            return False  # Not enough data

        # Extract values
        x_vals = np.array([d.get(x, 0.0) for d in data])
        y_vals = np.array([d.get(y, 0.0) for d in data])

        if len(cond_set) == 0:
            # Marginal independence: Pearson correlation
            if np.std(x_vals) == 0 or np.std(y_vals) == 0:
                return True
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            if np.isnan(corr):
                return True
            # Fisher z-transform
            z = 0.5 * np.log((1 + corr) / (1 - corr))
            se = 1.0 / np.sqrt(len(data) - 3)
            p_value = 2 * (1 - self._normal_cdf(abs(z / se)))
            return p_value > self.alpha
        else:
            # Conditional independence: partial correlation
            z_matrix = np.array([[d.get(z, 0.0) for z in cond_set] for d in data])
            return self._partial_correlation_test(x_vals, y_vals, z_matrix)

    def _partial_correlation_test(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> bool:
        """Test partial correlation using linear regression residuals."""
        try:
            # Regress X and Y on Z, then test correlation of residuals
            if z.ndim == 1:
                z = z.reshape(-1, 1)

            # Simple least squares regression
            z_with_intercept = np.column_stack([np.ones(len(z)), z])

            # X residuals
            beta_x = np.linalg.lstsq(z_with_intercept, x, rcond=None)[0]
            x_pred = z_with_intercept @ beta_x
            x_res = x - x_pred

            # Y residuals
            beta_y = np.linalg.lstsq(z_with_intercept, y, rcond=None)[0]
            y_pred = z_with_intercept @ beta_y
            y_res = y - y_pred

            if np.std(x_res) == 0 or np.std(y_res) == 0:
                return True

            corr = np.corrcoef(x_res, y_res)[0, 1]
            if np.isnan(corr):
                return True

            n = len(x)
            k = z.shape[1]
            df = n - k - 2

            if df <= 0:
                return False

            # t-statistic for partial correlation
            t_stat = corr * np.sqrt(df / (1 - corr**2))
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
            return p_value > self.alpha

        except (np.linalg.LinAlgError, ValueError):
            return False

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function."""
        import math

        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # For large df, approximate with normal
        if df > 30:
            return self._normal_cdf(t)
        # Simple approximation for small df
        x = df / (df + t**2)
        # Incomplete beta approximation (simplified)
        return 1.0 - 0.5 * x  # Very rough approximation


@dataclass
class CausalEffectEstimate:
    """Estimated causal effect of an intervention."""

    intervention: str  # Variable being intervened on
    target: str  # Outcome variable
    effect: float  # Estimated effect size
    confidence_interval: Tuple[float, float]
    method: str = "backdoor"  # backdoor, frontdoor, IV


class DoCalculusEngine:
    """
    Implements do-calculus for estimating causal effects.
    Given a causal graph, estimates P(Y | do(X)) for breeding interventions.
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def estimate_effect(
        self, intervention: str, target: str, data: List[Dict[str, float]]
    ) -> CausalEffectEstimate:
        """
        Estimate causal effect of intervening on 'intervention'
        on the 'target' variable using the backdoor criterion.
        """
        # Find backdoor adjustment set
        adjustment_set = self._find_backdoor_set(intervention, target)

        if adjustment_set is None:
            return CausalEffectEstimate(
                intervention=intervention,
                target=target,
                effect=0.0,
                confidence_interval=(0.0, 0.0),
                method="unidentified",
            )

        # Estimate effect via adjustment
        effect = self._backdoor_adjustment(data, intervention, target, adjustment_set)

        # Bootstrap confidence interval
        ci = self._bootstrap_ci(data, intervention, target, adjustment_set)

        return CausalEffectEstimate(
            intervention=intervention,
            target=target,
            effect=effect,
            confidence_interval=ci,
            method="backdoor",
        )

    def _find_backdoor_set(self, intervention: str, target: str) -> Optional[List[str]]:
        """
        Find a set of variables that blocks all backdoor paths
        from intervention to target.
        Simplified: return all parents of intervention.
        """
        parents = self.graph.parents(intervention)
        if parents:
            return parents
        # If no parents, empty adjustment set works
        return []

    def _backdoor_adjustment(
        self, data: List[Dict[str, float]], x: str, y: str, z: List[str]
    ) -> float:
        """
        Estimate E[Y | do(X=x)] via backdoor adjustment:
        Σ_z E[Y | X=x, Z=z] P(Z=z)
        """
        if not data:
            return 0.0

        # For continuous variables, use stratification
        if not z:
            # Simple regression: Y ~ X
            x_vals = np.array([d.get(x, 0.0) for d in data])
            y_vals = np.array([d.get(y, 0.0) for d in data])

            if np.std(x_vals) == 0:
                return 0.0

            # Linear regression coefficient
            x_mean = np.mean(x_vals)
            y_mean = np.mean(y_vals)
            numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
            denominator = np.sum((x_vals - x_mean) ** 2)

            if denominator == 0:
                return 0.0

            return numerator / denominator
        else:
            # Stratified by adjustment variables
            # For simplicity, use mean difference within strata
            strata = self._stratify_data(data, z)
            total_effect = 0.0

            for stratum_data in strata.values():
                if len(stratum_data) < 2:
                    continue
                x_vals = [d.get(x, 0.0) for d in stratum_data]
                y_vals = [d.get(y, 0.0) for d in stratum_data]

                if len(set(x_vals)) < 2:
                    continue

                # Difference in means for high vs low X
                median_x = np.median(x_vals)
                high_y = [
                    y_vals[i] for i in range(len(x_vals)) if x_vals[i] >= median_x
                ]
                low_y = [y_vals[i] for i in range(len(x_vals)) if x_vals[i] < median_x]

                if high_y and low_y:
                    stratum_effect = np.mean(high_y) - np.mean(low_y)
                    total_effect += stratum_effect * len(stratum_data) / len(data)

            return total_effect

    def _stratify_data(
        self, data: List[Dict[str, float]], z: List[str]
    ) -> Dict[str, List[Dict]]:
        """Stratify data by values of adjustment variables."""
        strata = {}
        for d in data:
            key = tuple(round(d.get(var, 0.0), 2) for var in z)
            if key not in strata:
                strata[key] = []
            strata[key].append(d)
        return strata

    def _bootstrap_ci(
        self,
        data: List[Dict[str, float]],
        x: str,
        y: str,
        z: List[str],
        n_bootstrap: int = 100,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for causal effect."""
        effects = []
        n = len(data)

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = [random.choice(data) for _ in range(n)]
            effect = self._backdoor_adjustment(sample, x, y, z)
            effects.append(effect)

        effects.sort()
        lower = effects[int(0.025 * n_bootstrap)]
        upper = effects[int(0.975 * n_bootstrap)]
        return (lower, upper)

    def counterfactual(
        self,
        data: List[Dict[str, float]],
        observation: Dict[str, float],
        intervention: str,
        intervention_value: float,
        target: str,
    ) -> float:
        """
        Estimate counterfactual: "What would Y be if X had been x?"
        Given observed data and a causal graph.

        Three steps (Pearl's causal hierarchy):
        1. Abduction: Infer exogenous variables from observation
        2. Action: Replace X with x in structural equations
        3. Prediction: Compute Y from modified equations
        """
        # Simplified: use linear approximation
        # Y = β₀ + β₁X + β₂Z + ε
        # Counterfactual Y' = β₀ + β₁x + β₂Z + ε (where ε from observation)

        parents = self.graph.parents(intervention)
        if not parents:
            # Direct effect only
            effect = self.estimate_effect(intervention, target, data)
            observed_x = observation.get(intervention, 0.0)
            observed_y = observation.get(target, 0.0)
            # Y' = Y + β(x - X)
            return observed_y + effect.effect * (intervention_value - observed_x)

        # With parents: need to account for confounding
        # Simplified: predict Y from parents, then adjust
        y_from_parents = self._predict_from_parents(observation, target)
        effect = self.estimate_effect(intervention, target, data)
        observed_x = observation.get(intervention, 0.0)

        return y_from_parents + effect.effect * (intervention_value - observed_x)

    def _predict_from_parents(
        self, observation: Dict[str, float], target: str
    ) -> float:
        """Predict target from its parents in the causal graph."""
        parents = self.graph.parents(target)
        if not parents:
            return observation.get(target, 0.0)

        # Simple average of parent values
        parent_vals = [observation.get(p, 0.0) for p in parents]
        return np.mean(parent_vals) if parent_vals else 0.0


class CausalBreeder:
    """
    Breeding daemon that uses causal discovery to guide evolution.

    Instead of random mutation and crossover, this breeder:
    1. Maintains a causal graph of breeding variables -> fitness
    2. Intervenes on variables with strongest causal effects
    3. Evaluates counterfactuals before committing mutations
    4. Updates causal graph as new data arrives
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        history_window: int = 200,
        causal_discovery_interval: int = 10,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.history_window = history_window
        self.causal_discovery_interval = causal_discovery_interval

        self.history: List[Dict[str, float]] = []
        self.generation = 0
        self.causal_graph: Optional[CausalGraph] = None
        self.effect_estimates: Dict[str, CausalEffectEstimate] = {}

    def record_observation(self, variables: Dict[str, float]):
        """Record a breeding observation for causal discovery."""
        self.history.append(variables.copy())
        if len(self.history) > self.history_window:
            self.history.pop(0)

    def discover_causal_graph(self, variables: List[str]) -> CausalGraph:
        """Run causal discovery on accumulated history."""
        if len(self.history) < 20:
            # Not enough data — return empty graph
            return CausalGraph(nodes=set(variables))

        engine = CausalDiscoveryEngine(alpha=0.05)
        self.causal_graph = engine.discover(self.history, variables)
        return self.causal_graph

    def estimate_effects(
        self, target: str = "fitness"
    ) -> Dict[str, CausalEffectEstimate]:
        """Estimate causal effects of all variables on target."""
        if self.causal_graph is None or not self.history:
            return {}

        do_engine = DoCalculusEngine(self.causal_graph)
        effects = {}

        for var in self.causal_graph.nodes:
            if var == target:
                continue
            estimate = do_engine.estimate_effect(var, target, self.history)
            effects[var] = estimate

        self.effect_estimates = effects
        return effects

    def select_intervention(self, target: str = "fitness") -> Optional[str]:
        """
        Select the variable to intervene on based on causal effect estimates.
        Returns the variable with the strongest positive causal effect on target.
        """
        if not self.effect_estimates:
            return None

        # Filter for positive effects with good confidence
        candidates = [
            (var, est.effect)
            for var, est in self.effect_estimates.items()
            if est.effect > 0 and est.confidence_interval[0] > 0
        ]

        if not candidates:
            # Fall back to any positive effect
            candidates = [
                (var, est.effect)
                for var, est in self.effect_estimates.items()
                if est.effect > 0
            ]

        if not candidates:
            return None

        # Select variable with highest effect
        return max(candidates, key=lambda x: x[1])[0]

    def counterfactual_fitness(
        self, genome: Dict[str, float], intervention_var: str, intervention_value: float
    ) -> float:
        """
        Estimate counterfactual fitness if we change one variable.
        """
        if self.causal_graph is None or not self.history:
            return 0.0

        do_engine = DoCalculusEngine(self.causal_graph)
        return do_engine.counterfactual(
            self.history, genome, intervention_var, intervention_value, "fitness"
        )

    def causal_mutation(self, genome: Dict[str, float]) -> Dict[str, float]:
        """
        Mutate a genome using causal knowledge.
        Instead of random mutation, intervene on the variable
        with the strongest causal effect on fitness.
        """
        mutated = genome.copy()

        # Discover/update causal graph periodically
        if self.generation % self.causal_discovery_interval == 0:
            self.discover_causal_graph(list(genome.keys()) + ["fitness"])
            self.estimate_effects("fitness")

        # Select intervention target
        target_var = self.select_intervention("fitness")

        if target_var is not None and target_var in mutated:
            # Counterfactual evaluation: would this improve fitness?
            current_val = mutated[target_var]
            # Proposed change (e.g., 10% perturbation)
            proposed_val = current_val * (1 + random.uniform(-0.2, 0.2))

            cf_current = self.counterfactual_fitness(mutated, target_var, current_val)
            cf_proposed = self.counterfactual_fitness(mutated, target_var, proposed_val)

            if cf_proposed > cf_current:
                mutated[target_var] = proposed_val
        else:
            # Fall back to random mutation
            for var in mutated:
                if var != "fitness" and random.random() < self.mutation_rate:
                    mutated[var] *= 1 + random.uniform(-0.1, 0.1)

        return mutated

    def breed_generation(
        self,
        population: List[Tuple[Dict[str, float], float]],
        task_fn: Callable[[Dict[str, float]], Dict[str, Any]],
    ) -> List[Tuple[Dict[str, float], float]]:
        """
        Run one generation of causal-informed breeding.

        Args:
            population: List of (genome, fitness) tuples
            task_fn: Function to evaluate a genome

        Returns:
            New population of (genome, fitness) tuples
        """
        self.generation += 1

        # Record observations from current population
        for genome, fitness in population:
            obs = genome.copy()
            obs["fitness"] = fitness
            self.record_observation(obs)

        # Create next generation
        new_population = []

        # Elitism: keep top 10%
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
        n_elite = max(1, len(sorted_pop) // 10)
        new_population.extend(sorted_pop[:n_elite])

        # Fill rest with causal-informed offspring
        while len(new_population) < self.population_size:
            # Select parents (tournament or causal-informed)
            parent1 = self._select_parent(sorted_pop)
            parent2 = self._select_parent(sorted_pop)

            # Crossover
            child = self._crossover(parent1[0], parent2[0])

            # Causal-informed mutation
            child = self.causal_mutation(child)

            # Evaluate
            result = task_fn(child)
            fitness = (
                result.get("fitness", 0.0)
                if isinstance(result, dict)
                else float(result)
            )

            # Record new observation
            obs = child.copy()
            obs["fitness"] = fitness
            self.record_observation(obs)

            new_population.append((child, fitness))

        return new_population

    def _select_parent(
        self, sorted_pop: List[Tuple[Dict, float]]
    ) -> Tuple[Dict, float]:
        """Tournament selection."""
        tournament_size = 3
        tournament = random.sample(sorted_pop, min(tournament_size, len(sorted_pop)))
        return max(tournament, key=lambda x: x[1])

    def _crossover(
        self, parent1: Dict[str, float], parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """Uniform crossover."""
        child = {}
        for key in parent1:
            if key in parent2:
                child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
            else:
                child[key] = parent1[key]
        return child

    def get_causal_summary(self) -> Dict:
        """Return summary of causal discoveries."""
        if self.causal_graph is None:
            return {"status": "insufficient_data", "history_size": len(self.history)}

        return {
            "status": "discovered",
            "history_size": len(self.history),
            "nodes": len(self.causal_graph.nodes),
            "edges": len(self.causal_graph.edges),
            "ci_tests": self.causal_graph.ci_tests,
            "top_effects": sorted(
                [
                    {
                        "variable": var,
                        "effect": est.effect,
                        "ci": est.confidence_interval,
                    }
                    for var, est in self.effect_estimates.items()
                ],
                key=lambda x: abs(x["effect"]),
                reverse=True,
            )[:5],
            "topological_order": self.causal_graph.topological_sort(),
        }
