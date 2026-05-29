"""Meta-Learning Breeder — Learns which mutation strategies work per problem class."""

import hashlib
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
from collections import defaultdict


@dataclass
class ProblemFingerprint:
    """A hashable fingerprint of a problem's characteristics."""
    dim: int
    constraint_types: Tuple[str, ...]
    landscape: str  # e.g., "smooth", "rugged", "multimodal"
    
    def to_key(self) -> str:
        return f"{self.dim}:{','.join(sorted(self.constraint_types))}:{self.landscape}"


@dataclass
class StrategyRecord:
    """Tracks success rate for one strategy on one problem class."""
    name: str
    attempts: int = 0
    successes: int = 0
    ema_rate: float = 0.5
    alpha: float = 0.3  # EMA smoothing factor
    
    def update(self, improved: bool) -> None:
        self.attempts += 1
        if improved:
            self.successes += 1
        self.ema_rate = self.alpha * (1.0 if improved else 0.0) + (1 - self.alpha) * self.ema_rate
    
    def score(self, temperature: float = 1.0) -> float:
        """Softmax-ready score with exploration bonus."""
        # Small exploration bonus for under-tried strategies (diminishes quickly)
        bonus = 0.1 / (1 + self.attempts)
        return (self.ema_rate + bonus) / temperature


class MetaLearningBreeder:
    """Breeder that learns which mutation strategies work best per problem class."""
    
    def __init__(
        self,
        strategies: Optional[List[Callable]] = None,
        temperature: float = 1.0,
        decay: float = 0.99,
    ):
        self.strategies = strategies or []
        self.temperature = temperature
        self.decay = decay
        self._registry: Dict[str, Dict[str, StrategyRecord]] = defaultdict(dict)
        self._history: List[Dict] = []
        self._stats = {"generations": 0, "evaluations": 0, "adaptations": 0}
    
    def add_strategy(self, name: str, func: Callable) -> None:
        self.strategies.append((name, func))
    
    def remove_strategy(self, name: str) -> None:
        self.strategies = [(n, f) for n, f in self.strategies if n != name]
    
    def fingerprint(self, genome: List[float], constraints: List[str], landscape: str = "unknown") -> ProblemFingerprint:
        """Extract problem fingerprint from genome and constraints."""
        return ProblemFingerprint(
            dim=len(genome),
            constraint_types=tuple(sorted(constraints)),
            landscape=landscape,
        )
    
    def select_strategy(self, fingerprint: ProblemFingerprint) -> Tuple[str, Callable]:
        """Select strategy using softmax over success rates."""
        key = fingerprint.to_key()
        records = self._registry[key]
        
        # Ensure all strategies have records
        for name, func in self.strategies:
            if name not in records:
                records[name] = StrategyRecord(name)
        
        if not records:
            # No strategies registered, pick random
            name, func = random.choice(self.strategies) if self.strategies else ("none", lambda x: x)
            return name, func
        
        # Softmax selection
        scores = [(name, rec.score(self.temperature)) for name, rec in records.items()]
        total = sum(math.exp(s) for _, s in scores)
        if total == 0 or math.isnan(total):
            name, func = random.choice(self.strategies)
            return name, func
        
        r = random.random() * total
        cumulative = 0.0
        for name, score in scores:
            cumulative += math.exp(score)
            if r <= cumulative:
                for n, f in self.strategies:
                    if n == name:
                        return name, f
        
        # Fallback
        name, func = random.choice(self.strategies)
        return name, func
    
    def mutate(self, genome: List[float], fingerprint: ProblemFingerprint) -> Tuple[List[float], str]:
        """Mutate using the selected strategy."""
        strategy_name, strategy_fn = self.select_strategy(fingerprint)
        return strategy_fn(genome), strategy_name
    
    def learn(
        self,
        fingerprint: ProblemFingerprint,
        strategy_name: str,
        parent_fitness: float,
        child_fitness: float,
    ) -> None:
        """Update strategy success rate based on fitness improvement."""
        key = fingerprint.to_key()
        records = self._registry[key]
        
        if strategy_name not in records:
            records[strategy_name] = StrategyRecord(strategy_name)
        
        improved = child_fitness > parent_fitness
        records[strategy_name].update(improved)
        self._stats["adaptations"] += 1
    
    def evolve(
        self,
        population: List[List[float]],
        fitness_fn: Callable[[List[float]], float],
        constraints: List[str],
        landscape: str = "unknown",
        generations: int = 10,
    ) -> List[Tuple[List[float], float]]:
        """Run meta-learning evolution for N generations."""
        fp = self.fingerprint(population[0], constraints, landscape)
        
        for gen in range(generations):
            self._stats["generations"] += 1
            new_pop = []
            
            for genome in population:
                child, strategy_name = self.mutate(genome, fp)
                parent_fitness = fitness_fn(genome)
                child_fitness = fitness_fn(child)
                self._stats["evaluations"] += 2
                
                self.learn(fp, strategy_name, parent_fitness, child_fitness)
                
                # Keep the better one
                if child_fitness > parent_fitness:
                    new_pop.append((child, child_fitness))
                else:
                    new_pop.append((genome, parent_fitness))
            
            # Sort by fitness, keep top half
            new_pop.sort(key=lambda x: x[1], reverse=True)
            population = [g for g, _ in new_pop[:len(new_pop) // 2]]
            # Refill with random mutations of elites
            while len(population) < len(new_pop):
                elite = random.choice(population)
                child, _ = self.mutate(elite, fp)
                population.append(child)
        
        # Return final population with fitness
        return [(g, fitness_fn(g)) for g in population]
    
    def get_strategy_stats(self, fingerprint: Optional[ProblemFingerprint] = None) -> Dict:
        """Return success rates per strategy."""
        if fingerprint is None:
            # Aggregate across all fingerprints
            all_records = {}
            for key, records in self._registry.items():
                for name, rec in records.items():
                    if name not in all_records:
                        all_records[name] = StrategyRecord(name)
                    all_records[name].attempts += rec.attempts
                    all_records[name].successes += rec.successes
            return {name: {"attempts": r.attempts, "successes": r.successes, "rate": r.ema_rate} for name, r in all_records.items()}
        
        key = fingerprint.to_key()
        records = self._registry.get(key, {})
        return {name: {"attempts": r.attempts, "successes": r.successes, "rate": r.ema_rate} for name, r in records.items()}
    
    def to_dict(self) -> Dict:
        return {
            "strategies": [name for name, _ in self.strategies],
            "temperature": self.temperature,
            "registry_size": len(self._registry),
            "stats": dict(self._stats),
        }
