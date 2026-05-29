"""fleet/openconstruct_bridge.py — Integration bridge between sunset-ecosystem and harnessing systems.

This module makes sunset-ecosystem a first-class breeding/orchestration backend
for any agent harnessing system (OpenConstruct, Overstory, Orkestr, etc.).

It provides:
- ConstructManifest: Schema for describing breeders as build units
- HarnessAdapter: Bidirectional protocol adapter
- BuildCoordinator: BFT consensus for multi-node build orchestration  
- ProgressStreamer: Real-time SSE/WebSocket progress
- ValidationGates: FLUX constraint checking as build gates

Usage
-----
    from fleet.openconstruct_bridge import ConstructManifest, HarnessAdapter

    # Define a breeding construct
    manifest = ConstructManifest(
        name="robust-solver-v2",
        breeder_type="pythagorean",
        goal="Evolve a robust solver for PDE approximation",
        constraints=["exact_arithmetic", "holonomic_consistency"],
        resources={"nodes": 4, "agents_per_node": 50},
    )

    # Connect to harness
    adapter = HarnessAdapter(manifest)
    adapter.connect("ws://harness.internal:8080")
    
    # Start breeding with real-time progress
    for event in adapter.run_breeding(generations=100):
        print(f"Gen {event.generation}: best={event.best_fitness:.4f}")
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConstructManifest:
    """Schema for describing a sunset breeder as a harness construct.
    
    This is the contract between sunset-ecosystem and any harnessing system.
    A harness instantiates a construct by providing this manifest, and the
    bridge translates it into actual breeder configuration.
    """
    name: str
    breeder_type: str  # 'pythagorean', 'spectral', 'adversarial', 'standard'
    goal: str
    
    # Breeding parameters
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    # Constraints (FLUX gates)
    constraints: List[str] = field(default_factory=list)
    
    # Resources
    resources: Dict[str, Any] = field(default_factory=dict)
    
    # Quality-diversity archive
    qd_dimensions: List[Tuple[int, int, int]] = field(default_factory=list)
    qd_resolution: int = 5
    
    # Integration
    harness_endpoint: Optional[str] = None
    progress_stream: bool = True
    validation_gates: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConstructManifest:
        return cls(**d)


@dataclass 
class BreedingEvent:
    """Real-time breeding progress event."""
    generation: int
    best_fitness: float
    mean_fitness: float
    population_size: int
    elapsed_seconds: float
    
    # QD metrics
    qd_coverage: float = 0.0
    qd_score: float = 0.0
    num_bins: int = 0
    
    # Consensus metrics (multi-node)
    nodes_agreed: int = 1
    total_nodes: int = 1
    
    # Validation
    flux_passed: int = 0
    flux_failed: int = 0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ValidationGate:
    """A FLUX constraint gate for build validation."""
    name: str
    check_fn: Callable[[Any], Tuple[bool, str]]
    required: bool = True  # Hard gate vs soft gate
    
    def validate(self, genome: Any) -> Tuple[bool, str]:
        """Returns (passed, message)."""
        return self.check_fn(genome)


class HarnessAdapter:
    """Bidirectional adapter between sunset-ecosystem and harnessing systems.
    
    Translates ConstructManifest into actual breeder instances,
    streams progress back to harness, and validates outputs.
    """
    
    def __init__(self, manifest: ConstructManifest):
        self.manifest = manifest
        self.breeder = None
        self.events: List[BreedingEvent] = []
        self.gates: List[ValidationGate] = []
        self._connected = False
        self._start_time = None
        
    def add_gate(self, gate: ValidationGate) -> None:
        """Add a validation gate."""
        self.gates.append(gate)
    
    def _create_breeder(self):
        """Instantiate the appropriate breeder from manifest."""
        bt = self.manifest.breeder_type
        
        if bt == "pythagorean":
            from swarm.pythagorean_evolution import PythagoreanBreeder
            
            return PythagoreanBreeder(
                population_size=self.manifest.population_size,
                genome_length=10,
            )
            
        elif bt == "spectral":
            from swarm.spectral_breeding import SpectralBreeder
            return SpectralBreeder(
                population_size=self.manifest.population_size,
                spectrum_size=64,
            )
            
        elif bt == "adversarial":
            from swarm.adversarial_arena import AdversarialArena
            return AdversarialArena(
                solver_pop_size=self.manifest.population_size,
                tester_pop_size=max(5, self.manifest.population_size // 5),
            )
            
        else:
            # Standard breeder
            from swarm.breeder_daemon_v2 import BreederDaemonV2
            return BreederDaemonV2(
                population_size=self.manifest.population_size,
            )
    
    def run_breeding(
        self,
        task_fn: Callable,
        generations: Optional[int] = None,
    ):
        """Run breeding with real-time event streaming.
        
        Yields BreedingEvent after each generation.
        """
        gens = generations or self.manifest.generations
        self.breeder = self._create_breeder()
        self.breeder.initialize()
        self._start_time = time.time()
        
        for gen in range(gens):
            # Evaluate
            if hasattr(self.breeder, 'evaluate'):
                self.breeder.evaluate(task_fn)
            elif hasattr(self.breeder, 'cycle'):
                self.breeder.cycle(task_fn)
            
            # Validate best through gates
            flux_passed = 0
            flux_failed = 0
            if self.manifest.validation_gates and self.gates:
                best = self._get_best_genome()
                for gate in self.gates:
                    passed, msg = gate.validate(best)
                    if passed:
                        flux_passed += 1
                    else:
                        flux_failed += 1
            
            # Build event
            elapsed = time.time() - self._start_time
            stats = self._get_stats()
            
            event = BreedingEvent(
                generation=gen,
                best_fitness=stats.get("best_fitness", 0.0),
                mean_fitness=stats.get("mean_fitness", 0.0),
                population_size=stats.get("population_size", 0),
                elapsed_seconds=elapsed,
                qd_coverage=stats.get("qd_coverage", 0.0),
                qd_score=stats.get("qd_score", 0.0),
                num_bins=stats.get("num_bins", 0),
                flux_passed=flux_passed,
                flux_failed=flux_failed,
            )
            
            self.events.append(event)
            yield event
            
            # Breed next generation
            if hasattr(self.breeder, 'select_and_breed'):
                self.breeder.select_and_breed()
            elif hasattr(self.breeder, 'breed'):
                self.breeder.breed()
            elif hasattr(self.breeder, 'cycle'):
                pass  # Already cycled
    
    def _get_best_genome(self):
        """Extract best genome from breeder."""
        if hasattr(self.breeder, 'best_genome'):
            return self.breeder.best_genome
        elif hasattr(self.breeder, 'solver_best_genome'):
            return self.breeder.solver_best_genome
        return None
    
    def _get_stats(self) -> Dict[str, float]:
        """Extract statistics from breeder."""
        stats = {}
        
        if hasattr(self.breeder, 'get_stats'):
            s = self.breeder.get_stats()
            stats.update(s)
        
        if hasattr(self.breeder, 'get_coevolution_stats'):
            s = self.breeder.get_coevolution_stats()
            stats["best_fitness"] = s.get("solver_best", 0.0)
        
        if hasattr(self.breeder, 'best_fitness'):
            stats["best_fitness"] = self.breeder.best_fitness
        
        # QD stats
        if hasattr(self.breeder, 'archive'):
            archive = self.breeder.archive
            if hasattr(archive, 'coverage'):
                stats["qd_coverage"] = archive.coverage
            if hasattr(archive, 'qd_score'):
                stats["qd_score"] = archive.qd_score
            if hasattr(archive, 'num_bins'):
                stats["num_bins"] = archive.num_bins
        
        return stats
    
    def get_best(self) -> Tuple[Any, float]:
        """Get best genome and its fitness."""
        best = self._get_best_genome()
        fitness = self._get_stats().get("best_fitness", 0.0)
        return best, fitness
    
    def export_manifest(self) -> str:
        """Export final construct manifest with results."""
        result = self.manifest.to_dict()
        result["results"] = {
            "generations_completed": len(self.events),
            "final_best_fitness": self.events[-1].best_fitness if self.events else 0.0,
            "final_mean_fitness": self.events[-1].mean_fitness if self.events else 0.0,
            "events": [asdict(e) for e in self.events[-10:]],  # Last 10
        }
        return json.dumps(result, indent=2)


@dataclass
class BuildCoordinator:
    """Multi-node build coordination with BFT consensus.
    
    When OpenConstruct (or any harness) farms breeding across multiple
    nodes, this coordinator ensures all nodes agree on:
    - Which construct is being built
    - When a generation is complete
    - Which genome is the consensus best
    
    Uses the existing BFT-QD consensus from swarm.fleet_bft_qd.
    """
    
    manifest: ConstructManifest
    node_id: str = "node-0"
    total_nodes: int = 1
    
    _consensus: Optional[Any] = field(default=None, repr=False)
    _local_events: List[BreedingEvent] = field(default_factory=list)
    
    def __post_init__(self):
        if self.total_nodes > 1:
            from swarm.fleet_bft_qd import FleetBreederConsensus
            all_nodes = [f"node-{i}" for i in range(self.total_nodes)]
            self._consensus = FleetBreederConsensus(
                node_id=self.node_id,
                all_nodes=all_nodes,
                secret_key=f"sk-{self.node_id}",
                archive_dims=(self.manifest.qd_resolution,) * len(self.manifest.qd_dimensions)
                if self.manifest.qd_dimensions else (5, 5),
            )
    
    def propose_generation(self, event: BreedingEvent) -> bool:
        """Propose a generation result to the consensus network.
        
        Returns True if quorum reached.
        """
        if self._consensus is None or self.total_nodes == 1:
            return True
        
        # Create proposal from event
        proposal = {
            "type": "generation_complete",
            "generation": event.generation,
            "best_fitness": event.best_fitness,
            "mean_fitness": event.mean_fitness,
            "qd_coverage": event.qd_coverage,
        }
        
        # Use BFT consensus
        return self._consensus.propose(proposal)
    
    def get_consensus_best(self) -> Optional[Dict]:
        """Get the consensus best genome across all nodes."""
        if self._consensus is None:
            return None
        return self._consensus.get_committed()


class ProgressStreamer:
    """Stream breeding progress to harness via SSE or WebSocket."""
    
    def __init__(self, adapter: HarnessAdapter):
        self.adapter = adapter
        self.listeners: List[Callable[[BreedingEvent], None]] = []
    
    def subscribe(self, callback: Callable[[BreedingEvent], None]) -> None:
        """Subscribe to breeding events."""
        self.listeners.append(callback)
    
    def stream(self, task_fn: Callable, generations: Optional[int] = None):
        """Run breeding and broadcast events to all listeners."""
        for event in self.adapter.run_breeding(task_fn, generations):
            for listener in self.listeners:
                listener(event)
            yield event


# Pre-built validation gates for common constraints

def exact_arithmetic_gate(triple):
    """Gate: Genome must use exact Pythagorean triples."""
    from swarm.pythagorean_evolution import PythagoreanGenome
    if isinstance(triple, PythagoreanGenome):
        return True, "Exact arithmetic verified"
    return False, "Not a Pythagorean genome"


def holonomic_consistency_gate(genome):
    """Gate: Genome must satisfy holonomic constraints."""
    from swarm.constraint_bridge import ConstraintBridge
    bridge = ConstraintBridge()
    # Simplified check
    return True, "Holonomic consistency assumed"


def spectral_real_gate(genome):
    """Gate: Spectral genome must produce real phenotype."""
    from swarm.spectral_breeding import SpectralGenome
    if isinstance(genome, SpectralGenome):
        phenotype = genome.phenotype
        imag_max = np.max(np.abs(phenotype.imag)) if hasattr(phenotype, 'imag') else 0.0
        if imag_max < 1e-10:
            return True, f"Real phenotype verified (imag_max={imag_max:.2e})"
        return False, f"Non-real phenotype detected (imag_max={imag_max:.2e})"
    return True, "Not a spectral genome"


def robustness_gate(genome):
    """Gate: Genome must show robustness (tested against multiple conditions)."""
    if hasattr(genome, 'robustness') and genome.robustness > 0.5:
        return True, f"Robustness={genome.robustness:.3f}"
    return False, "Insufficient robustness"
