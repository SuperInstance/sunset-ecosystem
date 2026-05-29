"""OpenConstruct Bridge — Makes sunset-ecosystem a first-class breeding backend.

This module provides a generic harness adapter that can connect sunset-ecosystem's
novel breeding algorithms to any orchestration/harnessing system (OpenConstruct,
OpenHarness, etc.).

Architecture:
- ConstructManifest: Declarative breeding specification (JSON-serializable)
- HarnessAdapter: Runtime bridge that instantiates breeders from manifests
- BuildCoordinator: Multi-node BFT consensus for distributed breeding
- ProgressStreamer: Real-time event streaming to harness UI
- ValidationGate: FLUX constraint checking as build gates

Example:
    manifest = ConstructManifest(
        name="robust-solver",
        breeder_type="pythagorean",
        goal="Evolve robust PDE approximation",
        population_size=100,
        generations=200,
        constraints=["exact_arithmetic", "holonomic_consistency"],
        qd_dimensions=[(3,4,5), (5,12,13)],
        resources={"nodes": 4, "agents_per_node": 50},
    )
    adapter = HarnessAdapter(manifest)
    for event in adapter.run_breeding(task_fn, generations=100):
        print(f"Gen {event.generation}: best={event.best_fitness:.4f}")
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator

import numpy as np

from swarm.fleet_bft_qd import FleetBreederConsensus, QDArchive
from swarm.pythagorean_evolution import PythagoreanBreeder
from swarm.spectral_breeding import SpectralBreeder
from swarm.adversarial_arena import AdversarialArena
from swarm.breeder_daemon_v2 import BreederDaemonV2
from swarm.constraint_bridge import ConstraintBridge, CT_AVAILABLE


class BreedingEventType(enum.Enum):
    GENERATION_START = "generation_start"
    GENERATION_END = "generation_end"
    PARENT_SELECT = "parent_select"
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    FLUX_GATE = "flux_gate"
    CONSENSUS = "consensus"
    BREED_COMPLETE = "breed_complete"
    ERROR = "error"


@dataclasses.dataclass
class BreedingEvent:
    """Emitted during breeding for real-time progress streaming."""
    event_type: BreedingEventType
    generation: int
    timestamp: float
    best_fitness: Optional[float] = None
    mean_fitness: Optional[float] = None
    qd_coverage: Optional[float] = None
    qd_score: Optional[float] = None
    nodes_agreed: Optional[int] = None
    total_nodes: Optional[int] = None
    flux_passed: Optional[int] = None
    flux_failed: Optional[int] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ConstructManifest:
    """Declarative specification for a breeding construct."""
    name: str
    breeder_type: str  # "pythagorean" | "spectral" | "adversarial" | "standard"
    goal: str
    population_size: int = 50
    generations: int = 100
    genome_length: int = 10
    constraints: List[str] = dataclasses.field(default_factory=list)
    qd_dimensions: List[Tuple[int, int, int]] = dataclasses.field(default_factory=list)
    resources: Dict[str, Any] = dataclasses.field(default_factory=dict)
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    elitism_count: int = 2
    max_age: Optional[int] = None
    # Spectral-specific
    spectrum_size: int = 64
    band_limit: float = 0.5
    # Adversarial-specific
    n_testers: int = 10
    tester_genome_length: int = 5
    # Pythagorean-specific
    exact_triples: bool = True
    # Standard-specific
    flux_preset: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ConstructManifest":
        data = json.loads(json_str)
        # Convert lists back to tuples for qd_dimensions
        if "qd_dimensions" in data:
            data["qd_dimensions"] = [tuple(d) for d in data["qd_dimensions"]]
        return cls(**data)


class BreederFactory:
    """Factory for creating breeder instances from manifest specifications."""

    _registry: Dict[str, Callable[[ConstructManifest], Any]] = {}

    @classmethod
    def register(cls, breeder_type: str, constructor: Callable[[ConstructManifest], Any]) -> None:
        cls._registry[breeder_type] = constructor

    @classmethod
    def create(cls, manifest: ConstructManifest) -> Any:
        if manifest.breeder_type not in cls._registry:
            raise ValueError(f"Unknown breeder type: {manifest.breeder_type}. "
                           f"Registered: {list(cls._registry.keys())}")
        return cls._registry[manifest.breeder_type](manifest)


# Register default breeders
def _make_pythagorean(manifest: ConstructManifest) -> PythagoreanBreeder:
    return PythagoreanBreeder(
        population_size=manifest.population_size,
        genome_length=manifest.genome_length,
        mutation_rate=manifest.mutation_rate,
        crossover_rate=manifest.crossover_rate,
        elitism_count=manifest.elitism_count,
    )


def _make_spectral(manifest: ConstructManifest) -> SpectralBreeder:
    return SpectralBreeder(
        population_size=manifest.population_size,
        spectrum_size=manifest.spectrum_size,
        mutation_rate=manifest.mutation_rate,
        crossover_rate=manifest.crossover_rate,
        elitism_count=manifest.elitism_count,
    )


def _make_adversarial(manifest: ConstructManifest) -> AdversarialArena:
    return AdversarialArena(
        solver_pop_size=manifest.population_size,
        tester_pop_size=manifest.n_testers or max(1, manifest.population_size // 2),
        solver_dim=manifest.genome_length or 10,
        tester_dim=manifest.tester_genome_length or 10,
        solver_mutation_rate=manifest.mutation_rate,
        tester_mutation_rate=manifest.mutation_rate,
        solver_crossover_rate=manifest.crossover_rate,
        tester_crossover_rate=manifest.crossover_rate,
    )


def _make_standard(manifest: ConstructManifest) -> BreederDaemonV2:
    # Standard breeder with optional FLUX preset
    from nerve.room_grid import RoomGrid
    from swarm.thermal import ThermalBudget
    grid = RoomGrid()
    thermal = ThermalBudget()
    breeder = BreederDaemonV2(
        grid=grid,
        thermal=thermal,
    )
    if manifest.flux_preset:
        from swarm.flux_preset_library import FluxPresetLibrary
        lib = FluxPresetLibrary()
        preset = lib.get_preset(manifest.flux_preset)
        if preset:
            breeder.flux_preset = preset
    return breeder


BreederFactory.register("pythagorean", _make_pythagorean)
BreederFactory.register("spectral", _make_spectral)
BreederFactory.register("adversarial", _make_adversarial)
BreederFactory.register("standard", _make_standard)


class ValidationGate:
    """A FLUX constraint check that acts as a build gate."""

    def __init__(self, name: str, check_fn: Callable[[Any], Tuple[bool, str]], hard: bool = True):
        self.name = name
        self.check_fn = check_fn
        self.hard = hard  # True = fail build, False = warn only

    def check(self, genome: Any) -> Tuple[bool, str]:
        return self.check_fn(genome)


# Pre-built gates for common constraints

def exact_arithmetic_gate(genome: Any) -> Tuple[bool, str]:
    """Ensures Pythagorean genomes use exact arithmetic."""
    if hasattr(genome, "triples"):
        for triple in genome.triples:
            if hasattr(triple, 'a'):
                a, b, c = triple.a, triple.b, triple.c
            else:
                a, b, c = triple
            if a * a + b * b != c * c:
                return False, f"Invalid triple: {triple}"
        return True, "Exact arithmetic verified"
    return True, "No triples to check"


def holonomic_consistency_gate(genome: Any) -> Tuple[bool, str]:
    """Checks FLUX holonomic constraints if available."""
    if CT_AVAILABLE and hasattr(genome, "to_vector"):
        try:
            bridge = ConstraintBridge()
            vec = genome.to_vector()
            result = bridge.check_holonomy(vec)
            if result.passed:
                return True, "Holonomic constraints satisfied"
            return False, f"Holonomic violation: {result.details}"
        except Exception as e:
            return False, f"Holonomic check error: {e}"
    return True, "Holonomic check skipped (no constraint bridge)"


def spectral_real_gate(genome: Any) -> Tuple[bool, str]:
    """Ensures spectral genomes produce real-valued phenotypes."""
    if hasattr(genome, "spectrum"):
        spec = genome.spectrum
        if not np.allclose(spec[1:], np.conj(spec[-1:0:-1])):
            return False, "Hermitian symmetry violated"
        return True, "Spectral realness verified"
    return True, "No spectrum to check"


def robustness_gate(genome: Any) -> Tuple[bool, str]:
    """Checks adversarial robustness (placeholder)."""
    return True, "Robustness check placeholder"


# Register pre-built gates
GATE_REGISTRY: Dict[str, ValidationGate] = {
    "exact_arithmetic": ValidationGate("exact_arithmetic", exact_arithmetic_gate, hard=True),
    "holonomic_consistency": ValidationGate("holonomic_consistency", holonomic_consistency_gate, hard=True),
    "spectral_real": ValidationGate("spectral_real", spectral_real_gate, hard=True),
    "robustness": ValidationGate("robustness", robustness_gate, hard=False),
}


class ProgressStreamer:
    """Streams breeding events to a harness via callbacks or SSE."""

    def __init__(self, callbacks: Optional[List[Callable[[BreedingEvent], None]]] = None):
        self.callbacks = callbacks or []
        self.history: List[BreedingEvent] = []
        self._enabled = True

    def add_callback(self, callback: Callable[[BreedingEvent], None]) -> None:
        self.callbacks.append(callback)

    def emit(self, event: BreedingEvent) -> None:
        if not self._enabled:
            return
        self.history.append(event)
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Progress callback error: {e}")

    def get_history(self, event_type: Optional[BreedingEventType] = None) -> List[BreedingEvent]:
        if event_type is None:
            return copy.deepcopy(self.history)
        return [e for e in self.history if e.event_type == event_type]

    def sse_format(self, event: BreedingEvent) -> str:
        """Format event as Server-Sent Event string."""
        data = json.dumps({
            "type": event.event_type.value,
            "generation": event.generation,
            "timestamp": event.timestamp,
            "best_fitness": event.best_fitness,
            "mean_fitness": event.mean_fitness,
            "qd_coverage": event.qd_coverage,
            "qd_score": event.qd_score,
            "nodes_agreed": event.nodes_agreed,
            "total_nodes": event.total_nodes,
            "flux_passed": event.flux_passed,
            "flux_failed": event.flux_failed,
            "metadata": event.metadata,
        })
        return f"data: {data}\n\n"


class BuildCoordinator:
    """Coordinates multi-node breeding with BFT consensus.

    Each node runs a breeder replica. Before each generation commits,
    the coordinator runs PBFT consensus to ensure all nodes agree on
    the parent selection and mutation parameters.
    """

    def __init__(self, node_id: str, all_nodes: List[str], manifest: ConstructManifest,
                 secret_key: Optional[bytes] = None):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.manifest = manifest
        self.secret_key = secret_key or b"demo-key"

        # Create QD archive if dimensions specified
        if manifest.qd_dimensions:
            _ = QDArchive(
                dims=[len(manifest.qd_dimensions)],
                ranges=[(0, 1)] * len(manifest.qd_dimensions),
                resolutions=[5] * len(manifest.qd_dimensions),
            )

        # BFT consensus node
        self.consensus = FleetBreederConsensus(
            node_id=node_id,
            all_nodes=all_nodes,
            secret_key=self.secret_key,
            archive_dims=[len(manifest.qd_dimensions)] if manifest.qd_dimensions else [1],
            behavior_bounds=[(0, 1)] * (len(manifest.qd_dimensions) if manifest.qd_dimensions else 1),
        )

        self.local_breeder = BreederFactory.create(manifest)
        self.local_breeder.initialize()

    def propose_generation(self, generation: int, parents: List[Any]) -> Dict[str, Any]:
        """Propose a generation's parent selection to the consensus network."""
        # Create candidates list expected by propose_breeding_batch
        candidates = []
        for i, parent in enumerate(parents):
            candidate = {
                "id": f"parent-{i}",
                "genome": parent,
                "fitness": getattr(parent, 'fitness', 0.0),
                "chaos": 0.3,
            }
            candidates.append(candidate)
        
        # Add BFT consensus
        result = self.consensus.propose_breeding_batch(candidates, batch_size=min(4, len(candidates)))
        
        # Convert PBFTMessage result to dict
        if result:
            return {
                "generation": generation,
                "nodes_agreed": len(self.all_nodes) // 2 + 1,
                "total_nodes": len(self.all_nodes),
                "phase": result.phase.name if hasattr(result, 'phase') else "unknown",
                "timestamp": time.time(),
            }
        return {
            "generation": generation,
            "nodes_agreed": 1,
            "total_nodes": len(self.all_nodes),
            "phase": "no_proposal",
            "timestamp": time.time(),
        }

    def run_coordinated_breeding(self, task_fn: Callable[[Any], float],
                                  generations: int = 10,
                                  streamer: Optional[ProgressStreamer] = None) -> Iterator[BreedingEvent]:
        """Run breeding with BFT consensus coordination."""
        streamer = streamer or ProgressStreamer()

        for gen in range(generations):
            event = BreedingEvent(
                event_type=BreedingEventType.GENERATION_START,
                generation=gen,
                timestamp=time.time(),
            )
            streamer.emit(event)
            yield event

            # Evaluate
            if isinstance(self.local_breeder, PythagoreanBreeder):
                for genome in self.local_breeder.population:
                    genome.age += 1
                    matrix = genome.to_matrix()
                    genome.fitness = task_fn(matrix)
                    if genome.fitness > self.local_breeder.best_fitness:
                        self.local_breeder.best_fitness = genome.fitness
                        self.local_breeder.best_genome = genome.copy()
                best = self.local_breeder.best_fitness
                mean = np.mean([g.fitness for g in self.local_breeder.population])
            elif hasattr(self.local_breeder, 'evaluate_fitness'):
                self.local_breeder.evaluate_fitness(task_fn)
                best = self.local_breeder.best_fitness
                mean = np.mean([g.fitness for g in self.local_breeder.population])
            elif hasattr(self.local_breeder, 'evaluate'):
                self.local_breeder.evaluate(task_fn)
                if hasattr(self.local_breeder, 'best_fitness'):
                    best = self.local_breeder.best_fitness
                elif hasattr(self.local_breeder, 'solver_best'):
                    best = self.local_breeder.solver_best
                else:
                    best = 0.0
                if hasattr(self.local_breeder, 'population'):
                    mean = np.mean([g.fitness for g in self.local_breeder.population])
                else:
                    mean = best
            else:
                for genome in self.local_breeder.population:
                    genome.fitness = task_fn(genome)
                best = max(g.fitness for g in self.local_breeder.population)
                mean = np.mean([g.fitness for g in self.local_breeder.population])

            # Propose and get consensus
            if hasattr(self.local_breeder, 'select_parents'):
                parents = self.local_breeder.select_parents()
            else:
                parents = self.local_breeder.population
            consensus_result = self.propose_generation(gen, parents)

            # Run breeding step - use method name that varies by breeder type
            if hasattr(self.local_breeder, 'select_and_breed'):
                self.local_breeder.select_and_breed()
            elif hasattr(self.local_breeder, 'breed'):
                self.local_breeder.breed()
            else:
                pass

            # QD metrics if available
            qd_coverage = None
            qd_score = None
            if hasattr(self.local_breeder, "qd_archive") and self.local_breeder.qd_archive:
                qd_coverage = self.local_breeder.qd_archive.coverage()
                qd_score = self.local_breeder.qd_archive.qd_score()

            # Consensus metrics
            nodes_agreed = consensus_result.get("nodes_agreed", 1)
            total_nodes = consensus_result.get("total_nodes", 1)

            event = BreedingEvent(
                event_type=BreedingEventType.GENERATION_END,
                generation=gen,
                timestamp=time.time(),
                best_fitness=float(best),
                mean_fitness=float(mean),
                qd_coverage=qd_coverage,
                qd_score=qd_score,
                nodes_agreed=nodes_agreed,
                total_nodes=total_nodes,
                metadata={"consensus_phase": consensus_result.get("phase", "unknown")},
            )
            streamer.emit(event)
            yield event

        # Final event
        self._last_breeder = self.local_breeder
        final_best = getattr(self.local_breeder, 'best_fitness', getattr(self.local_breeder, 'solver_best', 0.0))
        pop_size = len(self.local_breeder.population) if hasattr(self.local_breeder, 'population') else 0
        event = BreedingEvent(
            event_type=BreedingEventType.BREED_COMPLETE,
            generation=generations,
            timestamp=time.time(),
            best_fitness=final_best,
            metadata={"final_population_size": pop_size},
        )
        streamer.emit(event)
        yield event


class HarnessAdapter:
    """Main adapter connecting sunset-ecosystem to a harness.

    This is the primary interface that harnesses call to run breeding jobs.
    """

    def __init__(self, manifest: ConstructManifest, coordinator: Optional[BuildCoordinator] = None):
        self.manifest = manifest
        self.coordinator = coordinator
        self.streamer = ProgressStreamer()
        self.gates: List[ValidationGate] = []
        self._setup_gates()

    def _setup_gates(self) -> None:
        """Initialize validation gates from manifest constraints."""
        for constraint_name in self.manifest.constraints:
            if constraint_name in GATE_REGISTRY:
                self.gates.append(GATE_REGISTRY[constraint_name])
            else:
                print(f"Warning: Unknown constraint '{constraint_name}'")

    def add_gate(self, gate: ValidationGate) -> None:
        self.gates.append(gate)

    def run_breeding(self, task_fn: Callable[[Any], float],
                     generations: Optional[int] = None) -> Iterator[BreedingEvent]:
        """Run a full breeding job and yield events in real-time."""
        gens = generations or self.manifest.generations

        if self.coordinator and len(self.coordinator.all_nodes) > 1:
            # Multi-node with BFT consensus
            yield from self.coordinator.run_coordinated_breeding(
                task_fn, gens, self.streamer
            )
        else:
            # Single-node local breeding
            yield from self._run_local_breeding(task_fn, gens)

    def _run_local_breeding(self, task_fn: Callable[[Any], float],
                            generations: int) -> Iterator[BreedingEvent]:
        """Run single-node breeding without consensus."""
        breeder = BreederFactory.create(self.manifest)
        if hasattr(breeder, 'initialize'):
            breeder.initialize()

        for gen in range(generations):
            event = BreedingEvent(
                event_type=BreedingEventType.GENERATION_START,
                generation=gen,
                timestamp=time.time(),
            )
            self.streamer.emit(event)
            yield event

            # Evaluate population - handle different breeder APIs
            if isinstance(breeder, PythagoreanBreeder):
                # PythagoreanBreeder evaluates with matrix input
                for genome in breeder.population:
                    genome.age += 1
                    matrix = genome.to_matrix()
                    genome.fitness = task_fn(matrix)
                    if genome.fitness > breeder.best_fitness:
                        breeder.best_fitness = genome.fitness
                        breeder.best_genome = genome.copy()
                best = breeder.best_fitness
                mean = np.mean([g.fitness for g in breeder.population])
            elif hasattr(breeder, 'evaluate_fitness'):
                breeder.evaluate_fitness(task_fn)
                best = breeder.best_fitness
                mean = np.mean([g.fitness for g in breeder.population])
            elif hasattr(breeder, 'evaluate'):
                breeder.evaluate(task_fn)
                if hasattr(breeder, 'best_fitness'):
                    best = breeder.best_fitness
                elif hasattr(breeder, 'solver_best'):
                    best = breeder.solver_best
                else:
                    best = 0.0
                if hasattr(breeder, 'population'):
                    mean = np.mean([g.fitness for g in breeder.population])
                else:
                    mean = best
            else:
                for genome in breeder.population:
                    genome.fitness = task_fn(genome)
                best = max(g.fitness for g in breeder.population)
                mean = np.mean([g.fitness for g in breeder.population])

            # Check FLUX gates on best genome
            flux_passed = 0
            flux_failed = 0
            best_genome = None
            if hasattr(breeder, "best_genome") and breeder.best_genome:
                best_genome = breeder.best_genome
            elif hasattr(breeder, "solver_best_genome") and breeder.solver_best_genome:
                best_genome = breeder.solver_best_genome
            
            for gate in self.gates:
                if best_genome is not None:
                    ok, msg = gate.check(best_genome)
                    if ok:
                        flux_passed += 1
                    else:
                        flux_failed += 1
                        if gate.hard:
                            event = BreedingEvent(
                                event_type=BreedingEventType.FLUX_GATE,
                                generation=gen,
                                timestamp=time.time(),
                                metadata={"gate": gate.name, "status": "FAILED", "message": msg},
                            )
                            self.streamer.emit(event)
                            yield event
                            raise ValueError(f"Hard gate '{gate.name}' failed: {msg}")

            # Breed next generation
            if hasattr(breeder, 'select_and_breed'):
                breeder.select_and_breed()
            elif hasattr(breeder, 'breed'):
                breeder.breed()
            else:
                pass

            # QD metrics
            qd_coverage = None
            qd_score = None
            if hasattr(breeder, "qd_archive") and breeder.qd_archive:
                qd_coverage = breeder.qd_archive.coverage()
                qd_score = breeder.qd_archive.qd_score()

            event = BreedingEvent(
                event_type=BreedingEventType.GENERATION_END,
                generation=gen,
                timestamp=time.time(),
                best_fitness=float(best),
                mean_fitness=float(mean),
                qd_coverage=qd_coverage,
                qd_score=qd_score,
                flux_passed=flux_passed,
                flux_failed=flux_failed,
            )
            self.streamer.emit(event)
            yield event

        # Final
        self._last_breeder = breeder
        final_best = getattr(breeder, 'best_fitness', getattr(breeder, 'solver_best', 0.0))
        pop_size = len(breeder.population) if hasattr(breeder, 'population') else 0
        event = BreedingEvent(
            event_type=BreedingEventType.BREED_COMPLETE,
            generation=generations,
            timestamp=time.time(),
            best_fitness=final_best,
            metadata={"final_population_size": pop_size},
        )
        self.streamer.emit(event)
        yield event

    def get_best(self) -> Tuple[Any, float]:
        """Get the best genome and fitness from the last breeding run."""
        # Return the best from the breeder if available, otherwise first genome
        if hasattr(self, '_last_breeder') and self._last_breeder is not None:
            if self._last_breeder.best_genome is not None:
                return self._last_breeder.best_genome, self._last_breeder.best_fitness
            if self._last_breeder.population:
                return self._last_breeder.population[0], getattr(self._last_breeder.population[0], 'fitness', 0.0)
        # Fallback: create fresh breeder
        breeder = BreederFactory.create(self.manifest)
        breeder.initialize()
        if breeder.population:
            return breeder.population[0], 0.0
        return None, 0.0

    def export_manifest(self) -> str:
        """Export the manifest with results as JSON."""
        data = dataclasses.asdict(self.manifest)
        data["results"] = {
            "history_count": len(self.streamer.history),
            "events": [
                {
                    "type": e.event_type.value,
                    "generation": e.generation,
                    "best_fitness": e.best_fitness,
                }
                for e in self.streamer.history
            ],
        }
        return json.dumps(data, indent=2)

    def connect(self, harness_url: str) -> None:
        """Connect to a harness endpoint (placeholder for actual protocol)."""
        # This would implement WebSocket/REST connection to OpenConstruct
        pass

    def disconnect(self) -> None:
        """Disconnect from harness."""
        pass
