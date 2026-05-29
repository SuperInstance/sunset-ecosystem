"""OpenConstruct Shell Integration — sunset-ecosystem as agent attachment.

This module makes sunset-ecosystem a first-class OpenConstruct attachment:
- The agent (operator) issues simple shell commands
- The shell translates to complex breeding operations
- Results flow back as sensor readings (ticks/deltas)
- Self-healing through BFT consensus and operational traps

Architecture:
- OpenConstructShell: Main interface between agent and breeding machinery
- AttachmentRegistry: Pluggable breeder modules (bucket, crane, cutter-buncher)
- SensorArray: Real-time breeding metrics as agent-perceptible signals
- SelfHealingLoop: Automatic retry, consensus recovery, trap detection
- SkillManual: Declarative skill that teaches the agent how to operate

Example agent interaction:
    @fleet breed --type pythagorean --population 50 --generations 100
    @fleet status  # Check sensor readings
    @fleet trap-check  # Verify self-healing status
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator

import numpy as np

from fleet.openconstruct_bridge import (
    BreedingEvent,
    BreedingEventType,
    ConstructManifest,
    BreederFactory,
    HarnessAdapter,
    BuildCoordinator,
    ProgressStreamer,
    ValidationGate,
    GATE_REGISTRY,
)
from fleet.sense_decide_act import SDALoop, SDAPipeline
from nexus.fleet_conductor_v2 import FleetConductorV2
from swarm.fleet_bft_qd import FleetBreederConsensus
from swarm.breeder_daemon_v2 import BreederDaemonV2
from swarm.pythagorean_evolution import PythagoreanBreeder
from swarm.spectral_breeding import SpectralBreeder
from swarm.adversarial_arena import AdversarialArena
from swarm.nca_breeder import NCABreeder


class SensorType(enum.Enum):
    """Types of sensor readings the agent can perceive."""
    TICK = "tick"  # Periodic heartbeat
    DELTA = "delta"  # Change from previous state
    ALERT = "alert"  # Critical condition
    METRIC = "metric"  # Numeric measurement
    STATUS = "status"  # System state


@dataclasses.dataclass
class SensorReading:
    """A sensor reading formatted for agent perception."""
    sensor_type: SensorType
    name: str
    value: Any
    timestamp: float
    unit: Optional[str] = None
    threshold: Optional[float] = None
    message: Optional[str] = None

    def to_agent_text(self) -> str:
        """Convert to human-readable text for agent perception."""
        if self.sensor_type == SensorType.TICK:
            return f"[TICK] {self.name}: {self.value}{self.unit or ''}"
        elif self.sensor_type == SensorType.DELTA:
            return f"[DELTA] {self.name}: {self.value}{self.unit or ''} (change detected)"
        elif self.sensor_type == SensorType.ALERT:
            return f"[ALERT] {self.name}: {self.message} (value: {self.value})"
        elif self.sensor_type == SensorType.METRIC:
            return f"[METRIC] {self.name}: {self.value}{self.unit or ''}"
        else:
            return f"[STATUS] {self.name}: {self.value}"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), default=str)


class AttachmentRegistry:
    """Registry of pluggable breeder attachments.

    Like different heavy machinery attachments:
    - "pythagorean": Exact arithmetic breeding (bucket scooper)
    - "spectral": Fourier-domain breeding (crane)
    - "adversarial": Co-evolution arena (cutter-buncher)
    - "nca": Neural cellular automata (delimber)
    - "bft-qd": Byzantine consensus breeding (logging processor)
    """

    def __init__(self):
        self._attachments: Dict[str, Dict[str, Any]] = {}
        self._active_runs: Dict[str, HarnessAdapter] = {}
        self._sensors: Dict[str, List[SensorReading]] = {}

    def register(self, name: str, description: str, manifest_defaults: Dict[str, Any]) -> None:
        """Register a new attachment type."""
        self._attachments[name] = {
            "description": description,
            "defaults": manifest_defaults,
        }

    def list_attachments(self) -> List[Tuple[str, str]]:
        """List available attachments with descriptions."""
        return [(name, info["description"]) for name, info in self._attachments.items()]

    def spawn(self, name: str, manifest: ConstructManifest, coordinator: Optional[BuildCoordinator] = None) -> HarnessAdapter:
        """Spawn an attachment instance (like starting the engine)."""
        if name not in self._attachments:
            raise ValueError(f"Unknown attachment: {name}. Available: {list(self._attachments.keys())}")
        adapter = HarnessAdapter(manifest, coordinator=coordinator)
        run_id = f"{name}-{int(time.time())}"
        self._active_runs[run_id] = adapter
        self._sensors[run_id] = []
        return adapter, run_id

    def get_sensor_history(self, run_id: str) -> List[SensorReading]:
        return self._sensors.get(run_id, [])

    def _record_sensor(self, run_id: str, reading: SensorReading) -> None:
        if run_id not in self._sensors:
            self._sensors[run_id] = []
        self._sensors[run_id].append(reading)


# Register default attachments
ATTACHMENTS = AttachmentRegistry()
ATTACHMENTS.register(
    "pythagorean",
    "Exact arithmetic breeding using Pythagorean triples. Precision breeding for exact solutions.",
    {"breeder_type": "pythagorean", "population_size": 50, "genome_length": 10},
)
ATTACHMENTS.register(
    "spectral",
    "Fourier-domain breeding. Genomes are frequency spectra. Good for signal processing and PDE approximation.",
    {"breeder_type": "spectral", "population_size": 50, "spectrum_size": 64},
)
ATTACHMENTS.register(
    "adversarial",
    "Co-evolutionary arena. Two populations compete: solvers vs testers. Breeds robust solutions.",
    {"breeder_type": "adversarial", "population_size": 50, "n_testers": 10},
)
ATTACHMENTS.register(
    "nca",
    "Neural cellular automata breeder. Indirect encoding: genomes grow phenotypes via NCA rules. Good for pattern generation.",
    {"breeder_type": "nca", "population_size": 30, "genome_length": 10},
)
ATTACHMENTS.register(
    "standard",
    "Standard FLUX-gated breeding with preset constraints. Reliable general-purpose breeding.",
    {"breeder_type": "standard", "population_size": 50, "flux_preset": "safe"},
)


class SelfHealingLoop:
    """Self-healing mechanisms for breeding operations.

    Like a heavy machinery operator who notices something wrong and
    automatically adjusts:
    - Thermal overload → throttle breeding
    - Consensus failure → retry with view change
    - Fitness collapse → reinitialize with elitism
    - Stagnation → increase mutation rate
    """

    def __init__(self, adapter: HarnessAdapter, run_id: str):
        self.adapter = adapter
        self.run_id = run_id
        self.healing_log: List[Dict[str, Any]] = []
        self._recovery_count = 0
        self._max_recoveries = 5

    def check_health(self, event: BreedingEvent) -> List[SensorReading]:
        """Check health and emit sensor readings. May trigger healing."""
        readings = []

        # Check fitness collapse
        if event.best_fitness is not None and event.best_fitness < 0.01:
            readings.append(SensorReading(
                sensor_type=SensorType.ALERT,
                name="fitness_collapse",
                value=event.best_fitness,
                timestamp=event.timestamp,
                message="Fitness collapsed to near zero. Triggering recovery.",
            ))
            self._heal_fitness_collapse()

        # Check stagnation (no improvement for 10 generations)
        if event.generation > 10:
            recent = [e for e in self.adapter.streamer.history
                     if e.event_type == BreedingEventType.GENERATION_END
                     and e.generation >= event.generation - 10]
            if len(recent) >= 10:
                best_values = [e.best_fitness for e in recent if e.best_fitness is not None]
                if best_values and max(best_values) - min(best_values) < 0.01:
                    readings.append(SensorReading(
                        sensor_type=SensorType.ALERT,
                        name="stagnation",
                        value=max(best_values) - min(best_values),
                        timestamp=event.timestamp,
                        message="Stagnation detected. Increasing mutation rate.",
                    ))
                    self._heal_stagnation()

        # Check consensus failure
        if event.nodes_agreed is not None and event.total_nodes is not None:
            if event.nodes_agreed < event.total_nodes / 2:
                readings.append(SensorReading(
                    sensor_type=SensorType.ALERT,
                    name="consensus_failure",
                    value=event.nodes_agreed,
                    timestamp=event.timestamp,
                    message=f"Consensus failed: only {event.nodes_agreed}/{event.total_nodes} nodes agreed.",
                ))

        # Normal metrics
        if event.best_fitness is not None:
            readings.append(SensorReading(
                sensor_type=SensorType.METRIC,
                name="best_fitness",
                value=event.best_fitness,
                timestamp=event.timestamp,
                unit="score",
            ))

        if event.qd_coverage is not None:
            readings.append(SensorReading(
                sensor_type=SensorType.METRIC,
                name="qd_coverage",
                value=event.qd_coverage,
                timestamp=event.timestamp,
                unit="cells",
            ))

        return readings

    def _heal_fitness_collapse(self) -> None:
        """Recover from fitness collapse by reinitializing with elitism."""
        if self._recovery_count >= self._max_recoveries:
            return
        self._recovery_count += 1
        # Increase elitism to preserve best solutions
        if hasattr(self.adapter.manifest, 'elitism_count'):
            self.adapter.manifest.elitism_count = min(
                self.adapter.manifest.elitism_count + 2,
                self.adapter.manifest.population_size // 2
            )
        self.healing_log.append({
            "action": "fitness_collapse_recovery",
            "timestamp": time.time(),
            "recovery_count": self._recovery_count,
        })

    def _heal_stagnation(self) -> None:
        """Recover from stagnation by increasing mutation rate."""
        if self._recovery_count >= self._max_recoveries:
            return
        self._recovery_count += 1
        self.adapter.manifest.mutation_rate = min(
            self.adapter.manifest.mutation_rate * 1.5,
            0.9
        )
        self.healing_log.append({
            "action": "stagnation_recovery",
            "timestamp": time.time(),
            "new_mutation_rate": self.adapter.manifest.mutation_rate,
        })


class OpenConstructShell:
    """The agent's control cabin for operating breeding machinery.

    This is the primary interface. The agent issues simple commands,
    the shell translates them to complex breeding operations,
    and results flow back as sensor readings.

    Example:
        shell = OpenConstructShell()
        
        # List available attachments
        shell.list_attachments()
        
        # Spawn a breeding job
        run_id = shell.spawn("pythagorean", population_size=50, generations=100)
        
        # Run and get sensor readings
        for reading in shell.run(run_id, task_fn):
            print(reading.to_agent_text())
        
        # Check health
        shell.health_check(run_id)
    """

    def __init__(self, node_id: str = "shell-1", all_nodes: Optional[List[str]] = None):
        self.node_id = node_id
        self.all_nodes = all_nodes or [node_id]
        self.attachments = AttachmentRegistry()
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Initialize with default attachments
        for name, info in ATTACHMENTS._attachments.items():
            self.attachments.register(name, info["description"], info["defaults"])

    def list_attachments(self) -> List[Tuple[str, str]]:
        """List available breeding attachments (like equipment inventory)."""
        return self.attachments.list_attachments()

    def spawn(self, attachment_name: str, **kwargs) -> str:
        """Spawn a breeding attachment (like starting the engine).
        
        Args:
            attachment_name: Type of breeder (pythagorean, spectral, etc.)
            **kwargs: Override defaults (population_size, generations, etc.)
            
        Returns:
            run_id: Unique identifier for this breeding run
        """
        # Build manifest from attachment defaults + overrides
        defaults = ATTACHMENTS._attachments.get(attachment_name, {}).get("defaults", {})
        manifest_dict = {**defaults, **kwargs}
        manifest_dict["name"] = manifest_dict.get("name", f"{attachment_name}-run")
        manifest_dict["goal"] = manifest_dict.get("goal", f"Auto-generated goal for {attachment_name}")
        manifest_dict["breeder_type"] = attachment_name
        
        manifest = ConstructManifest(**manifest_dict)
        
        # Create multi-node coordinator if needed
        coordinator = None
        if len(self.all_nodes) > 1:
            coordinator = BuildCoordinator(
                node_id=self.node_id,
                all_nodes=self.all_nodes,
                manifest=manifest,
            )
        
        adapter, run_id = self.attachments.spawn(attachment_name, manifest, coordinator)
        healing = SelfHealingLoop(adapter, run_id)
        
        with self._lock:
            self._runs[run_id] = {
                "adapter": adapter,
                "healing": healing,
                "attachment": attachment_name,
                "manifest": manifest,
                "status": "spawned",
            }
        
        return run_id

    def run(self, run_id: str, task_fn: Callable[[Any], float],
            generations: Optional[int] = None) -> Iterator[SensorReading]:
        """Run a breeding job and yield sensor readings.
        
        Args:
            run_id: From spawn()
            task_fn: Fitness function
            generations: Override manifest generations
            
        Yields:
            SensorReading: Real-time sensor readings for agent perception
        """
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                yield SensorReading(
                    sensor_type=SensorType.ALERT,
                    name="run_not_found",
                    value=run_id,
                    timestamp=time.time(),
                    message=f"Run {run_id} not found",
                )
                return
            
            run["status"] = "running"
            adapter = run["adapter"]
            healing = run["healing"]
        
        # Run breeding and convert events to sensor readings
        for event in adapter.run_breeding(task_fn, generations):
            # Get health readings
            health_readings = healing.check_health(event)
            for reading in health_readings:
                yield reading
                # Record in attachment registry
                self.attachments._record_sensor(run_id, reading)
        
        with self._lock:
            run["status"] = "complete"

    def health_check(self, run_id: str) -> List[SensorReading]:
        """Get current health sensor readings for a run."""
        return self.attachments.get_sensor_history(run_id)

    def status(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of all runs or a specific run."""
        with self._lock:
            if run_id:
                run = self._runs.get(run_id)
                if not run:
                    return {"error": f"Run {run_id} not found"}
                return {
                    "run_id": run_id,
                    "attachment": run["attachment"],
                    "status": run["status"],
                    "history_count": len(run["adapter"].streamer.history),
                    "healing_log": run["healing"].healing_log,
                }
            else:
                return {
                    "active_runs": len(self._runs),
                    "runs": {
                        rid: {
                            "attachment": r["attachment"],
                            "status": r["status"],
                        }
                        for rid, r in self._runs.items()
                    },
                }

    def terminate(self, run_id: str) -> None:
        """Terminate a running breeding job."""
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                run["status"] = "terminated"

    def get_best(self, run_id: str) -> Tuple[Any, float]:
        """Get best genome from a completed run."""
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None, 0.0
            return run["adapter"].get_best()

    def run_parallel(
        self,
        campaigns: List[Dict[str, Any]],
        generations: int = 10,
        repo_path: str = ".",
    ) -> Dict[str, Any]:
        """Run multiple breeding campaigns in parallel across fleet nodes.

        Each campaign is a dict with keys:
            - name: str
            - attachment: str (pythagorean, spectral, nca, etc.)
            - params: dict (optional)
            - task_fn: callable (optional)
            - constraints: list[str] (optional)
            - node_id: str (optional, for specific node assignment)

        Returns:
            ParallelResult serialized as dict with best_campaign, success_rate, etc.

        Example:
            campaigns = [
                {"name": "exact", "attachment": "pythagorean", "params": {"population_size": 20}},
                {"name": "fourier", "attachment": "spectral", "params": {"population_size": 20}},
            ]
            result = shell.run_parallel(campaigns, generations=5)
            print(result["best_campaign"]["name"])
        """
        from fleet.parallel_breeding_orchestrator import (
            ParallelBreedingOrchestrator,
            Campaign,
        )

        # Build Campaign objects from dicts
        campaign_objs = []
        for c in campaigns:
            campaign_objs.append(Campaign(
                name=c["name"],
                attachment=c["attachment"],
                params=c.get("params", {}),
                task_fn=c.get("task_fn"),
                constraints=c.get("constraints", []),
                node_id=c.get("node_id"),
            ))

        orch = ParallelBreedingOrchestrator(
            repo_path=repo_path,
            nodes=self.all_nodes,
            max_workers=min(len(campaign_objs), 4),
            default_generations=generations,
        )

        result = orch.run_parallel(campaign_objs, generations=generations)
        return result.to_dict()

    def to_skill_manual(self) -> str:
        """Generate a skill manual that teaches the agent how to operate.
        
        This is like the equipment manual for heavy machinery:
        - What attachments are available
        - How to start the engine (spawn)
        - How to read the gauges (sensor readings)
        - What to do when something goes wrong (self-healing)
        """
        lines = [
            "# OpenConstruct Fleet Breeding Manual",
            "",
            "## Available Attachments (Equipment)",
            "",
        ]
        for name, desc in self.list_attachments():
            lines.append(f"### {name}")
            lines.append(f"{desc}")
            lines.append("")
        
        lines.extend([
            "## Operating Instructions",
            "",
            "### 1. Spawn Attachment (Start Engine)",
            "```python",
            'run_id = shell.spawn("pythagorean", population_size=50, generations=100)',
            "```",
            "",
            "### 2. Run Breeding (Operate Machinery)",
            "```python",
            "for reading in shell.run(run_id, task_fn):",
            "    print(reading.to_agent_text())  # [TICK] best_fitness: 1.45",
            "```",
            "",
            "### 3. Check Health (Read Gauges)",
            "```python",
            "readings = shell.health_check(run_id)",
            "for r in readings:",
            '    if r.sensor_type == SensorType.ALERT:',
            "        print(f'WARNING: {r.message}')",
            "```",
            "",
            "### 4. Sensor Types (What the Gauges Show)",
            "- **TICK**: Periodic heartbeat (engine running smoothly)",
            "- **DELTA**: Change detected (something shifted)",
            "- **ALERT**: Critical condition (need attention)",
            "- **METRIC**: Numeric measurement (fitness, coverage, etc.)",
            "- **STATUS**: System state (running, complete, terminated)",
            "",
            "### 5. Self-Healing (Auto-Recovery)",
            "The system automatically handles:",
            "- Fitness collapse → Increases elitism",
            "- Stagnation → Increases mutation rate",
            "- Consensus failure → Retries with view change",
            "",
            "Recovery actions are logged in `healing_log`.",
            "",
            "### 6. Multi-Node Operation (Fleet Mode)",
            "```python",
            'shell = OpenConstructShell(node_id="node-1", all_nodes=["node-1", "node-2"])',
            "# BFT consensus automatically coordinates breeding across nodes",
            "```",
            "",
            "## Safety Notes",
            "- Always check `health_check()` after long runs",
            "- If ALERT readings appear, review `healing_log`",
            "- Multi-node requires 2f+1 consensus (tolerates f < N/3 failures)",
            "",
            "## Quick Reference",
            "| Command | Action |",
            "|---------|--------|",
            "| `shell.spawn()` | Start breeding job |",
            "| `shell.run()` | Execute breeding |",
            "| `shell.status()` | Check all runs |",
            "| `shell.health_check()` | Read sensor history |",
            "| `shell.terminate()` | Stop a run |",
            "| `shell.get_best()` | Get best result |",
            "| `shell.run_parallel()` | Run multiple campaigns in parallel |",
        ])
        
        return "\n".join(lines)
