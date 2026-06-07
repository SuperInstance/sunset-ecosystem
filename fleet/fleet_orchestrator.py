"""FleetOrchestrator — Central conductor for the sunset-ecosystem fleet.

An emergent application that orchestrates all 21 fleet modules into a
unified operational system. The conductor manages health, schedules tasks,
optimizes configurations, distributes work, and maintains fleet coherence.

Think of this as the fleet's nervous system: every module reports in,
every decision is coordinated, every anomaly is escalated.

Usage
-----
    from fleet.fleet_orchestrator import FleetOrchestrator

    conductor = FleetOrchestrator(node_id="alpha")
    conductor.initialize_fleet()

    # Beat the fleet heart — one operational cycle
    conductor.beat()

    # Check fleet health
    health = conductor.check_fleet_health()
    print(f"Fleet status: {health['ternary_emoji']} {health['ternary_score']}")

    # Generate and execute tasks
    tasks = conductor.generate_tasks()
    conductor.execute_tasks(tasks)
"""

from __future__ import annotations

__all__ = [
    "FleetOrchestrator",
    "FleetBeat",
    "TaskSpec",
    "ExecutionResult",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fleet.breed_optimizer import BreedOptimizer
from fleet.cognitive_cache import CognitiveCache
from fleet.harbor import Harbor, ModuleEntry
from fleet.pattern_mine import PatternMine
from fleet.t_minus_bridge import TMinusBridge
from fleet.ternary_types import TernaryValue
from swarm.vector_swarm import VectorSwarm

logger = logging.getLogger(__name__)


@dataclass
class FleetBeat:
    """A single heartbeat of the fleet."""

    timestamp: float
    node_id: str
    cycle_number: int
    health_status: int  # ternary
    tasks_executed: int = 0
    tasks_failed: int = 0
    modules_checked: int = 0
    anomalies_found: int = 0
    duration_ms: float = 0.0


@dataclass
class TaskSpec:
    """A task specification for fleet execution."""

    task_id: str
    task_type: str  # health_check, optimize, breed, sync, mine, predict
    target_module: str | None = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    deadline_secs: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class ExecutionResult:
    """Result of task execution."""

    task_id: str
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


class FleetOrchestrator:
    """Central conductor for the fleet.

    Parameters
    ----------
    node_id : str
        Node identifier.
    workspace : str
        Path to sunset-ecosystem workspace.
    """

    def __init__(self, node_id: str = "default", workspace: str = ".") -> None:
        self.node_id = node_id
        self.workspace = workspace
        self.cycle_number = 0
        self.beats: list[FleetBeat] = []
        self.task_history: list[ExecutionResult] = []
        self._initialized = False

        # Subsystems (lazy initialization)
        self._harbor: Harbor | None = None
        self._tminus: TMinusBridge | None = None
        self._pattern_mine: PatternMine | None = None
        self._breed_optimizer: BreedOptimizer | None = None
        self._cognitive_cache: CognitiveCache | None = None
        self._vector_swarm: VectorSwarm | None = None

    # ── Initialization ───────────────────────────────────────

    def initialize_fleet(self) -> None:
        """Initialize all fleet subsystems."""
        logger.info("Initializing fleet orchestrator (node=%s)", self.node_id)

        self._harbor = Harbor(self.workspace)
        self._harbor.bootstrap_fleet()

        # Try to initialize optional subsystems
        try:
            self._tminus = TMinusBridge()
        except FileNotFoundError:
            logger.warning("TMinusBridge not available")

        try:
            self._pattern_mine = PatternMine()
        except Exception:
            logger.warning("PatternMine not available")

        try:
            self._breed_optimizer = BreedOptimizer(node_id=self.node_id)
        except Exception:
            logger.warning("BreedOptimizer not available")

        try:
            self._cognitive_cache = CognitiveCache()
        except Exception:
            logger.warning("CognitiveCache not available")

        try:
            self._vector_swarm = VectorSwarm(self.node_id)
        except Exception:
            logger.warning("VectorSwarm not available")

        self._initialized = True
        logger.info("Fleet initialization complete")

    # ── Fleet Beat ─────────────────────────────────────────

    def beat(self) -> FleetBeat:
        """Execute one operational cycle of the fleet.

        This is the main heartbeat function that:
        1. Checks fleet health
        2. Generates tasks
        3. Executes high-priority tasks
        4. Records the beat

        Returns
        -------
        FleetBeat
            Record of this beat.
        """
        start = time.time()
        self.cycle_number += 1

        if not self._initialized:
            self.initialize_fleet()

        logger.info("Fleet beat %d starting", self.cycle_number)

        # Phase 1: Health check
        health_status = self._check_health_phase()
        modules_checked = len(self._harbor.modules) if self._harbor else 0

        # Phase 2: Generate tasks based on health
        tasks = self._generate_tasks_phase(health_status)

        # Phase 3: Execute tasks
        executed = 0
        failed = 0
        for task in tasks:
            if task.priority >= 3:  # High/critical priority only in beat
                result = self._execute_task(task)
                if result.success:
                    executed += 1
                else:
                    failed += 1
                self.task_history.append(result)

        # Phase 4: Anomaly detection
        anomalies = self._detect_anomalies_phase()

        # Phase 5: Record beat
        duration_ms = (time.time() - start) * 1000
        beat = FleetBeat(
            timestamp=start,
            node_id=self.node_id,
            cycle_number=self.cycle_number,
            health_status=health_status,
            tasks_executed=executed,
            tasks_failed=failed,
            modules_checked=modules_checked,
            anomalies_found=anomalies,
            duration_ms=duration_ms,
        )
        self.beats.append(beat)

        logger.info(
            "Fleet beat %d complete: health=%s, tasks=%d/%d, anomalies=%d, duration=%.1fms",
            self.cycle_number,
            TernaryValue.to_string(health_status),
            executed,
            executed + failed,
            anomalies,
            duration_ms,
        )

        return beat

    # ── Health Check ───────────────────────────────────────

    def check_fleet_health(self) -> dict[str, Any]:
        """Check overall fleet health.

        Returns comprehensive health report with ternary classification.
        """
        if not self._harbor:
            return {"error": "Fleet not initialized"}

        report = self._harbor.generate_fleet_report()
        return {
            "node_id": self.node_id,
            "cycle_number": self.cycle_number,
            **report,
        }

    def _check_health_phase(self) -> int:
        """Internal health check phase. Returns ternary health status."""
        if not self._harbor:
            return TernaryValue.NEG

        report = self._harbor.generate_fleet_report()
        critical = report.get("critical", 0)
        degraded = report.get("degraded", 0)
        total = report.get("total_modules", 0)

        if critical > 0:
            return TernaryValue.NEG
        if degraded > total * 0.2:
            return TernaryValue.ZERO
        return TernaryValue.POS

    # ── Task Generation ──────────────────────────────────────

    def generate_tasks(self) -> list[TaskSpec]:
        """Generate tasks based on fleet state.

        Returns
        -------
        list[TaskSpec]
            Tasks sorted by priority (highest first).
        """
        if not self._initialized:
            self.initialize_fleet()

        tasks: list[TaskSpec] = []
        health = self.check_fleet_health()

        # Critical tasks: modules in critical state
        if health.get("critical", 0) > 0:
            for detail in health.get("module_details", []):
                if "🔴" in detail.get("health", ""):
                    tasks.append(TaskSpec(
                        task_id=f"critical-{detail['name']}-{self.cycle_number}",
                        task_type="health_check",
                        target_module=detail["name"],
                        priority=4,
                        payload={"reason": "critical health status"},
                    ))

        # High priority: integration gaps
        if self._harbor:
            gaps = self._harbor.find_integration_gaps()
            for gap in gaps[:3]:
                tasks.append(TaskSpec(
                    task_id=f"gap-{gap['source']}-{gap['target']}-{self.cycle_number}",
                    task_type="sync",
                    priority=3,
                    payload={"gap": gap},
                ))

        # Medium priority: pattern mining
        if self._pattern_mine:
            tasks.append(TaskSpec(
                task_id=f"mine-{self.cycle_number}",
                task_type="mine",
                priority=2,
                payload={"operation": "extract_patterns"},
            ))

        # Low priority: breeding optimization
        if self._breed_optimizer:
            tasks.append(TaskSpec(
                task_id=f"breed-{self.cycle_number}",
                task_type="breed",
                priority=1,
                payload={"operation": "optimize_archive"},
            ))

        # Schedule deadlines if tminus available
        if self._tminus and tasks:
            for task in tasks:
                if task.priority == 4:
                    task.deadline_secs = self._tminus.propagate_deadline(60.0, 30.0)
                elif task.priority == 3:
                    task.deadline_secs = self._tminus.propagate_deadline(300.0, 120.0)

        return sorted(tasks, key=lambda t: t.priority, reverse=True)

    def _generate_tasks_phase(self, health_status: int) -> list[TaskSpec]:
        """Internal task generation phase."""
        return self.generate_tasks()

    # ── Task Execution ───────────────────────────────────────

    def execute_task(self, task: TaskSpec) -> ExecutionResult:
        """Execute a single task.

        Parameters
        ----------
        task : TaskSpec
            Task to execute.

        Returns
        -------
        ExecutionResult
            Result of execution.
        """
        start = time.time()
        task.status = "running"

        try:
            output = self._execute_task_inner(task)
            duration_ms = (time.time() - start) * 1000
            task.status = "completed"
            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            task.status = "failed"
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _execute_task(self, task: TaskSpec) -> ExecutionResult:
        """Internal task execution wrapper."""
        return self.execute_task(task)

    def _execute_task_inner(self, task: TaskSpec) -> dict[str, Any]:
        """Execute task based on type."""
        if task.task_type == "health_check":
            return self._execute_health_check(task)
        elif task.task_type == "optimize":
            return self._execute_optimize(task)
        elif task.task_type == "breed":
            return self._execute_breed(task)
        elif task.task_type == "sync":
            return self._execute_sync(task)
        elif task.task_type == "mine":
            return self._execute_mine(task)
        elif task.task_type == "predict":
            return self._execute_predict(task)
        else:
            return {"status": "unknown_task_type", "task_type": task.task_type}

    def _execute_health_check(self, task: TaskSpec) -> dict[str, Any]:
        """Execute health check task."""
        if self._harbor and task.target_module:
            health = self._harbor.get_module_health(task.target_module)
            return {"health": health, "module": task.target_module}
        return {"status": "no_harbor_or_target"}

    def _execute_optimize(self, task: TaskSpec) -> dict[str, Any]:
        """Execute optimization task."""
        if self._breed_optimizer:
            archive = self._breed_optimizer.optimize_archive(iterations=10)
            return {"archive_coverage": archive.coverage, "qd_score": archive.qd_score}
        return {"status": "no_breed_optimizer"}

    def _execute_breed(self, task: TaskSpec) -> dict[str, Any]:
        """Execute breeding task."""
        if self._breed_optimizer:
            # Use dummy pool for demonstration
            pool = [
                {"id": f"agent_{i}", "traits": [0.1 + i * 0.1, 0.2 + i * 0.05]}
                for i in range(5)
            ]
            parents = self._breed_optimizer.select_parents(pool, k=2)
            return {"parents_selected": len(parents), "diversity_scores": [p.diversity_score for p in parents]}
        return {"status": "no_breed_optimizer"}

    def _execute_sync(self, task: TaskSpec) -> dict[str, Any]:
        """Execute synchronization task."""
        gap = task.payload.get("gap", {})
        return {"gap": gap, "action": "sync_initiated"}

    def _execute_mine(self, task: TaskSpec) -> dict[str, Any]:
        """Execute pattern mining task."""
        if self._pattern_mine:
            rules = self._pattern_mine.to_fleet_monitor_rules()
            templates = self._pattern_mine.to_task_templates()
            return {
                "rules_found": len(rules),
                "templates_found": len(templates),
                "rule_types": list(set(r.component for r in rules)),
            }
        return {"status": "no_pattern_mine"}

    def _execute_predict(self, task: TaskSpec) -> dict[str, Any]:
        """Execute prediction task."""
        if self._cognitive_cache:
            return {"status": "prediction_not_implemented"}
        return {"status": "no_cognitive_cache"}

    # ── Anomaly Detection ───────────────────────────────────

    def detect_anomalies(self) -> list[dict[str, Any]]:
        """Detect fleet-wide anomalies.

        Returns
        -------
        list[dict]
            Detected anomalies with details.
        """
        anomalies: list[dict[str, Any]] = []

        if not self._harbor:
            return anomalies

        # Check for critical modules
        report = self._harbor.generate_fleet_report()
        for detail in report.get("module_details", []):
            if "🔴" in detail.get("health", ""):
                anomalies.append({
                    "type": "critical_module",
                    "module": detail["name"],
                    "severity": 4,
                })

        # Check for repeated beat failures
        if len(self.beats) >= 3:
            recent = self.beats[-3:]
            if all(b.tasks_failed > b.tasks_executed for b in recent):
                anomalies.append({
                    "type": "beat_failure_streak",
                    "severity": 3,
                    "cycles": [b.cycle_number for b in recent],
                })

        # Check for health degradation trend
        if len(self.beats) >= 5:
            recent_health = [b.health_status for b in self.beats[-5:]]
            if all(h == TernaryValue.NEG for h in recent_health):
                anomalies.append({
                    "type": "health_degradation",
                    "severity": 4,
                    "cycles": [b.cycle_number for b in self.beats[-5:]],
                })

        return anomalies

    def _detect_anomalies_phase(self) -> int:
        """Internal anomaly detection phase. Returns count of anomalies."""
        return len(self.detect_anomalies())

    # ── Statistics ───────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get conductor statistics."""
        if not self.beats:
            return {"beats": 0, "cycles": 0, "mean_duration_ms": 0.0}

        durations = [b.duration_ms for b in self.beats]
        total_tasks = sum(b.tasks_executed + b.tasks_failed for b in self.beats)
        total_failed = sum(b.tasks_failed for b in self.beats)

        return {
            "beats": len(self.beats),
            "cycles": self.cycle_number,
            "mean_duration_ms": sum(durations) / len(durations),
            "total_tasks_executed": total_tasks - total_failed,
            "total_tasks_failed": total_failed,
            "success_rate": (total_tasks - total_failed) / total_tasks if total_tasks > 0 else 0.0,
            "current_health": TernaryValue.to_string(self.beats[-1].health_status) if self.beats else "UNKNOWN",
        }

    # ── Reports ──────────────────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive fleet report."""
        health = self.check_fleet_health()
        stats = self.get_stats()
        anomalies = self.detect_anomalies()

        return {
            "node_id": self.node_id,
            "cycle_number": self.cycle_number,
            "initialized": self._initialized,
            "health": health,
            "stats": stats,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "recent_beats": [
                {
                    "cycle": b.cycle_number,
                    "health": TernaryValue.to_string(b.health_status),
                    "tasks": f"{b.tasks_executed}/{b.tasks_executed + b.tasks_failed}",
                    "duration_ms": round(b.duration_ms, 1),
                }
                for b in self.beats[-5:]
            ],
            "subsystems": {
                "harbor": self._harbor is not None,
                "tminus": self._tminus is not None,
                "pattern_mine": self._pattern_mine is not None,
                "breed_optimizer": self._breed_optimizer is not None,
                "cognitive_cache": self._cognitive_cache is not None,
                "vector_swarm": self._vector_swarm is not None,
            },
        }
