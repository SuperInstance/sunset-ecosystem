"""Parallel Breeding Orchestrator — Multi-campaign dispatch across fleet nodes.

Combines OpenConstructShell with BernsteinOrchestrator to run multiple
breeding campaigns in parallel, each in an isolated git worktree, with
HMAC-audited decisions and automatic result aggregation.

Usage
-----
    orch = ParallelBreedingOrchestrator(
        repo_path="/path/to/sunset-ecosystem",
        nodes=["node-1", "node-2", "node-3"],
    )

    # Define multiple campaigns
    campaigns = [
        Campaign("pythagorean-robust", "pythagorean",
                 {"population_size": 50, "genome_length": 10},
                 lambda m: float(np.sum(m))),
        Campaign("spectral-noise", "spectral",
                 {"population_size": 30, "spectrum_size": 64},
                 lambda s: float(np.max(np.abs(s)))),
    ]

    # Run all campaigns in parallel
    result = orch.run_parallel(campaigns, generations=10)

    # Aggregate best results
    best = result.best_campaign
    print(f"Winner: {best.name} with fitness {best.best_fitness}")

Architecture
------------
- Campaign: declarative breeding specification + task function
- ParallelBreedingOrchestrator: composes Bernstein + OpenConstructShell
- CampaignResult: per-campaign outcome with full sensor history
- ParallelResult: aggregate across all campaigns
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from fleet.openconstruct_shell import (
    OpenConstructShell,
    SensorReading,
    SensorType,
    SelfHealingLoop,
)
from fleet.bernstein_orchestrator import (
    BernsteinOrchestrator,
    OrchestratorConfig,
    SchedulerTask,
    ScheduleResult,
)
from fleet.openconstruct_bridge import ConstructManifest, HarnessAdapter

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────


@dataclass
class Campaign:
    """A single breeding campaign specification."""

    name: str
    attachment: str  # e.g., "pythagorean", "spectral"
    params: Dict[str, Any] = field(default_factory=dict)
    task_fn: Optional[Callable[[Any], float]] = None
    constraints: List[str] = field(default_factory=list)
    node_id: Optional[str] = None  # specific node, or None for auto-assign


@dataclass
class CampaignResult:
    """Outcome of a single campaign."""

    campaign: Campaign
    status: str  # "success", "failure", "timeout"
    run_id: str = ""
    best_fitness: float = 0.0
    best_genome: Any = None
    sensor_history: List[SensorReading] = field(default_factory=list)
    generations: int = 0
    duration: float = 0.0
    error: str = ""
    worktree_path: str = ""
    audit_hash: str = ""  # HMAC chain hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.campaign.name,
            "status": self.status,
            "best_fitness": self.best_fitness,
            "generations": self.generations,
            "duration": round(self.duration, 3),
            "error": self.error,
            "sensor_count": len(self.sensor_history),
            "audit_hash": self.audit_hash,
        }


@dataclass
class ParallelResult:
    """Aggregate result across all campaigns."""

    campaigns: List[CampaignResult] = field(default_factory=list)
    total_duration: float = 0.0
    audit_entries: int = 0
    _overall_status: str = field(default="pending", repr=False)

    @property
    def overall_status(self) -> str:
        """Auto-compute overall status from campaign results."""
        if not self.campaigns:
            return self._overall_status
        success_count = sum(1 for c in self.campaigns if c.status == "success")
        if success_count == len(self.campaigns):
            return "complete"
        elif success_count > 0:
            return "partial"
        else:
            return "failed"

    @overall_status.setter
    def overall_status(self, value: str) -> None:
        self._overall_status = value

    @property
    def best_campaign(self) -> Optional[CampaignResult]:
        """Return the campaign with highest best_fitness."""
        if not self.campaigns:
            return None
        succeeded = [c for c in self.campaigns if c.status == "success"]
        if not succeeded:
            return None
        return max(succeeded, key=lambda c: c.best_fitness)

    @property
    def success_rate(self) -> float:
        if not self.campaigns:
            return 0.0
        succeeded = sum(1 for c in self.campaigns if c.status == "success")
        return succeeded / len(self.campaigns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "campaign_count": len(self.campaigns),
            "success_rate": round(self.success_rate, 3),
            "total_duration": round(self.total_duration, 3),
            "best_campaign": self.best_campaign.to_dict()
            if self.best_campaign
            else None,
            "audit_entries": self.audit_entries,
            "campaigns": [c.to_dict() for c in self.campaigns],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ── Parallel Breeding Orchestrator ────────────────────────────


class ParallelBreedingOrchestrator:
    """Run multiple breeding campaigns in parallel across fleet nodes.

    Each campaign gets its own:
    - OpenConstructShell instance (isolated breeding state)
    - Git worktree (code isolation via Bernstein)
    - HMAC audit entry (tamper-evident log)

    Results are aggregated and the best campaign is identified.
    """

    def __init__(
        self,
        repo_path: str,
        nodes: List[str],
        max_workers: int = 4,
        default_generations: int = 10,
    ):
        self.repo_path = repo_path
        self.nodes = nodes
        self.max_workers = max_workers
        self.default_generations = default_generations

        # Bernstein orchestrator for parallel task scheduling
        self._bernstein = BernsteinOrchestrator(
            config=OrchestratorConfig(
                max_workers=max_workers,
                default_timeout=600.0,
                cleanup_on_success=True,
                cleanup_on_failure=True,
            )
        )

        # Node assignment round-robin tracker
        self._node_index = 0

    # ── Public API ────────────────────────────────────────────

    def run_parallel(
        self,
        campaigns: List[Campaign],
        generations: Optional[int] = None,
    ) -> ParallelResult:
        """Run all campaigns in parallel and aggregate results.

        Returns a ParallelResult with full per-campaign details.
        """
        gens = generations or self.default_generations
        start_time = time.perf_counter()

        # Build scheduler tasks from campaigns
        tasks = []
        for campaign in campaigns:
            node_id = campaign.node_id or self._next_node()
            task = self._build_task(campaign, node_id, gens)
            tasks.append(task)

        # Run via Bernstein orchestrator
        bernstein_result = self._bernstein.orchestrate(
            repo_path=self.repo_path,
            tasks=tasks,
        )

        # If aborted by GatewayPacing, return early
        if bernstein_result.get("aborted", False):
            return ParallelResult(
                campaigns=[],
                overall_status="aborted",
                total_duration=time.perf_counter() - start_time,
                audit_entries=bernstein_result.get("audit_entries", 0),
            )

        # Convert ScheduleResults to CampaignResults
        campaign_results = []
        scheduled = bernstein_result.get("scheduled", {})
        for campaign in campaigns:
            task_id = campaign.name
            schedule_result = scheduled.get(task_id)
            if schedule_result is None:
                # Build a failure result
                cr = CampaignResult(
                    campaign=campaign,
                    status="failure",
                    error="Task was not scheduled",
                )
            else:
                cr = self._convert_result(campaign, schedule_result)
            campaign_results.append(cr)

        total_duration = time.perf_counter() - start_time

        # Determine overall status
        success_count = sum(1 for c in campaign_results if c.status == "success")
        if success_count == len(campaigns):
            overall = "complete"
        elif success_count > 0:
            overall = "partial"
        else:
            overall = "failed"

        return ParallelResult(
            campaigns=campaign_results,
            total_duration=total_duration,
            audit_entries=bernstein_result.get("audit_entries", 0),
        )

    def run_parallel_streaming(
        self,
        campaigns: List[Campaign],
        generations: Optional[int] = None,
    ) -> Iterator[Tuple[str, SensorReading]]:
        """Run campaigns in parallel, yielding (campaign_name, sensor) events.

        This is the agent's "sensor feed" — it receives shadow readings
        from all active campaigns in real-time.
        """
        # For true streaming, we'd need async channels. For now, we
        # yield aggregated results as they complete.
        result = self.run_parallel(campaigns, generations)
        for cr in result.campaigns:
            for reading in cr.sensor_history:
                yield (cr.campaign.name, reading)

    def get_audit_chain(self) -> List[Dict[str, Any]]:
        """Return the HMAC audit chain as a list of dicts."""
        chain = self._bernstein.get_audit_chain()
        return [
            {
                "timestamp": e.timestamp,
                "decision_type": e.decision_type,
                "task_id": e.task_id,
                "details": e.details,
                "hash": e.compute_hash(),
            }
            for e in chain.entries
        ]

    def verify_audit_chain(self) -> Tuple[bool, int]:
        """Verify the HMAC audit chain."""
        return self._bernstein.get_audit_chain().verify_chain()

    # ── Internal ──────────────────────────────────────────────

    def _next_node(self) -> str:
        """Round-robin node assignment."""
        node = self.nodes[self._node_index % len(self.nodes)]
        self._node_index += 1
        return node

    def _build_task(
        self,
        campaign: Campaign,
        node_id: str,
        generations: int,
    ) -> SchedulerTask:
        """Build a SchedulerTask that runs a breeding campaign."""

        # Capture campaign in closure
        def _run_breeding() -> Dict[str, Any]:
            shell = OpenConstructShell(node_id=node_id, all_nodes=self.nodes)
            run_id = shell.spawn(
                campaign.attachment,
                name=campaign.name,
                **campaign.params,
            )

            # Run with sensor capture
            sensor_history = []
            start = time.perf_counter()

            try:
                for reading in shell.run(
                    run_id,
                    campaign.task_fn or (lambda g: 1.0),
                    generations=generations,
                ):
                    sensor_history.append(reading)

                duration = time.perf_counter() - start
                best_genome, best_fitness = shell.get_best(run_id)

                return {
                    "run_id": run_id,
                    "status": "success",
                    "best_fitness": best_fitness,
                    "best_genome": best_genome,
                    "sensor_history": sensor_history,
                    "generations": generations,
                    "duration": duration,
                    "error": "",
                }
            except Exception as exc:
                duration = time.perf_counter() - start
                return {
                    "run_id": run_id,
                    "status": "failure",
                    "best_fitness": 0.0,
                    "best_genome": None,
                    "sensor_history": sensor_history,
                    "generations": 0,
                    "duration": duration,
                    "error": str(exc),
                }

        return SchedulerTask(
            task_id=campaign.name,
            command=_run_breeding,
            expected_outputs=[],
            timeout=600.0,
            max_retries=1,
        )

    def _convert_result(
        self,
        campaign: Campaign,
        schedule_result: Dict[str, Any],
    ) -> CampaignResult:
        """Convert a Bernstein scheduled dict to a CampaignResult."""
        output = schedule_result.get("output") or {}
        if isinstance(output, dict):
            pass
        else:
            output = {}

        return CampaignResult(
            campaign=campaign,
            status=schedule_result.get("status", "failure"),
            run_id=output.get("run_id", ""),
            best_fitness=output.get("best_fitness", 0.0),
            best_genome=output.get("best_genome"),
            sensor_history=output.get("sensor_history", []),
            generations=output.get("generations", 0),
            duration=schedule_result.get("duration", 0.0),
            error=output.get("error", schedule_result.get("error", "")),
            worktree_path=schedule_result.get("worktree_path", ""),
            audit_hash=schedule_result.get("worktree_path", ""),
        )
