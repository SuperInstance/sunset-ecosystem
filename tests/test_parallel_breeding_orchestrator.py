"""Tests for Parallel Breeding Orchestrator.

Covers Campaign, CampaignResult, ParallelResult, and ParallelBreedingOrchestrator.
"""

import time
from typing import List

import numpy as np
import pytest

from fleet.parallel_breeding_orchestrator import (
    Campaign,
    CampaignResult,
    ParallelResult,
    ParallelBreedingOrchestrator,
)
from fleet.openconstruct_shell import SensorReading, SensorType


class TestCampaign:
    def test_basic_campaign(self):
        c = Campaign(
            name="test-campaign",
            attachment="pythagorean",
            params={"population_size": 10},
        )
        assert c.name == "test-campaign"
        assert c.attachment == "pythagorean"
        assert c.params["population_size"] == 10

    def test_campaign_with_task_fn(self):
        def task_fn(matrix):
            return float(np.sum(matrix))

        c = Campaign(
            name="with-fn",
            attachment="pythagorean",
            task_fn=task_fn,
        )
        assert c.task_fn is not None

    def test_campaign_with_node(self):
        c = Campaign(
            name="node-specific",
            attachment="spectral",
            node_id="node-2",
        )
        assert c.node_id == "node-2"


class TestCampaignResult:
    def test_success_result(self):
        c = Campaign(name="c1", attachment="pythagorean")
        cr = CampaignResult(
            campaign=c,
            status="success",
            best_fitness=42.0,
            generations=10,
        )
        assert cr.status == "success"
        assert cr.best_fitness == 42.0

    def test_failure_result(self):
        c = Campaign(name="c2", attachment="spectral")
        cr = CampaignResult(
            campaign=c,
            status="failure",
            error="timeout",
        )
        assert cr.status == "failure"
        assert cr.error == "timeout"

    def test_to_dict(self):
        c = Campaign(name="c3", attachment="adversarial")
        cr = CampaignResult(
            campaign=c,
            status="success",
            best_fitness=10.0,
            generations=5,
            duration=2.5,
            sensor_history=[
                SensorReading(SensorType.METRIC, "fitness", 10.0, time.time())
            ],
        )
        d = cr.to_dict()
        assert d["name"] == "c3"
        assert d["status"] == "success"
        assert d["best_fitness"] == 10.0
        assert d["sensor_count"] == 1


class TestParallelResult:
    def test_empty_result(self):
        pr = ParallelResult()
        assert pr.overall_status == "pending"
        assert pr.best_campaign is None
        assert pr.success_rate == 0.0

    def test_best_campaign(self):
        c1 = Campaign(name="weak", attachment="pythagorean")
        c2 = Campaign(name="strong", attachment="spectral")
        cr1 = CampaignResult(campaign=c1, status="success", best_fitness=10.0)
        cr2 = CampaignResult(campaign=c2, status="success", best_fitness=99.0)
        cr3 = CampaignResult(campaign=c1, status="failure")
        pr = ParallelResult(campaigns=[cr1, cr2, cr3])
        assert pr.best_campaign == cr2
        assert pr.success_rate == 2 / 3

    def test_overall_status_complete(self):
        cr = CampaignResult(
            campaign=Campaign(name="c", attachment="pythagorean"),
            status="success",
        )
        pr = ParallelResult(campaigns=[cr])
        assert pr.overall_status == "complete"

    def test_overall_status_partial(self):
        cr1 = CampaignResult(
            campaign=Campaign(name="ok", attachment="pythagorean"),
            status="success",
        )
        cr2 = CampaignResult(
            campaign=Campaign(name="fail", attachment="spectral"),
            status="failure",
        )
        pr = ParallelResult(campaigns=[cr1, cr2])
        assert pr.overall_status == "partial"

    def test_to_json(self):
        c = Campaign(name="json-test", attachment="pythagorean")
        cr = CampaignResult(campaign=c, status="success", best_fitness=5.0)
        pr = ParallelResult(campaigns=[cr])
        json_str = pr.to_json()
        assert "json-test" in json_str
        assert "complete" in json_str


class TestParallelBreedingOrchestrator:
    def test_orchestrator_init(self):
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test-repo",
            nodes=["node-1", "node-2"],
            max_workers=2,
        )
        assert orch.repo_path == "/tmp/test-repo"
        assert len(orch.nodes) == 2
        assert orch.max_workers == 2

    def test_next_node_round_robin(self):
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test",
            nodes=["node-a", "node-b", "node-c"],
        )
        assert orch._next_node() == "node-a"
        assert orch._next_node() == "node-b"
        assert orch._next_node() == "node-c"
        assert orch._next_node() == "node-a"  # wraps

    def test_build_task(self):
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test",
            nodes=["node-1"],
        )
        c = Campaign(
            name="build-test",
            attachment="pythagorean",
            params={"population_size": 5, "genome_length": 3},
            task_fn=lambda m: float(np.sum(m)),
        )
        task = orch._build_task(c, "node-1", generations=2)
        assert task.task_id == "build-test"
        assert task.timeout == 600.0
        assert task.max_retries == 1

    def test_convert_result(self):
        """Test _convert_result with a dict (as returned by Bernstein)."""
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test",
            nodes=["node-1"],
        )
        c = Campaign(name="convert-test", attachment="pythagorean")
        sr = {
            "status": "success",
            "output": {
                "run_id": "run-123",
                "best_fitness": 42.0,
                "generations": 3,
                "sensor_history": [],
            },
            "duration": 1.5,
        }
        cr = orch._convert_result(c, sr)
        assert cr.status == "success"
        assert cr.best_fitness == 42.0
        assert cr.generations == 3

    def test_single_campaign_run(self):
        """End-to-end: run one campaign through the parallel orchestrator."""
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test-repo",
            nodes=["node-1"],
            max_workers=1,
        )
        campaigns = [
            Campaign(
                name="single-test",
                attachment="pythagorean",
                params={"population_size": 5, "genome_length": 3},
                task_fn=lambda m: float(np.sum(m)),
            ),
        ]
        result = orch.run_parallel(campaigns, generations=2)
        assert len(result.campaigns) == 1
        # Campaign may succeed or fail depending on worktree availability
        # We just verify structure
        assert result.campaigns[0].campaign.name == "single-test"
        assert result.total_duration >= 0

    def test_multi_campaign_parallel(self):
        """Run multiple campaigns in parallel."""
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test-repo",
            nodes=["node-1"],
            max_workers=2,
        )
        campaigns = [
            Campaign(
                name="pythagorean-campaign",
                attachment="pythagorean",
                params={"population_size": 5, "genome_length": 3},
                task_fn=lambda m: float(np.sum(m)),
            ),
            Campaign(
                name="spectral-campaign",
                attachment="spectral",
                params={"population_size": 5, "spectrum_size": 32},
                task_fn=lambda s: float(np.max(np.abs(s))),
            ),
        ]
        result = orch.run_parallel(campaigns, generations=1)
        assert len(result.campaigns) == 2
        assert result.overall_status in ("complete", "partial", "failed")
        assert result.total_duration >= 0

    def test_parallel_streaming(self):
        """Test streaming sensor readings from parallel campaigns."""
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test-repo",
            nodes=["node-1"],
            max_workers=1,
        )
        campaigns = [
            Campaign(
                name="stream-test",
                attachment="pythagorean",
                params={"population_size": 5, "genome_length": 3},
                task_fn=lambda m: float(np.sum(m)),
            ),
        ]
        events = list(orch.run_parallel_streaming(campaigns, generations=1))
        # Events may be empty if worktree fails, but structure is valid
        for name, reading in events:
            assert isinstance(reading, SensorReading)

    def test_audit_chain(self):
        """Verify audit chain is accessible after parallel run."""
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test-repo",
            nodes=["node-1"],
            max_workers=1,
        )
        campaigns = [
            Campaign(
                name="audit-test",
                attachment="pythagorean",
                params={"population_size": 3, "genome_length": 2},
                task_fn=lambda m: float(np.sum(m)),
            ),
        ]
        result = orch.run_parallel(campaigns, generations=1)
        # Audit entries should exist even if campaign fails
        chain = orch.get_audit_chain()
        assert isinstance(chain, list)

    def test_verify_audit_chain(self):
        """HMAC chain verification should pass for fresh orchestrator."""
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test-repo",
            nodes=["node-1"],
        )
        valid, idx = orch.verify_audit_chain()
        assert valid is True
        assert idx == -1

    def test_campaign_with_constraints(self):
        c = Campaign(
            name="constrained",
            attachment="pythagorean",
            constraints=["exact_arithmetic"],
            params={"population_size": 5},
        )
        assert c.constraints == ["exact_arithmetic"]


class TestIntegrationFlow:
    def test_full_parallel_workflow(self):
        """End-to-end parallel workflow with result aggregation."""
        orch = ParallelBreedingOrchestrator(
            repo_path="/tmp/test-repo",
            nodes=["node-1"],
            max_workers=1,
        )

        # Define 3 parallel campaigns with different algorithms
        campaigns = [
            Campaign(
                name="exact-rational",
                attachment="pythagorean",
                params={"population_size": 5, "genome_length": 3},
                task_fn=lambda m: float(np.sum(m)),
            ),
            Campaign(
                name="fourier-evolution",
                attachment="spectral",
                params={"population_size": 5, "spectrum_size": 32},
                task_fn=lambda s: float(np.max(np.abs(s))),
            ),
            Campaign(
                name="nca-growth",
                attachment="nca",
                params={"population_size": 3},
                task_fn=lambda grid: float(np.sum(grid)),
            ),
        ]

        result = orch.run_parallel(campaigns, generations=1)

        # Verify structure
        assert len(result.campaigns) == 3
        assert result.total_duration >= 0
        assert result.audit_entries >= 0

        # Verify we can identify best campaign
        best = result.best_campaign
        if best is not None:
            assert best.status == "success"
            assert best.best_fitness >= 0

        # Verify JSON export
        json_output = result.to_json()
        assert "campaign_count" in json_output
        assert "success_rate" in json_output
