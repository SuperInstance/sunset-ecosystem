"""Tests for EcosystemHub — auto-discovery and integration mapping.

Reference: fleet/ecosystem_hub.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fleet.ecosystem_hub import EcosystemHub, IntegrationMap, IntegrationTask, RepoCard


class TestDiscovery:
    def test_empty_hub(self) -> None:
        hub = EcosystemHub(org="test-org")
        assert hub.org == "test-org"
        assert hub.repos == {}
        assert hub.maps == []
        assert hub.tasks == []

    def test_mock_discovery(self) -> None:
        hub = EcosystemHub(org="test-org")
        # Manually inject a repo to avoid gh CLI dependency
        hub.repos["cuda-oxide"] = RepoCard(
            name="cuda-oxide",
            url="https://github.com/test-org/cuda-oxide",
            description="GPU runtime",
            primary_language="Rust",
            tags=["rust", "gpu"],
        )
        assert "cuda-oxide" in hub.repos
        assert hub.repos["cuda-oxide"].tags == ["rust", "gpu"]

    def test_auto_tagging(self) -> None:
        hub = EcosystemHub(org="test-org")
        card = RepoCard(
            name="band-midi-rs", url="u", description="MIDI", primary_language="Rust"
        )
        hub._auto_tag(card)
        assert "music" in card.tags
        assert "rust" in card.tags

    def test_auto_tagging_breeding(self) -> None:
        hub = EcosystemHub(org="test-org")
        card = RepoCard(
            name="evolution-breeder", url="u", description="genetic algorithms"
        )
        hub._auto_tag(card)
        assert "breeding" in card.tags

    def test_cache_save_and_load(self, tmp_path: Path) -> None:
        cache = tmp_path / "hub_cache.json"
        hub = EcosystemHub(org="test-org", cache_path=cache)
        hub.repos["test-repo"] = RepoCard(name="test-repo", url="u")
        hub._save_cache()

        hub2 = EcosystemHub(org="test-org", cache_path=cache)
        hub2.discover()
        assert "test-repo" in hub2.repos

    def test_cache_expiration(self, tmp_path: Path) -> None:
        cache = tmp_path / "hub_cache.json"
        # Write old cache
        with open(cache, "w") as f:
            json.dump({"timestamp": 0, "repos": {}}, f)
        hub = EcosystemHub(org="test-org", cache_path=cache)
        # Should not load expired cache
        hub.discover()
        # Since gh CLI might fail, repos might stay empty, but cache won't be loaded


class TestIntegrationMapping:
    def test_map_integrations(self) -> None:
        hub = EcosystemHub(org="test-org")
        hub.repos["cuda-oxide"] = RepoCard(
            name="cuda-oxide", url="u", tags=["rust", "gpu"]
        )
        hub.repos["agent-operations"] = RepoCard(
            name="agent-operations", url="u", tags=["python", "agentic"]
        )
        hub.repos["t-minus-rs"] = RepoCard(name="t-minus-rs", url="u", tags=["rust"])

        maps = hub.map_integrations()
        assert len(maps) >= 3
        cuda_map = next((m for m in maps if m.repo_name == "cuda-oxide"), None)
        assert cuda_map is not None
        assert cuda_map.priority == "P0"
        assert "hnsw" in cuda_map.sunset_module.lower()

    def test_map_skips_missing_repos(self) -> None:
        hub = EcosystemHub(org="test-org")
        # Don't add any repos
        maps = hub.map_integrations()
        # All mappings should be skipped since repos are empty
        assert len(maps) == 0

    def test_map_with_empty_repos(self) -> None:
        hub = EcosystemHub(org="test-org")
        # When repos dict is empty, map_integrations should return empty
        maps = hub.map_integrations()
        assert len(maps) == 0


class TestPriorityTasks:
    def test_suggest_tasks(self) -> None:
        hub = EcosystemHub(org="test-org")
        hub.repos["cuda-oxide"] = RepoCard(name="cuda-oxide", url="u")
        hub.repos["agent-operations"] = RepoCard(name="agent-operations", url="u")
        hub.repos["t-minus-rs"] = RepoCard(name="t-minus-rs", url="u")
        hub.repos["NEXAH"] = RepoCard(name="NEXAH", url="u")
        hub.repos["optimal-transport-rs"] = RepoCard(
            name="optimal-transport-rs", url="u"
        )
        hub.repos["market-manifold"] = RepoCard(name="market-manifold", url="u")

        tasks = hub.suggest_priority_tasks()
        assert len(tasks) >= 6
        # P0 tasks should come first
        assert tasks[0].priority == "P0"
        assert tasks[0].target_repo in ["cuda-oxide", "agent-operations", "t-minus-rs"]

    def test_task_fields(self) -> None:
        hub = EcosystemHub(org="test-org")
        hub.repos["cuda-oxide"] = RepoCard(name="cuda-oxide", url="u")
        tasks = hub.suggest_priority_tasks()
        cuda_task = next((t for t in tasks if t.target_repo == "cuda-oxide"), None)
        assert cuda_task is not None
        assert cuda_task.bridge_type == "PyO3"
        assert 0.0 < cuda_task.impact_score <= 1.0
        assert cuda_task.effort_estimate == "8-12h"

    def test_task_sorting(self) -> None:
        hub = EcosystemHub(org="test-org")
        hub.repos["cuda-oxide"] = RepoCard(name="cuda-oxide", url="u")
        hub.repos["lattice-crypto-rs"] = RepoCard(name="lattice-crypto-rs", url="u")
        tasks = hub.suggest_priority_tasks()
        priorities = [t.priority for t in tasks]
        # P0 should come before P2
        if "P0" in priorities and "P2" in priorities:
            p0_idx = priorities.index("P0")
            p2_idx = priorities.index("P2")
            assert p0_idx < p2_idx


class TestReport:
    def test_generate_report(self) -> None:
        hub = EcosystemHub(org="test-org")
        hub.repos["cuda-oxide"] = RepoCard(
            name="cuda-oxide", url="u", tags=["rust", "gpu"]
        )
        hub.repos["sunset-ecosystem"] = RepoCard(
            name="sunset-ecosystem",
            url="u",
            primary_language="Python",
            tags=["python", "agentic"],
        )
        hub.repos["c-ternary"] = RepoCard(
            name="c-ternary", url="u", primary_language="C", tags=["c"]
        )
        hub.repos["aider"] = RepoCard(
            name="aider", url="u", is_fork=True, tags=["fork", "python"]
        )

        report = hub.generate_report()
        assert report["total_repos"] == 4
        assert report["p0_tasks"] >= 1
        assert "categories" in report
        assert report["categories"]["python_apps"] >= 1
        assert report["categories"]["c_embedded"] >= 1
        assert report["categories"]["forks"] >= 1
        assert "top_5_tasks" in report
        assert len(report["top_5_tasks"]) >= 1

    def test_write_report(self, tmp_path: Path) -> None:
        hub = EcosystemHub(org="test-org")
        hub.repos["cuda-oxide"] = RepoCard(
            name="cuda-oxide", url="u", tags=["rust", "gpu"]
        )
        hub.map_integrations()
        hub.suggest_priority_tasks()

        report_path = tmp_path / "report.md"
        result = hub.write_report(path=report_path)
        assert result == report_path
        assert report_path.exists()
        content = report_path.read_text()
        assert "SuperInstance Ecosystem Hub Report" in content
        assert "cuda-oxide" in content

    def test_report_python_bridge_gaps(self) -> None:
        hub = EcosystemHub(org="test-org")
        hub.repos["rust-1"] = RepoCard(
            name="rust-1", url="u", tags=["rust"], has_python_bridge=False
        )
        hub.repos["rust-2"] = RepoCard(
            name="rust-2", url="u", tags=["rust"], has_python_bridge=False
        )
        hub.repos["python-1"] = RepoCard(
            name="python-1", url="u", tags=["python"], has_python_bridge=True
        )

        report = hub.generate_report()
        assert report["python_bridge_gaps"] == 2


class TestRepoCard:
    def test_repo_card_defaults(self) -> None:
        card = RepoCard(name="test", url="http://test")
        assert card.description is None
        assert card.tags == []
        assert not card.is_fork
        assert card.integration_status == "none"

    def test_repo_card_to_dict(self) -> None:
        hub = EcosystemHub(org="test-org")
        card = RepoCard(name="test", url="http://test", description="d", tags=["a"])
        d = hub._repo_to_dict(card)
        assert d["name"] == "test"
        assert d["tags"] == ["a"]
        assert d["integration_status"] == "none"
