"""
Tests for SuperInstance Ecosystem Scanner.

Covers: RepositoryInfo, EcosystemScanner.
"""

import pytest

from fleet.ecosystem_scanner import RepositoryInfo, EcosystemScanner


class TestRepositoryInfo:
    def test_to_dict(self):
        info = RepositoryInfo(
            name="test-repo",
            owner="SuperInstance",
            description="A test repo",
            topics=["python", "ai"],
            language="Python",
            stars=100,
            has_tests=True,
        )
        d = info.to_dict()
        assert d["name"] == "test-repo"
        assert d["stars"] == 100
        assert d["has_tests"] is True
        assert d["compatibility"] == 0.0


class TestEcosystemScanner:
    def test_init(self):
        scanner = EcosystemScanner("SuperInstance")
        assert scanner.organization == "SuperInstance"
        assert len(scanner.KNOWN_REPOS) > 0

    def test_scan_mock(self):
        scanner = EcosystemScanner()
        repos = scanner.scan_repositories(mock=True)
        assert len(repos) == len(scanner.KNOWN_REPOS)
        assert all(isinstance(r, RepositoryInfo) for r in repos)

    def test_compute_compatibility_python(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(
            name="test", owner="test", language="Python", has_tests=True
        )
        score = scanner.compute_compatibility(repo)
        assert score > 0.3
        assert "Python codebase" in repo.compatibility_reasons

    def test_compute_compatibility_no_tests(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(
            name="test",
            owner="test",
            language="Python",
            has_tests=False,
            has_ci=False,
            has_docs=False,
        )
        score = scanner.compute_compatibility(repo)
        assert score < 0.5

    def test_compute_compatibility_rust(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(
            name="test", owner="test", language="Rust", has_tests=True
        )
        score = scanner.compute_compatibility(repo)
        assert score < 0.5  # Less than Python

    def test_compute_compatibility_ai_topic(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(
            name="test", owner="test", language="Python", topics=["ai", "evolution"]
        )
        score = scanner.compute_compatibility(repo)
        assert score >= 0.5
        assert "AI/evolution topic match" in repo.compatibility_reasons

    def test_find_compatible_repos(self):
        scanner = EcosystemScanner()
        repos = scanner.scan_repositories()
        compatible = scanner.find_compatible_repos(repos, threshold=0.0)
        assert len(compatible) <= len(repos)
        # Should be sorted by compatibility
        if len(compatible) >= 2:
            assert compatible[0].compatibility >= compatible[1].compatibility

    def test_find_compatible_repos_high_threshold(self):
        scanner = EcosystemScanner()
        repos = scanner.scan_repositories()
        compatible = scanner.find_compatible_repos(repos, threshold=1.0)
        assert len(compatible) == 0

    def test_recommend_integration_breeding(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(
            name="test", owner="test", topics=["evolution", "breeding"]
        )
        rec = scanner.recommend_integration(repo)
        assert rec == "swarm/breeder_daemon_v2.py"

    def test_recommend_integration_worldmodel(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(name="test", owner="test", topics=["rl", "world-model"])
        rec = scanner.recommend_integration(repo)
        assert rec == "fleet/worldmodel_projector.py"

    def test_recommend_integration_monitoring(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(
            name="test", owner="test", topics=["monitoring", "health"]
        )
        rec = scanner.recommend_integration(repo)
        assert rec == "fleet/cocapn_dashboard.py"

    def test_recommend_integration_default(self):
        scanner = EcosystemScanner()
        repo = RepositoryInfo(name="test", owner="test", topics=["misc"])
        rec = scanner.recommend_integration(repo)
        assert rec == "fleet/openconstruct_shell.py"

    def test_generate_integration_map(self):
        scanner = EcosystemScanner()
        data = scanner.generate_integration_map()
        assert data["organization"] == "SuperInstance"
        assert data["total_repos"] == len(scanner.KNOWN_REPOS)
        assert "compatible_repos" in data
        assert "recommended_priority" in data

    def test_export_markdown(self):
        scanner = EcosystemScanner()
        md = scanner.export_markdown()
        assert "SuperInstance Ecosystem Scan" in md
        assert "Compatible Repositories" in md
        assert "|" in md  # Table format

    def test_get_stats(self):
        scanner = EcosystemScanner()
        stats = scanner.get_stats()
        assert "total_repos" in stats
        assert "compatible_repos" in stats
        assert "avg_compatibility" in stats
        assert "languages" in stats

    def test_catalog_populated(self):
        scanner = EcosystemScanner()
        scanner.scan_repositories()
        assert len(scanner.catalog) > 0
        assert "sunset-ecosystem" in scanner.catalog
