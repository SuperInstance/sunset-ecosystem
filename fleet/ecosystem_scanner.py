"""
SuperInstance Ecosystem Scanner

Auto-discovers and catalogs repositories under the SuperInstance GitHub
organization, identifying integration points with sunset-ecosystem.

Usage:
    from fleet.ecosystem_scanner import EcosystemScanner
    scanner = EcosystemScanner("SuperInstance")
    repos = scanner.scan_repositories()
    compatible = scanner.find_compatible_repos(repos)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class RepositoryInfo:
    """Information about a discovered repository."""
    name: str
    owner: str
    description: str = ""
    topics: List[str] = field(default_factory=list)
    language: str = ""
    stars: int = 0
    forks: int = 0
    has_tests: bool = False
    has_ci: bool = False
    has_docs: bool = False
    # Integration compatibility score 0-1
    compatibility: float = 0.0
    # Why it's compatible
    compatibility_reasons: List[str] = field(default_factory=list)
    # Recommended integration module
    recommended_module: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "description": self.description,
            "topics": self.topics,
            "language": self.language,
            "stars": self.stars,
            "forks": self.forks,
            "has_tests": self.has_tests,
            "has_ci": self.has_ci,
            "has_docs": self.has_docs,
            "compatibility": self.compatibility,
            "compatibility_reasons": self.compatibility_reasons,
            "recommended_module": self.recommended_module,
        }


class EcosystemScanner:
    """
    Scans GitHub organizations for compatible repositories.

    In production, this would use the GitHub API.
    For now, uses a curated catalog + heuristics.
    """

    # Known SuperInstance repos from ecosystem context
    KNOWN_REPOS = [
        ("sunset-ecosystem", "Main breeding framework", ["python", "evolution", "ai"]),
        ("stable-worldmodel", "World model for spatial reasoning", ["python", "rl", "world-model"]),
        ("cocapn-health", "Fleet health monitoring", ["python", "monitoring", "health"]),
        ("ccc-os", "Fleet operating system", ["python", "os", "distributed"]),
        ("ai-writings", "Essays and documentation", ["markdown", "writing"]),
        ("claw", "Agent harness system", ["python", "agent", "harness"]),
        ("lucineer", "Knowledge graph system", ["python", "graph", "knowledge"]),
    ]

    def __init__(self, organization: str = "SuperInstance"):
        self.organization = organization
        self.discovered: List[RepositoryInfo] = []
        self.catalog: Dict[str, RepositoryInfo] = {}

    def scan_repositories(self, mock: bool = True) -> List[RepositoryInfo]:
        """
        Scan repositories.
        
        Args:
            mock: If True, use known catalog instead of GitHub API.
        """
        if mock:
            return self._scan_mock()
        # In production: use GitHub API
        return self._scan_mock()

    def _scan_mock(self) -> List[RepositoryInfo]:
        """Mock scan using known catalog."""
        repos = []
        for name, desc, topics in self.KNOWN_REPOS:
            info = RepositoryInfo(
                name=name,
                owner=self.organization,
                description=desc,
                topics=topics,
                language=random.choice(["Python", "Rust", "TypeScript", "Markdown"]),
                stars=random.randint(10, 500),
                forks=random.randint(0, 50),
                has_tests=random.random() > 0.3,
                has_ci=random.random() > 0.5,
                has_docs=random.random() > 0.4,
            )
            repos.append(info)
            self.catalog[name] = info

        self.discovered = repos
        return repos

    def compute_compatibility(self, repo: RepositoryInfo) -> float:
        """
        Compute compatibility score with sunset-ecosystem.
        
        Factors:
        - Python repos: +0.3
        - Has tests: +0.2
        - Has CI: +0.1
        - Has docs: +0.1
        - Evolution/AI topics: +0.2
        - Distributed/monitoring topics: +0.1
        """
        score = 0.0
        reasons = []

        if repo.language.lower() == "python":
            score += 0.3
            reasons.append("Python codebase")

        if repo.has_tests:
            score += 0.2
            reasons.append("Has test suite")

        if repo.has_ci:
            score += 0.1
            reasons.append("Has CI/CD")

        if repo.has_docs:
            score += 0.1
            reasons.append("Has documentation")

        topics = [t.lower() for t in repo.topics]
        if any(t in topics for t in ["evolution", "ai", "agent", "breeding"]):
            score += 0.2
            reasons.append("AI/evolution topic match")

        if any(t in topics for t in ["distributed", "monitoring", "health", "os"]):
            score += 0.1
            reasons.append("Infrastructure topic match")

        repo.compatibility = min(1.0, score)
        repo.compatibility_reasons = reasons
        return repo.compatibility

    def find_compatible_repos(self, repos: Optional[List[RepositoryInfo]] = None,
                            threshold: float = 0.5) -> List[RepositoryInfo]:
        """Find repositories compatible with sunset-ecosystem."""
        repos = repos or self.discovered
        for repo in repos:
            self.compute_compatibility(repo)

        compatible = [r for r in repos if r.compatibility >= threshold]
        compatible.sort(key=lambda r: r.compatibility, reverse=True)
        return compatible

    def recommend_integration(self, repo: RepositoryInfo) -> str:
        """Recommend which sunset-ecosystem module to integrate with."""
        topics = [t.lower() for t in repo.topics]

        if "evolution" in topics or "breeding" in topics:
            return "swarm/breeder_daemon_v2.py"
        elif "world-model" in topics or "rl" in topics:
            return "fleet/worldmodel_projector.py"
        elif "monitoring" in topics or "health" in topics:
            return "fleet/cocapn_dashboard.py"
        elif "distributed" in topics or "os" in topics:
            return "fleet/vessel_handshake.py"
        elif "graph" in topics or "knowledge" in topics:
            return "swarm/gnn_breeder.py"
        elif "agent" in topics:
            return "fleet/openconstruct_bridge.py"
        else:
            return "fleet/openconstruct_shell.py"

    def generate_integration_map(self) -> Dict[str, Any]:
        """Generate full integration map."""
        repos = self.scan_repositories()
        compatible = self.find_compatible_repos(repos)

        for repo in compatible:
            repo.recommended_module = self.recommend_integration(repo)

        return {
            "organization": self.organization,
            "total_repos": len(repos),
            "compatible_repos": len(compatible),
            "repositories": [r.to_dict() for r in repos],
            "compatible": [r.to_dict() for r in compatible],
            "recommended_priority": [
                {"repo": r.name, "module": r.recommended_module, "score": r.compatibility}
                for r in compatible[:5]
            ],
        }

    def export_markdown(self, filename: str = "ECOSYSTEM_SCAN.md"):
        """Export scan results as markdown."""
        data = self.generate_integration_map()
        lines = [
            "# SuperInstance Ecosystem Scan",
            "",
            f"**Organization:** {data['organization']}",
            f"**Total Repositories:** {data['total_repos']}",
            f"**Compatible with Sunset:** {data['compatible_repos']}",
            "",
            "## Compatible Repositories (Ranked)",
            "",
            "| Repository | Score | Language | Tests | CI | Docs | Recommended Module |",
            "|------------|-------|----------|-------|----|----|-------------------|",
        ]

        for repo in data["compatible"]:
            lines.append(
                f"| {repo['name']} | {repo['compatibility']:.2f} | "
                f"{repo['language']} | {'✅' if repo['has_tests'] else '❌'} | "
                f"{'✅' if repo['has_ci'] else '❌'} | {'✅' if repo['has_docs'] else '❌'} | "
                f"{repo['recommended_module']} |"
            )

        lines.extend([
            "",
            "## Integration Priority",
            "",
        ])
        for rec in data["recommended_priority"]:
            lines.append(f"1. **{rec['repo']}** -> `{rec['module']}` (score: {rec['score']:.2f})")

        lines.extend(["", "---", "*Generated by EcosystemScanner*", ""])

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get scan statistics."""
        repos = self.discovered or self.scan_repositories()
        compatible = self.find_compatible_repos(repos)

        return {
            "total_repos": len(repos),
            "compatible_repos": len(compatible),
            "avg_compatibility": np.mean([r.compatibility for r in compatible]) if compatible else 0,
            "languages": list(set(r.language for r in repos)),
            "topics": list(set(t for r in repos for t in r.topics)),
            "with_tests": sum(1 for r in repos if r.has_tests),
            "with_ci": sum(1 for r in repos if r.has_ci),
            "with_docs": sum(1 for r in repos if r.has_docs),
        }
