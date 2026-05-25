#!/usr/bin/env python3
"""Beta-Test Persona Framework — simulated external visitors discovering fleet repos.

Seven personas (from behavioral synthesis) roleplay as outsiders who find our
repos on GitHub and try to use them for real-world projects. The framework
codifies this validation pattern so any repo can be persona-tested
automatically.

Reference: fleet/behavioral_synthesis.md (Beta-Test Personas pattern)
"""

from __future__ import annotations

__all__ = [
    "Persona",
    "BetaTestScenario",
    "BetaTestResult",
    "PersonaLibrary",
    "BetaTestRunner",
]

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ── data structures ───────────────────────────────────────────

@dataclass(frozen=True)
class Persona:
    """A simulated external visitor."""

    name: str
    role: str
    expertise_level: int  # 1–5
    goals: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    patience_seconds: float = 300.0  # how long before they give up


@dataclass(frozen=True)
class BetaTestScenario:
    """What the visitor is trying to do with the repo."""

    repo_name: str
    repo_metadata: dict[str, Any]
    task_description: str
    expected_outcome: str


@dataclass(frozen=True)
class BetaTestResult:
    """Outcome of one persona × scenario test."""

    persona: Persona
    scenario: BetaTestScenario
    rating: int  # 1–5
    blockers: list[str] = field(default_factory=list)
    time_to_first_success: float = 0.0  # seconds, 0 if never
    notes: str = ""

    def is_passing(self) -> bool:
        return self.rating >= 3


# ── persona library ───────────────────────────────────────────

class PersonaLibrary:
    """Pre-built personas from behavioral synthesis."""

    _PERSONAS: dict[str, Persona] = {
        "devops_engineer": Persona(
            name="DevOps Engineer",
            role="infrastructure",
            expertise_level=4,
            goals=[
                "Deploy to production in <30 minutes",
                "Health checks and monitoring out of the box",
                "Docker / k8s manifests",
            ],
            pain_points=[
                "Hardcoded hosts or ports",
                "Missing environment variable docs",
                "No observability hooks",
            ],
            success_criteria=[
                "README has deploy section",
                "docker-compose.yml or k8s/ folder exists",
                "Health endpoint documented",
            ],
            patience_seconds=600,
        ),
        "sre_oncall": Persona(
            name="SRE On-Call",
            role="reliability",
            expertise_level=5,
            goals=[
                "Understand failure modes fast",
                "Clear runbook for 3 AM pages",
                "Metrics exposed in Prometheus format",
            ],
            pain_points=[
                "Anonymous service names in logs",
                "No circuit breaker configuration",
                "Missing alert rule examples",
            ],
            success_criteria=[
                "Runbook in docs/RUNBOOK.md",
                "Metrics endpoint at /metrics",
                "Alert rules in alerts/ folder",
            ],
            patience_seconds=300,
        ),
        "junior_developer": Persona(
            name="Junior Developer",
            role="learning",
            expertise_level=2,
            goals=[
                "Run the project locally in <10 minutes",
                "Understand the architecture from README",
                "Find working examples",
            ],
            pain_points=[
                "No setup instructions",
                "Assumes deep domain knowledge",
                "Examples don't compile",
            ],
            success_criteria=[
                "README has quickstart",
                "examples/ folder with runnable code",
                "Architecture diagram or description",
            ],
            patience_seconds=900,
        ),
        "security_auditor": Persona(
            name="Security Auditor",
            role="security",
            expertise_level=5,
            goals=[
                "Audit supply chain in <1 hour",
                "Verify signing / provenance",
                "Check secret handling",
            ],
            pain_points=[
                "No SBOM or dependency list",
                "Secrets in code or config",
                "Unsigned releases",
            ],
            success_criteria=[
                "SBOM available",
                "No hardcoded secrets",
                "Signed commits or releases",
            ],
            patience_seconds=1200,
        ),
        "fleet_operator": Persona(
            name="Fleet Operator",
            role="orchestration",
            expertise_level=4,
            goals=[
                "Deploy across multiple nodes",
                "Understand node topology",
                "Scale without recompiling",
            ],
            pain_points=[
                "Single-node assumptions",
                "No node discovery mechanism",
                "Hardcoded topology",
            ],
            success_criteria=[
                "Multi-node config documented",
                "Node discovery or registry exists",
                "Horizontal scaling guide",
            ],
            patience_seconds=600,
        ),
        "agent_developer": Persona(
            name="Agent Developer",
            role="extension",
            expertise_level=3,
            goals=[
                "Write a plugin in <1 hour",
                "Clear API surface",
                "Good error messages",
            ],
            pain_points=[
                "No plugin API",
                "Internal APIs change without notice",
                "No type hints or docs",
            ],
            success_criteria=[
                "Plugin API in docs/",
                "register_plugin() or similar entry point",
                "Stable API versioning",
            ],
            patience_seconds=1200,
        ),
        "infrastructure_engineer": Persona(
            name="Infrastructure Engineer",
            role="platform",
            expertise_level=4,
            goals=[
                "Integrate with existing CI/CD",
                "Understand build dependencies",
                "Reproducible builds",
            ],
            pain_points=[
                "No CI config",
                "Undeclared system dependencies",
                "Non-reproducible builds",
            ],
            success_criteria=[
                ".github/workflows or similar CI",
                "requirements.txt / Cargo.toml / package.json clear",
                "Build docs in README",
            ],
            patience_seconds=600,
        ),
    }

    @classmethod
    def get_persona(cls, name: str) -> Persona:
        key = name.lower().replace(" ", "_")
        if key not in cls._PERSONAS:
            raise KeyError(f"Unknown persona: {name!r} (try one of {list(cls._PERSONAS.keys())})")
        return cls._PERSONAS[key]

    @classmethod
    def list_personas(cls) -> list[Persona]:
        return list(cls._PERSONAS.values())

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._PERSONAS.keys())


# ── beta test runner ──────────────────────────────────────────

class BetaTestRunner:
    """Simulate persona discovery and rate the experience."""

    # Weighted checks per persona type
    _CHECKS: dict[str, list[tuple[str, Callable[[dict], bool]]]] = {
        "devops_engineer": [
            ("README deploy section", lambda m: "deploy" in m.get("readme", "").lower()),
            ("Docker or k8s manifests", lambda m: m.get("has_docker", False) or m.get("has_k8s", False)),
            ("Health endpoint documented", lambda m: "health" in m.get("readme", "").lower()),
        ],
        "sre_oncall": [
            ("Runbook exists", lambda m: m.get("has_runbook", False)),
            ("Metrics endpoint", lambda m: "/metrics" in m.get("readme", "")),
            ("Alert rules", lambda m: m.get("has_alerts", False)),
        ],
        "junior_developer": [
            ("Quickstart in README", lambda m: "quickstart" in m.get("readme", "").lower()),
            ("Examples folder", lambda m: m.get("has_examples", False)),
            ("Architecture description", lambda m: "architecture" in m.get("readme", "").lower()),
        ],
        "security_auditor": [
            ("SBOM available", lambda m: m.get("has_sbom", False)),
            ("No hardcoded secrets", lambda m: not m.get("has_secrets", True)),
            ("Signed releases", lambda m: m.get("signed_releases", False)),
        ],
        "fleet_operator": [
            ("Multi-node config", lambda m: "multi-node" in m.get("readme", "").lower()),
            ("Node discovery", lambda m: m.get("has_discovery", False)),
            ("Scaling guide", lambda m: "scale" in m.get("readme", "").lower()),
        ],
        "agent_developer": [
            ("Plugin API docs", lambda m: "plugin" in m.get("readme", "").lower()),
            ("register function", lambda m: m.get("has_register_api", False)),
            ("API versioning", lambda m: m.get("api_version", "") != ""),
        ],
        "infrastructure_engineer": [
            ("CI config", lambda m: m.get("has_ci", False)),
            ("Clear dependency list", lambda m: m.get("has_deps_list", False)),
            ("Build docs", lambda m: "build" in m.get("readme", "").lower()),
        ],
    }

    @classmethod
    def simulate_discovery(
        cls,
        repo_metadata: dict[str, Any],
        persona: Persona | None = None,
        persona_name: str | None = None,
    ) -> BetaTestResult:
        """Simulate a persona discovering a repo on GitHub."""
        if persona is None:
            if persona_name is None:
                raise ValueError("Provide persona or persona_name")
            persona = PersonaLibrary.get_persona(persona_name)

        key = "_".join(persona.name.lower().replace("-", "").split())
        checks = cls._CHECKS.get(key, [])

        passed = 0
        blockers: list[str] = []
        t0 = time.perf_counter()

        for check_name, check_fn in checks:
            if check_fn(repo_metadata):
                passed += 1
            else:
                blockers.append(check_name)

        elapsed = time.perf_counter() - t0

        # Rating: 1 per check passed, max 5
        rating = min(passed + 1, 5) if checks else 3
        if not blockers:
            rating = 5
        elif len(blockers) >= len(checks):
            rating = 1

        # Time to first success: if they found at least one thing, estimate
        time_to_first = elapsed if passed > 0 else 0.0

        notes = f"{passed}/{len(checks)} checks passed. "
        if rating >= 4:
            notes += f"{persona.name} would likely adopt this repo."
        elif rating >= 3:
            notes += f"{persona.name} would adopt with minor friction."
        else:
            notes += f"{persona.name} would bounce to an alternative."

        return BetaTestResult(
            persona=persona,
            scenario=BetaTestScenario(
                repo_name=repo_metadata.get("name", "unknown"),
                repo_metadata=repo_metadata,
                task_description="Discovery and first-use simulation",
                expected_outcome="Successful onboarding",
            ),
            rating=rating,
            blockers=blockers,
            time_to_first_success=time_to_first,
            notes=notes,
        )

    @classmethod
    def run_all_tests(cls, repo_metadata: dict[str, Any]) -> list[BetaTestResult]:
        """Run all 7 personas against a repo."""
        results = []
        for persona in PersonaLibrary.list_personas():
            results.append(cls.simulate_discovery(repo_metadata, persona=persona))
        return results

    @classmethod
    def generate_report(cls, results: list[BetaTestResult]) -> str:
        """Generate a markdown report."""
        lines = [
            "# Beta-Test Persona Report",
            "",
            f"**Repo:** {results[0].scenario.repo_name}",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
            "",
            "## Summary",
            "",
        ]

        total = sum(r.rating for r in results)
        avg = total / len(results)
        passing = sum(1 for r in results if r.is_passing())

        lines.append(f"- **Average rating:** {avg:.1f}/5")
        lines.append(f"- **Passing personas:** {passing}/{len(results)}")
        lines.append("")

        # Ratings table
        lines.append("| Persona | Rating | Blockers | Time to First Success |")
        lines.append("|---------|--------|----------|----------------------|")
        for r in results:
            blockers_str = ", ".join(r.blockers) if r.blockers else "None"
            time_str = f"{r.time_to_first_success:.2f}s" if r.time_to_first_success > 0 else "N/A"
            lines.append(f"| {r.persona.name} | {'★' * r.rating}{'☆' * (5 - r.rating)} | {blockers_str} | {time_str} |")

        lines.append("")
        lines.append("## Detailed Notes")
        lines.append("")
        for r in results:
            lines.append(f"### {r.persona.name} ({r.persona.role}, level {r.persona.expertise_level})")
            lines.append(f"- **Rating:** {r.rating}/5")
            lines.append(f"- **Blockers:** {', '.join(r.blockers) if r.blockers else 'None'}")
            lines.append(f"- **Notes:** {r.notes}")
            lines.append("")

        # Recommendations
        all_blockers: list[str] = []
        for r in results:
            all_blockers.extend(r.blockers)

        from collections import Counter
        common = Counter(all_blockers).most_common(5)
        if common:
            lines.append("## Top Recommendations")
            lines.append("")
            for blocker, count in common:
                lines.append(f"- **{blocker}** — affects {count} personas")
            lines.append("")

        return "\n".join(lines)
