"""Tests for Beta-Test Persona framework (fleet/beta_test_personas.py).

Covers:
    - All 7 personas load correctly
    - simulate_discovery returns rating 1-5
    - Rating drops when README missing
    - Rating drops when no setup instructions
    - run_all_tests returns 7 results
    - generate_report includes all blockers
    - Time-to-first-success tracked
"""

from __future__ import annotations

import pytest

from fleet.beta_test_personas import (
    BetaTestResult,
    BetaTestRunner,
    BetaTestScenario,
    Persona,
    PersonaLibrary,
)


# ── 1. Persona library ──────────────────────────────────────

class TestPersonaLibrary:
    def test_all_seven_personas_exist(self):
        names = PersonaLibrary.names()
        expected = [
            "devops_engineer",
            "sre_oncall",
            "junior_developer",
            "security_auditor",
            "fleet_operator",
            "agent_developer",
            "infrastructure_engineer",
        ]
        for e in expected:
            assert e in names, f"Missing persona: {e}"

    def test_get_persona_by_name(self):
        p = PersonaLibrary.get_persona("junior_developer")
        assert p.name == "Junior Developer"
        assert p.expertise_level == 2

    def test_get_persona_by_display_name(self):
        p = PersonaLibrary.get_persona("DevOps Engineer")
        assert p.role == "infrastructure"
        assert p.expertise_level == 4

    def test_list_returns_all(self):
        personas = PersonaLibrary.list_personas()
        assert len(personas) == 7

    def test_unknown_persona_raises(self):
        with pytest.raises(KeyError, match="Unknown persona"):
            PersonaLibrary.get_persona("nonexistent")


# ── 2. simulate_discovery ─────────────────────────────────

class TestSimulateDiscovery:
    def test_perfect_repo_gets_5_stars(self):
        repo = {
            "name": "awesome-fleet",
            "readme": "Quickstart Deploy Architecture Multi-node Scale Plugin Build Health",
            "has_docker": True,
            "has_k8s": True,
            "has_runbook": True,
            "has_examples": True,
            "has_sbom": True,
            "has_secrets": False,
            "signed_releases": True,
            "has_discovery": True,
            "has_register_api": True,
            "api_version": "v2",
            "has_ci": True,
            "has_deps_list": True,
            "has_alerts": True,
        }
        result = BetaTestRunner.simulate_discovery(repo, persona_name="devops_engineer")
        assert isinstance(result, BetaTestResult)
        assert result.rating == 5
        assert result.blockers == []
        assert result.is_passing()

    def test_empty_repo_gets_1_star(self):
        repo = {"name": "empty", "readme": ""}
        result = BetaTestRunner.simulate_discovery(repo, persona_name="junior_developer")
        assert result.rating == 1
        assert len(result.blockers) == 3

    def test_rating_is_int_between_1_and_5(self):
        repo = {"name": "test", "readme": "something"}
        for name in PersonaLibrary.names():
            result = BetaTestRunner.simulate_discovery(repo, persona_name=name)
            assert isinstance(result.rating, int)
            assert 1 <= result.rating <= 5

    def test_time_to_first_success_is_float(self):
        repo = {"name": "test", "readme": "deploy"}
        result = BetaTestRunner.simulate_discovery(repo, persona_name="devops_engineer")
        assert isinstance(result.time_to_first_success, float)
        assert result.time_to_first_success >= 0

    def test_notes_include_persona_name(self):
        repo = {"name": "test", "readme": ""}
        result = BetaTestRunner.simulate_discovery(repo, persona_name="sre_oncall")
        assert "SRE On-Call" in result.notes

    def test_scenario_embeds_repo_metadata(self):
        repo = {"name": "test", "readme": ""}
        result = BetaTestRunner.simulate_discovery(repo, persona_name="fleet_operator")
        assert result.scenario.repo_name == "test"
        assert result.scenario.repo_metadata == repo


# ── 3. run_all_tests ──────────────────────────────────────

class TestRunAllTests:
    def test_returns_seven_results(self):
        repo = {"name": "test", "readme": "deploy runbook quickstart"}
        results = BetaTestRunner.run_all_tests(repo)
        assert len(results) == 7
        for r in results:
            assert isinstance(r, BetaTestResult)

    def test_each_result_has_different_persona(self):
        repo = {"name": "test", "readme": ""}
        results = BetaTestRunner.run_all_tests(repo)
        names = [r.persona.name for r in results]
        assert len(set(names)) == 7

    def test_repo_name_propagated(self):
        repo = {"name": "my-repo", "readme": ""}
        results = BetaTestRunner.run_all_tests(repo)
        for r in results:
            assert r.scenario.repo_name == "my-repo"


# ── 4. generate_report ────────────────────────────────────

class TestGenerateReport:
    def test_report_contains_all_personas(self):
        repo = {"name": "test", "readme": ""}
        results = BetaTestRunner.run_all_tests(repo)
        report = BetaTestRunner.generate_report(results)
        for persona in PersonaLibrary.list_personas():
            assert persona.name in report

    def test_report_has_summary_section(self):
        repo = {"name": "test", "readme": "deploy"}
        results = BetaTestRunner.run_all_tests(repo)
        report = BetaTestRunner.generate_report(results)
        assert "Summary" in report
        assert "Average rating" in report

    def test_report_has_recommendations_when_blockers_exist(self):
        repo = {"name": "test", "readme": ""}
        results = BetaTestRunner.run_all_tests(repo)
        report = BetaTestRunner.generate_report(results)
        assert "Recommendations" in report

    def test_report_is_markdown(self):
        repo = {"name": "test", "readme": ""}
        results = BetaTestRunner.run_all_tests(repo)
        report = BetaTestRunner.generate_report(results)
        assert report.startswith("# Beta-Test Persona Report")

    def test_report_shows_star_ratings(self):
        repo = {"name": "test", "readme": "deploy"}
        results = BetaTestRunner.run_all_tests(repo)
        report = BetaTestRunner.generate_report(results)
        assert "★" in report or "☆" in report


# ── 5. Specific persona checks ────────────────────────────

class TestPersonaSpecificChecks:
    def test_devops_needs_docker_or_k8s(self):
        repo_with = {"name": "t", "readme": "deploy health", "has_docker": True}
        repo_without = {"name": "t", "readme": "deploy health"}
        r1 = BetaTestRunner.simulate_discovery(repo_with, persona_name="devops_engineer")
        r2 = BetaTestRunner.simulate_discovery(repo_without, persona_name="devops_engineer")
        assert r1.rating > r2.rating

    def test_sre_needs_runbook(self):
        repo_with = {"name": "t", "readme": "/metrics", "has_runbook": True}
        repo_without = {"name": "t", "readme": "/metrics"}
        r1 = BetaTestRunner.simulate_discovery(repo_with, persona_name="sre_oncall")
        r2 = BetaTestRunner.simulate_discovery(repo_without, persona_name="sre_oncall")
        assert r1.rating > r2.rating

    def test_junior_needs_examples(self):
        repo_with = {"name": "t", "readme": "quickstart architecture", "has_examples": True}
        repo_without = {"name": "t", "readme": "quickstart architecture"}
        r1 = BetaTestRunner.simulate_discovery(repo_with, persona_name="junior_developer")
        r2 = BetaTestRunner.simulate_discovery(repo_without, persona_name="junior_developer")
        assert r1.rating > r2.rating

    def test_security_needs_sbom(self):
        repo_with = {"name": "t", "readme": "", "has_sbom": True, "has_secrets": False}
        repo_without = {"name": "t", "readme": "", "has_secrets": False}
        r1 = BetaTestRunner.simulate_discovery(repo_with, persona_name="security_auditor")
        r2 = BetaTestRunner.simulate_discovery(repo_without, persona_name="security_auditor")
        assert r1.rating > r2.rating

    def test_fleet_needs_discovery(self):
        repo_with = {"name": "t", "readme": "multi-node scale", "has_discovery": True}
        repo_without = {"name": "t", "readme": "multi-node scale"}
        r1 = BetaTestRunner.simulate_discovery(repo_with, persona_name="fleet_operator")
        r2 = BetaTestRunner.simulate_discovery(repo_without, persona_name="fleet_operator")
        assert r1.rating > r2.rating

    def test_agent_needs_register_api(self):
        repo_with = {"name": "t", "readme": "plugin", "has_register_api": True, "api_version": "v1"}
        repo_without = {"name": "t", "readme": "plugin"}
        r1 = BetaTestRunner.simulate_discovery(repo_with, persona_name="agent_developer")
        r2 = BetaTestRunner.simulate_discovery(repo_without, persona_name="agent_developer")
        assert r1.rating > r2.rating

    def test_infra_needs_ci(self):
        repo_with = {"name": "t", "readme": "build", "has_ci": True, "has_deps_list": True}
        repo_without = {"name": "t", "readme": "build"}
        r1 = BetaTestRunner.simulate_discovery(repo_with, persona_name="infrastructure_engineer")
        r2 = BetaTestRunner.simulate_discovery(repo_without, persona_name="infrastructure_engineer")
        assert r1.rating > r2.rating
