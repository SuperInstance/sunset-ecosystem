"""Tests for BetaTestPersonas — simulated external visitors rating fleet repos.

Covers Persona dataclass, PersonaLibrary, BetaTestRunner simulate_discovery,
run_all_tests, and markdown report generation.
"""

import pytest

from fleet.beta_test_personas import (
    BetaTestResult,
    BetaTestRunner,
    BetaTestScenario,
    Persona,
    PersonaLibrary,
)


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

class TestPersona:
    def test_defaults(self):
        p = Persona(name="Test", role="tester", expertise_level=3)
        assert p.goals == []
        assert p.patience_seconds == 300.0


# ---------------------------------------------------------------------------
# BetaTestResult
# ---------------------------------------------------------------------------

class TestBetaTestResult:
    def test_is_passing_true(self):
        p = Persona(name="T", role="r", expertise_level=3)
        s = BetaTestScenario(repo_name="x", repo_metadata={}, task_description="t", expected_outcome="ok")
        r = BetaTestResult(persona=p, scenario=s, rating=3)
        assert r.is_passing()

    def test_is_passing_false(self):
        p = Persona(name="T", role="r", expertise_level=3)
        s = BetaTestScenario(repo_name="x", repo_metadata={}, task_description="t", expected_outcome="ok")
        r = BetaTestResult(persona=p, scenario=s, rating=2)
        assert not r.is_passing()


# ---------------------------------------------------------------------------
# PersonaLibrary
# ---------------------------------------------------------------------------

class TestPersonaLibrary:
    def test_get_devops(self):
        p = PersonaLibrary.get_persona("devops_engineer")
        assert p.name == "DevOps Engineer"
        assert p.role == "infrastructure"

    def test_get_by_display_name(self):
        p = PersonaLibrary.get_persona("DevOps Engineer")
        assert p.name == "DevOps Engineer"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            PersonaLibrary.get_persona("ghost")

    def test_list_personas(self):
        personas = PersonaLibrary.list_personas()
        assert len(personas) == 7

    def test_names(self):
        names = PersonaLibrary.names()
        assert "devops_engineer" in names
        assert "sre_oncall" in names


# ---------------------------------------------------------------------------
# BetaTestRunner simulate_discovery
# ---------------------------------------------------------------------------

class TestSimulateDiscovery:
    def test_perfect_repo(self):
        meta = {
            "name": "perfect",
            "readme": "deploy health with docker quickstart architecture scale build multi-node plugin register",
            "has_docker": True,
            "has_k8s": True,
            "has_runbook": True,
            "has_alerts": True,
            "has_examples": True,
            "has_sbom": True,
            "has_secrets": False,
            "signed_releases": True,
            "has_discovery": True,
            "has_register_api": True,
            "api_version": "v1",
            "has_ci": True,
            "has_deps_list": True,
        }
        result = BetaTestRunner.simulate_discovery(meta, persona_name="devops_engineer")
        assert result.rating == 5
        assert result.blockers == []
        assert result.is_passing()

    def test_bad_repo(self):
        meta = {"name": "bad", "readme": ""}
        result = BetaTestRunner.simulate_discovery(meta, persona_name="devops_engineer")
        assert result.rating == 1
        assert len(result.blockers) == 3
        assert not result.is_passing()

    def test_mixed_repo(self):
        meta = {"name": "ok", "readme": "deploy health", "has_docker": True}
        result = BetaTestRunner.simulate_discovery(meta, persona_name="devops_engineer")
        assert result.rating == 5  # all 3 checks pass
        assert result.blockers == []

    def test_junior_developer(self):
        meta = {"name": "x", "readme": "quickstart architecture", "has_examples": True}
        result = BetaTestRunner.simulate_discovery(meta, persona_name="junior_developer")
        assert result.rating == 5

    def test_security_auditor(self):
        meta = {"name": "x", "has_sbom": True, "has_secrets": False, "signed_releases": True}
        result = BetaTestRunner.simulate_discovery(meta, persona_name="security_auditor")
        assert result.rating == 5

    def test_time_to_first(self):
        meta = {"name": "x", "readme": "deploy"}
        result = BetaTestRunner.simulate_discovery(meta, persona_name="devops_engineer")
        assert result.time_to_first_success > 0

    def test_no_checks_persona(self):
        p = Persona(name="Ghost", role="unknown", expertise_level=1)
        meta = {"name": "x"}
        result = BetaTestRunner.simulate_discovery(meta, persona=p)
        # no checks defined for "ghost", no blockers = rating 5
        assert result.rating == 5
        assert result.blockers == []

    def test_requires_persona_or_name(self):
        with pytest.raises(ValueError):
            BetaTestRunner.simulate_discovery({})


# ---------------------------------------------------------------------------
# run_all_tests
# ---------------------------------------------------------------------------

class TestRunAllTests:
    def test_all_personas(self):
        meta = {"name": "perfect", "readme": "deploy", "has_docker": True}
        results = BetaTestRunner.run_all_tests(meta)
        assert len(results) == 7
        # DevOps should pass, others may vary
        assert any(r.persona.name == "DevOps Engineer" for r in results)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_markdown_output(self):
        meta = {"name": "perfect", "readme": "deploy", "has_docker": True}
        results = BetaTestRunner.run_all_tests(meta)
        report = BetaTestRunner.generate_report(results)
        assert "# Beta-Test Persona Report" in report
        assert "Average rating" in report
        assert "DevOps Engineer" in report
        assert "|" in report  # table

    def test_recommendations(self):
        meta = {"name": "bad", "readme": ""}
        results = BetaTestRunner.run_all_tests(meta)
        report = BetaTestRunner.generate_report(results)
        assert "Top Recommendations" in report
        assert "affects" in report

    def test_no_results_raises(self):
        with pytest.raises(IndexError):
            BetaTestRunner.generate_report([])
