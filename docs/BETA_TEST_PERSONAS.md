# Beta-Test Persona Framework

Simulated external visitors discover fleet repos on GitHub and rate the
onboarding experience. This codifies the behavioral-synthesis validation
pattern so any repo can be persona-tested automatically.

## Quick Start

```python
from fleet.beta_test_personas import BetaTestRunner, PersonaLibrary

# Describe your repo
repo = {
    "name": "sunset-ecosystem",
    "readme": "Quickstart: pip install -e . && pytest...",
    "has_docker": True,
    "has_examples": True,
    "has_ci": True,
    "has_deps_list": True,
    "has_runbook": False,
    "has_alerts": False,
    "has_sbom": False,
    "signed_releases": True,
}

# Test one persona
result = BetaTestRunner.simulate_discovery(repo, persona_name="junior_developer")
print(result.rating)  # 1–5
print(result.blockers)  # what stopped them

# Test all 7
results = BetaTestRunner.run_all_tests(repo)
report = BetaTestRunner.generate_report(results)
print(report)
```

## The Seven Personas

| Persona | Role | Expertise | What They Care About |
|---------|------|-----------|---------------------|
| **DevOps Engineer** | infrastructure | 4/5 | Docker, k8s, health checks, deploy in <30min |
| **SRE On-Call** | reliability | 5/5 | Runbooks, metrics, alert rules, failure modes |
| **Junior Developer** | learning | 2/5 | Quickstart, examples, architecture diagram |
| **Security Auditor** | security | 5/5 | SBOM, no secrets, signed releases |
| **Fleet Operator** | orchestration | 4/5 | Multi-node config, discovery, scaling |
| **Agent Developer** | extension | 3/5 | Plugin API, register function, stable versioning |
| **Infrastructure Engineer** | platform | 4/5 | CI config, dependency list, reproducible builds |

## Rating Scale

- **★★★★★ (5)** — Would adopt immediately, no friction
- **★★★★☆ (4)** — Would adopt with minor tweaks
- **★★★☆☆ (3)** — Would adopt if no alternative
- **★★☆☆☆ (2)** — Would look for alternatives first
- **★☆☆☆☆ (1)** — Would bounce immediately

## Checks Per Persona

Each persona has 3 weighted checks. The rating is derived from how many pass:

- 0/3 checks → 1 star
- 1/3 checks → 2 stars
- 2/3 checks → 3–4 stars (depending on which ones)
- 3/3 checks → 5 stars

## Report Output

`generate_report()` produces markdown with:
- Summary (average rating, passing count)
- Table of all personas with ratings, blockers, time-to-success
- Detailed notes per persona
- Top recommendations (most common blockers across personas)

## Integration with FleetConductorV2

The framework can be wired into `FleetConductorV2` as an SDA pipeline:

```python
from fleet.sense_decide_act import Sense, Decide, Act
from fleet.beta_test_personas import BetaTestRunner


class RepoDiscoverySense(Sense):
    def observe(self):
        # Scan repo metadata from git / README / files
        return {"repo_metadata": scan_repo(".")}


class PersonaTestDecide(Decide):
    def evaluate(self, observation):
        repo = observation.metrics["repo_metadata"]
        results = BetaTestRunner.run_all_tests(repo)
        avg = sum(r.rating for r in results) / len(results)
        return Decision(
            action="generate_report" if avg < 4 else "no_op",
            confidence=avg / 5,
        )
```

## Reference

- `fleet/beta_test_personas.py` — implementation
- `tests/test_beta_test_personas.py` — 18 tests
- `fleet/behavioral_synthesis.md` — pattern origin
