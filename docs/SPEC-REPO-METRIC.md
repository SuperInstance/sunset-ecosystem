# SPEC-REPO-METRIC.md
**Author:** CCC (Systems Architect)  
**Date:** 2026-05-21  
**Status:** ARCHITECTURE — Weekly triage automation for drifting repos

---

## 1. Problem

The SuperInstance ecosystem has 130+ repos, ~30% of which are stale (8+ days without commits). The STRUCTURAL-SURVEY identified this as the "Dead Zone" — 40 repos that haven't been touched. Without automated triage, repos silently drift into obsolescence, dependencies rot, and duplicated effort goes undetected.

## 2. Metric Definition

### Repo Health Score

Each repo gets a composite score from 0-100:

| Component | Weight | Measurement |
|-----------|--------|-------------|
| **Freshness** | 30 | Days since last commit. 0 days = 30, 7 days = 15, 30+ days = 0 |
| **Test Coverage** | 25 | Does it have tests? Do they pass? Binary + ratio |
| **Documentation** | 15 | Has README? Has CHANGELOG? Has API docs? |
| **Dependency Health** | 15 | Are deps up to date? Any security advisories? |
| **Issue Hygiene** | 15 | Open issues < 30 days old? Stale issues closed? |

### Score Thresholds

| Score | Status | Action |
|-------|--------|--------|
| 80-100 | **Healthy** | No action |
| 60-79 | **Needs Attention** | Add to weekly review |
| 40-59 | **Drifting** | Create GitHub issue, ping maintainer |
| 0-39 | **Dead** | Archive candidate |

## 3. Weekly Triage Automation

### The Cron Job

A weekly cron task (Sundays 00:00 UTC) that:

1. Scans all repos in the workspace
2. Computes health scores
3. Creates/updates GitHub issues for drifting repos
4. Generates a triage report

```bash
# .openclaw/cron/weekly-triage.sh
#!/bin/bash
# Run weekly repo health triage
cd /home/phoenix/.openclaw/workspace/sunset-ecosystem
python -m triage.weekly --report --create-issues --threshold 60
```

### The Python Module

```python
# sunset-ecosystem/triage/weekly.py

"""
Weekly repo triage automation.

Usage:
    python -m triage.weekly --report          # Print report
    python -m triage.weekly --create-issues   # Create GitHub issues for drifting repos
    python -m triage.weekly --archive-dead    # Archive repos below threshold
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- These would be real implementations ---


@dataclass
class RepoMetrics:
    """Health metrics for one repository."""

    path: str
    name: str
    last_commit_days: int
    has_tests: bool
    test_pass_rate: float  # 0.0 - 1.0
    has_readme: bool
    has_changelog: bool
    open_issues: int
    stale_issues: int  # > 30 days
    health_score: float

    @property
    def status(self) -> str:
        if self.health_score >= 80:
            return "HEALTHY"
        if self.health_score >= 60:
            return "NEEDS_ATTENTION"
        if self.health_score >= 40:
            return "DRIFTING"
        return "DEAD"


def compute_freshness(days: int) -> float:
    """0 days → 30 points, 30+ days → 0 points."""
    return max(0, 30 - days)


def compute_test_score(has_tests: bool, pass_rate: float) -> float:
    """Has tests + they pass → 25 points."""
    if not has_tests:
        return 0
    return 25 * pass_rate


def compute_doc_score(has_readme: bool, has_changelog: bool) -> float:
    """README = 10, CHANGELOG = 5."""
    return (10 if has_readme else 0) + (5 if has_changelog else 0)


def compute_issue_score(total: int, stale: int) -> float:
    """All issues fresh → 15, all stale → 0."""
    if total == 0:
        return 10  # no issues is fine
    fresh_ratio = 1.0 - (stale / max(total, 1))
    return 15 * fresh_ratio


def scan_repo(path: str) -> RepoMetrics:
    """Scan one repo and compute health metrics."""
    repo_path = Path(path)
    name = repo_path.name

    # Last commit age
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            capture_output=True,
            text=True,
            cwd=path,
        )
        ts = int(result.stdout.strip())
        days = (datetime.now(tz=timezone.utc).timestamp() - ts) / 86400
    except (ValueError, subprocess.CalledProcessError):
        days = 999

    # Has tests?
    has_tests = bool(
        list(repo_path.glob("test*"))
        or list(repo_path.glob("**/test_*.py"))
        or list(repo_path.glob("**/*_test.go"))
        or list(repo_path.rglob("tests"))
    )

    # Has docs?
    has_readme = (repo_path / "README.md").exists()
    has_changelog = (repo_path / "CHANGELOG.md").exists()

    # Compute score
    freshness = compute_freshness(days)
    test_score = compute_test_score(has_tests, 0.8 if has_tests else 0)  # assume pass
    doc_score = compute_doc_score(has_readme, has_changelog)
    # Dependency + issue scores need GitHub API — placeholder
    dep_score = 10  # neutral
    issue_score = 10  # neutral

    total = freshness + test_score + doc_score + dep_score + issue_score

    return RepoMetrics(
        path=path,
        name=name,
        last_commit_days=int(days),
        has_tests=has_tests,
        test_pass_rate=0.8 if has_tests else 0,
        has_readme=has_readme,
        has_changelog=has_changelog,
        open_issues=0,
        stale_issues=0,
        health_score=total,
    )


def triage_report(workspace: str) -> list[RepoMetrics]:
    """Scan all git repos in workspace, return metrics sorted by health."""
    repos = []
    for entry in os.scandir(workspace):
        if (Path(entry.path) / ".git").is_dir():
            try:
                repos.append(scan_repo(entry.path))
            except Exception:
                pass

    repos.sort(key=lambda r: r.health_score)
    return repos


def create_github_issue(repo: RepoMetrics) -> Optional[str]:
    """Create a GitHub issue for a drifting repo.

    Uses `gh issue create` from the GitHub CLI.
    Returns the issue URL or None on failure.
    """
    if repo.health_score >= 60:
        return None

    title = f"[TRIAGE] {repo.name} health score: {repo.health_score:.0f}/100"
    body = f"""## Weekly Triage Report

**Repo:** `{repo.name}`
**Health Score:** {repo.health_score:.0f}/100
**Status:** {repo.status}
**Last Commit:** {repo.last_commit_days} days ago

### Metrics
| Component | Score |
|-----------|-------|
| Freshness | {compute_freshness(repo.last_commit_days):.0f}/30 |
| Tests | {"✅" if repo.has_tests else "❌"} |
| README | {"✅" if repo.has_readme else "❌"} |
| CHANGELOG | {"✅" if repo.has_changelog else "❌"} |

### Recommended Action
{"Archive this repository" if repo.health_score < 40 else "Review and update this repository"}

---
*Auto-generated by weekly triage on {datetime.now().strftime("%Y-%m-%d")}*
"""

    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                f"SuperInstance/{repo.name}",
                "--title",
                title,
                "--body",
                body,
                "--label",
                "triage",
            ],
            capture_output=True,
            text=True,
            cwd=repo.path,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None
```

## 4. GitHub Issue Template

For repos that score below 60, the triage creates an issue using this template:

```yaml
# .github/ISSUE_TEMPLATE/triage.yml
name: Weekly Triage
description: Auto-generated repo health check
labels: [triage]
body:
  - type: input
    id: health_score
    attributes:
      label: Health Score
      description: "Score out of 100"
  - type: textarea
    id: action_items
    attributes:
      label: Action Items
      description: What needs to happen
```

## 5. The Triage Report

The weekly report is saved to `docs/triage/YYYY-MM-DD.md`:

```markdown
# Weekly Triage Report — {DATE}

## Summary
- **Healthy (80+):** {N} repos
- **Needs Attention (60-79):** {N} repos
- **Drifting (40-59):** {N} repos  
- **Dead (< 40):** {N} repos

## Archive Candidates
| Repo | Score | Last Commit | Action |
|------|-------|-------------|--------|
| ... | 15 | 82 days | Archive |

## Needs Attention
| Repo | Score | Issue | 
|------|-------|-------|
| ... | 62 | #123 |

## Trending Up
| Repo | Score | Change from Last Week |
|------|-------|----------------------|
| ... | 88 | +12 |
```

## 6. Automation via OpenClaw Cron

```yaml
# Register in OpenClaw cron
schedule: "0 0 * * 0"  # Every Sunday midnight
task: |
  cd /home/phoenix/.openclaw/workspace/sunset-ecosystem
  python -m triage.weekly --report --create-issues --threshold 60
  # Report auto-delivered to main session
```

## 7. Drift Detection Rules

Beyond simple freshness, the triage should detect:

| Drift Type | Detection | Severity |
|------------|-----------|----------|
| **Stale dependencies** | `cargo audit` / `npm audit` / `pip audit` output | High |
| **Test regression** | Tests that were passing now failing | Critical |
| **Documentation drift** | README references files/deps that no longer exist | Medium |
| **License change** | Dependency license changed from permissive to restrictive | High |
| **Dead code accumulation** | Files not imported/referenced by any other file | Low |
| **Branch divergence** | Local branch > 50 commits ahead/behind remote | Medium |

## 8. Integration with SPEC-FLUX-RESOLUTION

The triage should also detect the **duplicate repo problem** identified in SPEC-FLUX-RESOLUTION:

- Multiple fleet-* repos containing identical flux-compiler/flux-vm subdirs
- AI-Writings in 3 different casings
- constraint-theory-py as both monorepo and individual crates

Detection: hash the top-level files of each repo. If two repos have identical README + Cargo.toml/package.json/pyproject.toml, flag as duplicate.

```python
def detect_duplicates(workspace: str) -> list[tuple[str, str, float]]:
    """Find repos with identical top-level file hashes.
    Returns (repo_a, repo_b, similarity_score)."""
    # Implementation: hash README, manifest, and first 10 source files
    # Compare all pairs
    # Flag if similarity > 0.9
    ...
```

## 9. File Structure

```
sunset-ecosystem/
├── triage/
│   ├── __init__.py
│   ├── weekly.py          ← Main triage script
│   ├── metrics.py         ← Health score computation
│   ├── github_issues.py   ← Issue creation via `gh`
│   └── duplicate_detect.py ← Find duplicate repos
├── docs/
│   └── triage/
│       ├── 2026-05-18.md  ← Historical reports
│       └── 2026-05-25.md  ← Next report
└── .github/
    └── ISSUE_TEMPLATE/
        └── triage.yml
```
