"""Triage — Weekly repo health and issue hygiene automation.

Per SPEC-REPO-METRIC:
  - metrics:          five-component health score
  - github_issues:    GitHub REST API wrapper
  - duplicate_detect: TF-IDF duplicate issue detection
  - repo_duplicate:   Cross-repo duplicate detection via file hashing
  - drift_detect:   Structural drift detection (deps, tests, docs, dead code)
  - weekly:          orchestration runner
"""

from __future__ import annotations

from triage.drift_detect import DriftDetector, DriftReport, detect_drift
from triage.duplicate_detect import DuplicateDetector, DuplicatePair, find_duplicates
from triage.github_issues import GitHubIssues, IssueState
from triage.metrics import RepoHealthMetrics, HealthScore, run_health_check
from triage.repo_duplicate import (
    RepoDuplicateDetector,
    RepoDuplicatePair,
    find_repo_duplicates,
)
from triage.weekly import WeeklyTriage, TriageReport, run_triage

__all__ = [
    # metrics
    "RepoHealthMetrics",
    "HealthScore",
    "run_health_check",
    # github_issues
    "GitHubIssues",
    "IssueState",
    # duplicate_detect
    "DuplicateDetector",
    "DuplicatePair",
    "find_duplicates",
    # repo_duplicate
    "RepoDuplicateDetector",
    "RepoDuplicatePair",
    "find_repo_duplicates",
    # drift_detect
    "DriftDetector",
    "DriftReport",
    "detect_drift",
    # weekly
    "WeeklyTriage",
    "TriageReport",
    "run_triage",
]
