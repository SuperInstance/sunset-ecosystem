"""Triage — Weekly repo health and issue hygiene automation.

Per SPEC-REPO-METRIC:
  - metrics:       five-component health score
  - github_issues:  GitHub REST API wrapper
  - duplicate_detect: TF-IDF duplicate issue detection
  - weekly:         orchestration runner
"""
from __future__ import annotations

from triage.metrics import RepoHealthMetrics, HealthScore, run_health_check
from triage.github_issues import GitHubIssues, IssueState
from triage.duplicate_detect import DuplicateDetector, find_duplicates, DuplicatePair
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
    "find_duplicates",
    "DuplicatePair",
    # weekly
    "WeeklyTriage",
    "TriageReport",
    "run_triage",
]
