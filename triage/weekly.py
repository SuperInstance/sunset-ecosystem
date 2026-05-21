"""Weekly Triage Runner — SPEC-REPO-METRIC §4 implementation.

Orchestrates the full weekly triage workflow:
  1. Compute repo health score
  2. Fetch + analyze GitHub issues
  3. Detect issue duplicates
  4. Detect cross-repo duplicates (SPEC-REPO-METRIC §8)
  5. Run drift detection (SPEC-REPO-METRIC §7)
  6. Generate triage report + auto-label stale issues

Intended to be invoked by cron or CI weekly.
"""
from __future__ import annotations

__all__ = ["WeeklyTriage", "run_triage"]

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from triage.drift_detect import DriftDetector, DriftReport, detect_drift
from triage.duplicate_detect import DuplicateDetector, DuplicatePair, find_duplicates
from triage.github_issues import GitHubIssues, IssueState
from triage.metrics import RepoHealthMetrics, HealthScore, run_health_check
from triage.repo_duplicate import RepoDuplicateDetector, RepoDuplicatePair, find_repo_duplicates

logger = logging.getLogger(__name__)


@dataclass
class TriageReport:
    """Structured output of one triage run."""

    repo: str
    run_at: str
    health: HealthScore
    open_issue_count: int
    duplicate_pairs: List[dict] = field(default_factory=list)
    stale_issues: List[int] = field(default_factory=list)
    unlabeled_issues: List[int] = field(default_factory=list)
    drift: Optional[DriftReport] = None
    repo_duplicates: List[dict] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "run_at": self.run_at,
            "health": self.health.to_dict(),
            "open_issue_count": self.open_issue_count,
            "duplicate_pairs": self.duplicate_pairs,
            "stale_issues": self.stale_issues,
            "unlabeled_issues": self.unlabeled_issues,
            "drift": self.drift.to_dict() if self.drift else None,
            "repo_duplicates": self.repo_duplicates,
            "actions_taken": self.actions_taken,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Triage Report: {self.repo}",
            f"**Run at:** {self.run_at}",
            "",
            "## Health Score",
            f"- **Total:** {self.health.total} ({self.health.traffic_light})",
            f"- Freshness: {self.health.freshness:.1f} / 30",
            f"- Test Coverage: {self.health.test_coverage:.1f} / 25",
            f"- Documentation: {self.health.documentation:.1f} / 15",
            f"- Dependency Health: {self.health.dependency_health:.1f} / 15",
            f"- Issue Hygiene: {self.health.issue_hygiene:.1f} / 15",
            "",
            f"## Issues",
            f"- Open: {self.open_issue_count}",
        ]
        if self.stale_issues:
            lines.append(f"- Stale (>60d): {', '.join(map(str, self.stale_issues))}")
        if self.unlabeled_issues:
            lines.append(f"- Unlabeled: {', '.join(map(str, self.unlabeled_issues))}")
        if self.duplicate_pairs:
            lines.append("- Potential Duplicates:")
            for pair in self.duplicate_pairs:
                lines.append(
                    f"  - #{pair['issue_a']} ≈ #{pair['issue_b']} "
                    f"(sim={pair['similarity']}, shared={pair['shared_terms']})"
                )
        if self.repo_duplicates:
            lines.append("- Cross-Repo Duplicates:")
            for pair in self.repo_duplicates:
                lines.append(
                    f"  - {pair['repo_a']} ≈ {pair['repo_b']} "
                    f"(sim={pair['similarity']}, identical={len(pair.get('identical_files', []))})"
                )
        if self.drift:
            lines.extend(["", "## Drift Detection"])
            if self.drift.stale_dependencies:
                lines.append("- **Stale Dependencies:**")
                for d in self.drift.stale_dependencies:
                    lines.append(f"  - {d}")
            if self.drift.test_regression:
                lines.append(f"- **Test Regression:** {self.drift.test_regression}")
            if self.drift.doc_drift:
                lines.append("- **Documentation Drift:**")
                for d in self.drift.doc_drift:
                    lines.append(f"  - {d}")
            if self.drift.dead_code:
                lines.append("- **Dead Code:**")
                for d in self.drift.dead_code:
                    lines.append(f"  - {d}")
            if self.drift.branch_divergence:
                lines.append(f"- **Branch Divergence:** {self.drift.branch_divergence}")
            lines.append(f"- **Severity:** {self.drift.severity}")
        if self.actions_taken:
            lines.extend(["", "## Actions Taken"] + [f"- {a}" for a in self.actions_taken])
        return "\n".join(lines)


class WeeklyTriage:
    """Weekly triage orchestrator."""

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        repo_root: str,
        workspace: Optional[str] = None,
        github_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        auto_label: bool = False,
    ) -> None:
        self.owner = repo_owner
        self.name = repo_name
        self.root = Path(repo_root)
        self.workspace = Path(workspace) if workspace else self.root.parent
        self.gh = GitHubIssues(
            owner=repo_owner,
            repo=repo_name,
            token=github_token or os.environ.get("GITHUB_TOKEN"),
        )
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / ".triage_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_label = auto_label

    def run(self) -> TriageReport:
        """Execute full weekly triage workflow."""
        now = datetime.now(timezone.utc).isoformat()
        actions: List[str] = []

        # 1. Health score
        health = RepoHealthMetrics(self.root).run()
        logger.info("Health score: %.1f (%s)", health.total, health.traffic_light)

        # 2. Fetch issues
        open_issues = self.gh.list_issues(IssueState.OPEN)
        logger.info("Open issues: %d", len(open_issues))

        # 3. Stale detection (>60 days no update)
        stale: List[int] = []
        unlabeled: List[int] = []
        for issue in open_issues:
            updated = datetime.fromisoformat(issue.updated_at.replace("Z", "+00:00"))
            days = (datetime.now(timezone.utc) - updated).days
            if days > 60:
                stale.append(issue.number)
                if self.auto_label:
                    try:
                        self.gh.add_labels(issue.number, ["stale"])
                        actions.append(f"Labeled #{issue.number} stale")
                    except Exception as e:
                        logger.warning("Failed to label #%d: %s", issue.number, e)
            if not issue.labels:
                unlabeled.append(issue.number)

        # 4. Issue duplicate detection
        issue_dicts = [
            {"number": i.number, "title": i.title, "body": i.body}
            for i in open_issues
        ]
        dupes = find_duplicates(issue_dicts)
        duplicate_pairs = [dict(d) for d in dupes]
        if dupes and self.auto_label:
            for pair in dupes:
                try:
                    self.gh.add_labels(pair.issue_a, ["possible-duplicate"])
                    self.gh.add_labels(pair.issue_b, ["possible-duplicate"])
                    actions.append(
                        f"Labeled #{pair.issue_a} / #{pair.issue_b} possible-duplicate"
                    )
                except Exception as e:
                    logger.warning("Failed to label duplicates: %s", e)

        # 5. Drift detection
        drift = None
        try:
            drift = detect_drift(self.root)
            logger.info("Drift severity: %s", drift.severity)
        except Exception as e:
            logger.warning("Drift detection failed: %s", e)

        # 6. Cross-repo duplicate detection
        repo_dupes = []
        try:
            if self.workspace.exists():
                repo_dupe_pairs = find_repo_duplicates(self.workspace)
                repo_dupes = [dict(d) for d in repo_dupe_pairs]
                logger.info("Cross-repo duplicates: %d", len(repo_dupes))
        except Exception as e:
            logger.warning("Repo duplicate detection failed: %s", e)

        # 7. Cache hygiene score for next run
        issue_hygiene_path = self.cache_dir / "issue_hygiene.json"
        try:
            hygiene = self.gh.hygiene_score()
            issue_hygiene_path.write_text(
                json.dumps({"score": hygiene, "updated": now})
            )
            logger.info("Issue hygiene cached: %.1f", hygiene)
        except Exception as e:
            logger.warning("Could not compute hygiene score: %s", e)

        report = TriageReport(
            repo=f"{self.owner}/{self.name}",
            run_at=now,
            health=health,
            open_issue_count=len(open_issues),
            duplicate_pairs=duplicate_pairs,
            stale_issues=stale,
            unlabeled_issues=unlabeled,
            drift=drift,
            repo_duplicates=repo_dupes,
            actions_taken=actions,
        )

        # 8. Persist report
        report_path = self.cache_dir / f"triage-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
        report_path.write_text(report.to_json())
        logger.info("Report saved: %s", report_path)

        return report


def run_triage(
    owner: str,
    repo: str,
    repo_root: str,
    workspace: Optional[str] = None,
    token: Optional[str] = None,
    auto_label: bool = False,
) -> TriageReport:
    """One-liner entrypoint for CLI / cron invocation."""
    return WeeklyTriage(
        repo_owner=owner,
        repo_name=repo,
        repo_root=repo_root,
        workspace=workspace,
        github_token=token,
        auto_label=auto_label,
    ).run()
