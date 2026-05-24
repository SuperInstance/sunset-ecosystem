"""GitHub Issues API wrapper for triage.

Lightweight wrapper around GitHub REST API v3 for issue
fetching, labeling, and lifecycle tracking.
"""
from __future__ import annotations

__all__ = ["GitHubIssues", "IssueState"]

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import requests


class IssueState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


@dataclass
class Issue:
    number: int
    title: str
    state: str
    labels: List[str]
    body: str
    created_at: str
    updated_at: str
    closed_at: Optional[str] = None
    assignees: List[str] = None
    milestone: Optional[str] = None

    def __post_init__(self) -> None:
        if self.assignees is None:
            self.assignees = []


class GitHubIssues:
    """Fetch and manipulate GitHub issues for a repository."""

    API_BASE = "https://api.github.com"

    def __init__(
        self,
        owner: str,
        repo: str,
        token: Optional[str] = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub token required. Pass token= or set GITHUB_TOKEN env var."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        url = f"{self.API_BASE}/repos/{self.owner}/{self.repo}/{endpoint}"
        resp = self._session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, json_body: dict) -> Any:
        url = f"{self.API_BASE}/repos/{self.owner}/{self.repo}/{endpoint}"
        resp = self._session.post(url, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _patch(self, endpoint: str, json_body: dict) -> Any:
        url = f"{self.API_BASE}/repos/{self.owner}/{self.repo}/{endpoint}"
        resp = self._session.patch(url, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_issues(
        self,
        state: IssueState = IssueState.OPEN,
        labels: Optional[List[str]] = None,
        since: Optional[str] = None,
        per_page: int = 100,
    ) -> List[Issue]:
        """Fetch issues from the repository."""
        params: Dict[str, Any] = {
            "state": state.value,
            "per_page": per_page,
        }
        if labels:
            params["labels"] = ",".join(labels)
        if since:
            params["since"] = since

        data = self._get("issues", params)
        issues = []
        for item in data:
            # Pull requests masquerade as issues in the API — skip them
            if "pull_request" in item:
                continue
            issues.append(
                Issue(
                    number=item["number"],
                    title=item["title"],
                    state=item["state"],
                    labels=[lbl["name"] for lbl in item.get("labels", [])],
                    body=item.get("body", "") or "",
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                    closed_at=item.get("closed_at"),
                    assignees=[a["login"] for a in item.get("assignees", [])],
                    milestone=item.get("milestone", {}).get("title")
                    if item.get("milestone")
                    else None,
                )
            )
        return issues

    def add_labels(self, issue_number: int, labels: List[str]) -> None:
        """Add labels to an existing issue."""
        self._post(f"issues/{issue_number}/labels", {"labels": labels})

    def update_issue(
        self,
        issue_number: int,
        state: Optional[str] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
    ) -> None:
        """Update issue state, title, or body."""
        payload: Dict[str, Any] = {}
        if state is not None:
            payload["state"] = state
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if payload:
            self._patch(f"issues/{issue_number}", payload)

    def hygiene_score(self) -> float:
        """Compute issue hygiene score (0-15) for the repo.

        Metrics:
          - stale issues (no update > 60 days) penalize
          - unlabeled issues penalize
          - high close-rate rewards
        """
        open_issues = self.list_issues(IssueState.OPEN)
        closed_issues = self.list_issues(IssueState.CLOSED, per_page=100)

        if not open_issues and not closed_issues:
            return 7.5  # neutral for empty repo

        total = len(open_issues) + len(closed_issues)

        # 1. Stale penalty (max -5 pts)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        stale = 0
        for issue in open_issues:
            updated = datetime.fromisoformat(issue.updated_at.replace("Z", "+00:00"))
            days = (now - updated).days
            if days > 60:
                stale += 1
        stale_ratio = stale / max(len(open_issues), 1)
        stale_penalty = 5.0 * stale_ratio

        # 2. Unlabeled penalty (max -5 pts)
        unlabeled = sum(1 for i in open_issues if not i.labels)
        unlabeled_ratio = unlabeled / max(len(open_issues), 1)
        label_penalty = 5.0 * unlabeled_ratio

        # 3. Close-rate reward (max 15 pts)
        close_rate = len(closed_issues) / total
        close_reward = 15.0 * close_rate

        score = max(0.0, close_reward - stale_penalty - label_penalty)
        return round(score, 1)
