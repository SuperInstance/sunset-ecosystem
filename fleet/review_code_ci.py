#!/usr/bin/env python3
"""CI entry point for fleet code review personas.

Runs 5 AST-based personas on changed files in a PR and outputs structured JSON
for GitHub Actions comment posting.
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
import glob
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from fleet.review_code import ReviewPersona, ReviewFindings, review_code


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def severity_rank(s: str) -> int:
    return SEVERITY_ORDER.get(s, 0)


# ---------------------------------------------------------------------------
# Git diff helper
# ---------------------------------------------------------------------------


def get_changed_files(base_ref: str = "HEAD~1") -> List[str]:
    """Return list of changed Python files in the PR."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref, "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    return [f for f in files if f.endswith(".py")]


def get_pr_files_from_env() -> List[str]:
    """Read changed files from GitHub event payload."""
    github_event_path = Path(os.environ.get("GITHUB_EVENT_PATH", "/dev/null"))
    if not github_event_path.exists():
        return []
    try:
        data = json.loads(github_event_path.read_text())
        # Extract changed files from pull_request payload
        files = []
        for f in data.get("pull_request", {}).get("files", []):
            filename = f.get("filename", "")
            if filename.endswith(".py"):
                files.append(filename)
        return files
    except Exception:
        return []


# ---------------------------------------------------------------------------
# CI review runner
# ---------------------------------------------------------------------------


def run_ci_review(
    files: List[str],
    fail_on_severity: str = "critical",
) -> Dict[str, Any]:
    """Run all personas on changed files and return structured output."""
    personas = ReviewPersona.all()
    all_findings: List[Dict[str, Any]] = []
    file_summaries: Dict[str, Dict[str, Any]] = {}

    max_fail_rank = severity_rank(fail_on_severity)

    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            continue
        code = path.read_text()
        file_findings: List[Dict[str, Any]] = []
        for persona in personas:
            findings = review_code(code, persona)
            for finding in findings:
                finding_dict = {
                    "persona": persona.name,
                    "severity": finding.severity,
                    "message": finding.message,
                    "line": finding.line,
                }
                file_findings.append(finding_dict)
                all_findings.append(finding_dict)
        file_summaries[filepath] = {
            "findings": file_findings,
            "count": len(file_findings),
            "critical": sum(1 for f in file_findings if f["severity"] == "critical"),
            "warning": sum(1 for f in file_findings if f["severity"] == "warning"),
            "info": sum(1 for f in file_findings if f["severity"] == "info"),
        }

    critical_count = sum(1 for f in all_findings if f["severity"] == "critical")
    warning_count = sum(1 for f in all_findings if f["severity"] == "warning")
    info_count = sum(1 for f in all_findings if f["severity"] == "info")
    should_fail = any(
        severity_rank(f["severity"]) >= max_fail_rank for f in all_findings
    )

    return {
        "files": file_summaries,
        "summary": {
            "total_files": len(files),
            "total_findings": len(all_findings),
            "critical": critical_count,
            "warning": warning_count,
            "info": info_count,
            "should_fail": should_fail,
        },
        "findings": all_findings,
    }


def format_markdown_comment(result: Dict[str, Any]) -> str:
    """Format CI review result as GitHub markdown comment."""
    summary = result["summary"]
    lines = [
        "## 🦀 Fleet Code Review — Persona Report",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Files reviewed | {summary['total_files']} |",
        f"| Total findings | {summary['total_findings']} |",
        f"| 🔴 Critical | {summary['critical']} |",
        f"| 🟡 Warning | {summary['warning']} |",
        f"| 🔵 Info | {summary['info']} |",
        "",
    ]
    if summary["should_fail"]:
        lines.append("⚠️ **Critical findings detected. CI will fail.**")
        lines.append("")
    for filepath, info in result["files"].items():
        if info["count"] == 0:
            continue
        lines.append(f"### `{filepath}`")
        lines.append(
            f"{info['critical']} critical, {info['warning']} warning, {info['info']} info"
        )
        lines.append("")
        for f in info["findings"]:
            emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                f["severity"], "⚪"
            )
            lines.append(
                f"- {emoji} **{f['persona']}** (line {f['line']}): {f['message']}"
            )
        lines.append("")
    return "\n".join(lines)


def format_json_output(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fleet Code Review CI Runner")
    parser.add_argument(
        "--pr-files", action="store_true", help="Read changed files from git diff"
    )
    parser.add_argument("--files", nargs="+", help="Explicit file list to review")
    parser.add_argument(
        "--output", choices=["json", "markdown"], default="json", help="Output format"
    )
    parser.add_argument(
        "--fail-on-severity",
        default="critical",
        choices=["info", "warning", "critical"],
        help="Severity threshold to fail CI",
    )
    parser.add_argument(
        "--comment-file", help="Write markdown comment to file for GitHub Actions"
    )
    args = parser.parse_args()

    files: List[str] = []
    if args.files:
        files = args.files
    elif args.pr_files:
        # Try GitHub event first, then git diff
        files = get_pr_files_from_env()
        if not files:
            files = get_changed_files()
    else:
        # Default: all Python files in repo
        files = [str(p) for p in Path(".").rglob("*.py") if ".git" not in str(p)]

    if not files:
        print(
            json.dumps(
                {
                    "summary": {
                        "total_files": 0,
                        "total_findings": 0,
                        "should_fail": False,
                    }
                }
            )
        )
        sys.exit(0)

    result = run_ci_review(files, fail_on_severity=args.fail_on_severity)

    if args.output == "json":
        print(format_json_output(result))
    elif args.output == "markdown":
        print(format_markdown_comment(result))

    if args.comment_file:
        Path(args.comment_file).write_text(format_markdown_comment(result))

    if result["summary"]["should_fail"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
