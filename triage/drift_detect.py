"""Drift Detection Rules — SPEC-REPO-METRIC §7.

Detects structural drift in a repo beyond simple freshness:
  - Stale dependencies (security advisories)
  - Test regression (tests that were passing now failing)
  - Documentation drift (README references missing files)
  - Dead code accumulation (unimported files)
  - Branch divergence (local vs remote)
"""
from __future__ import annotations

__all__ = [
    "DriftDetector",
    "DriftReport",
    "detect_drift",
]

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class DriftReport:
    """Structured drift findings for one repository."""

    repo: str
    stale_dependencies: List[str] = field(default_factory=list)
    test_regression: Optional[str] = None
    doc_drift: List[str] = field(default_factory=list)
    dead_code: List[str] = field(default_factory=list)
    branch_divergence: Optional[str] = None
    license_change: Optional[str] = None
    severity: str = "low"  # low / medium / high / critical

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "stale_dependencies": self.stale_dependencies,
            "test_regression": self.test_regression,
            "doc_drift": self.doc_drift,
            "dead_code": self.dead_code,
            "branch_divergence": self.branch_divergence,
            "license_change": self.license_change,
            "severity": self.severity,
        }


class DriftDetector:
    """Detect structural drift in a local git repository."""

    def __init__(self, repo_path: str | Path) -> None:
        self.root = Path(repo_path).resolve()
        self.repo_name = self.root.name

    # ── 1. Stale Dependencies ────────────────────────────────────

    def _stale_dependencies(self) -> List[str]:
        """Run dependency audit tools and return findings."""
        findings: List[str] = []

        # Python: pip-audit or safety
        if (self.root / "requirements.txt").exists() or (self.root / "pyproject.toml").exists():
            try:
                result = subprocess.run(
                    ["pip-audit", "--desc"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    # pip-audit returns non-zero when vulnerabilities found
                    findings.append(f"pip-audit findings:\n{result.stdout[:500]}")
            except FileNotFoundError:
                pass  # pip-audit not installed
            except subprocess.TimeoutExpired:
                findings.append("pip-audit timed out")

        # Rust: cargo audit
        if (self.root / "Cargo.toml").exists():
            try:
                result = subprocess.run(
                    ["cargo", "audit"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if "error" in result.stderr.lower() or result.returncode != 0:
                    findings.append(f"cargo audit findings:\n{result.stdout[:500]}")
            except FileNotFoundError:
                pass
            except subprocess.TimeoutExpired:
                findings.append("cargo audit timed out")

        return findings

    # ── 2. Test Regression ──────────────────────────────────────

    def _test_regression(self) -> Optional[str]:
        """Run tests and report if they fail."""
        if not (self.root / "tests").exists() and not list(self.root.glob("test_*.py")):
            return None  # no tests to run

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-q", "--tb=short"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                # Extract failure summary
                lines = result.stdout.splitlines()
                failures = [l for l in lines if "FAILED" in l or "ERROR" in l]
                summary = " | ".join(failures[:3]) if failures else "Tests failed"
                return summary
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return "pytest timed out"
        return None

    # ── 3. Documentation Drift ───────────────────────────────────

    def _doc_drift(self) -> List[str]:
        """Find README references to files that no longer exist."""
        findings: List[str] = []
        readme = self.root / "README.md"
        if not readme.exists():
            return findings

        content = readme.read_text()
        # Find markdown links and code references
        # Pattern: `filename.py` or [text](path) or plain filenames in code blocks
        code_refs = re.findall(r"`([^`]+\.(?:py|rs|go|js|ts|toml|json|yaml|yml))`", content)
        md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

        for ref in code_refs:
            if not (self.root / ref).exists() and not any(
                (self.root / d / ref).exists() for d in ("src", "lib", "app")
            ):
                findings.append(f"README references missing file: `{ref}`")

        for text, link in md_links:
            # Skip external URLs
            if link.startswith(("http://", "https://", "#")):
                continue
            target = self.root / link
            if not target.exists():
                findings.append(f"README links to missing path: [{text}]({link})")

        return findings

    # ── 4. Dead Code ─────────────────────────────────────────────

    def _dead_code(self) -> List[str]:
        """Find Python files that are never imported."""
        findings: List[str] = []
        py_files: Set[str] = set()
        imported: Set[str] = set()

        # Collect all .py files
        for f in self.root.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            rel = str(f.relative_to(self.root))
            py_files.add(rel)

        # Scan imports
        for f in self.root.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name.split(".")[0]
                        imported.add(mod + ".py")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mod = node.module.split(".")[0]
                        imported.add(mod + ".py")

        # Find unimported modules (excluding __init__.py and test files)
        for rel in py_files:
            if rel in ("__init__.py", "setup.py") or rel.startswith("tests/"):
                continue
            base = Path(rel).name
            if base not in imported and rel not in imported:
                findings.append(f"Potentially unimported: {rel}")

        return findings[:20]  # cap to avoid noise

    # ── 5. Branch Divergence ─────────────────────────────────────

    def _branch_divergence(self) -> Optional[str]:
        """Check if local branch is far ahead/behind remote."""
        try:
            result = subprocess.run(
                ["git", "rev-list", "--left-right", "--count", "HEAD...@{u}"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None  # no upstream configured
            ahead, behind = map(int, result.stdout.strip().split("\t"))
            if ahead > 50:
                return f"Local branch is {ahead} commits ahead of remote"
            if behind > 50:
                return f"Local branch is {behind} commits behind remote"
            return None
        except Exception:
            return None

    # ── 6. License Change (placeholder) ──────────────────────────

    def _license_change(self) -> Optional[str]:
        """Detect if LICENSE file changed recently.

        Currently a placeholder — full implementation would need
        a license classification tool or manual review.
        """
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ct", "--", "LICENSE*"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                import time
                ts = int(result.stdout.strip())
                days = (time.time() - ts) / 86400
                if days < 30:
                    return f"LICENSE modified {days:.0f} days ago (review for changes)"
            return None
        except Exception:
            return None

    # ── Orchestrator ──────────────────────────────────────────────

    def run(self) -> DriftReport:
        """Run all drift detectors and return a composite report."""
        stale = self._stale_dependencies()
        regression = self._test_regression()
        doc = self._doc_drift()
        dead = self._dead_code()
        branch = self._branch_divergence()
        license_ = self._license_change()

        # Severity scoring
        severity = "low"
        if regression:
            severity = "critical"
        elif stale or license_:
            severity = "high"
        elif doc or branch:
            severity = "medium"

        return DriftReport(
            repo=self.repo_name,
            stale_dependencies=stale,
            test_regression=regression,
            doc_drift=doc,
            dead_code=dead,
            branch_divergence=branch,
            license_change=license_,
            severity=severity,
        )


def detect_drift(repo_path: str | Path) -> DriftReport:
    """One-liner entrypoint."""
    return DriftDetector(repo_path).run()
