"""Cross-Repo Duplicate Detection — SPEC-REPO-METRIC §8.

Scans a workspace of git repos and flags pairs with identical or
near-identical top-level files (README, manifest, first N source files).

This catches the duplicate-repo problem identified in
SPEC-FLUX-RESOLUTION:
  - flux-compiler/ appearing in multiple fleet-* repos
  - AI-Writings in 3 different casings
  - constraint-theory-py as both monorepo and individual crates
"""

from __future__ import annotations

__all__ = ["RepoDuplicateDetector", "find_repo_duplicates"]

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Files that define a repo's identity — hash these for comparison
IDENTITY_FILES = [
    "README.md",
    "README.rst",
    "README",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "poetry.lock",
    "Pipfile",
    "go.mod",
    "Makefile",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
]


def _hash_file(path: Path) -> str:
    """SHA-256 hash of file contents."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def _hash_repo(repo_path: Path, max_source_files: int = 10) -> Dict[str, str]:
    """Build a fingerprint dict for a repo.

    Keys are relative paths, values are truncated SHA-256 hashes.
    """
    hashes: Dict[str, str] = {}

    # 1. Identity files at root
    for name in IDENTITY_FILES:
        p = repo_path / name
        if p.exists():
            hashes[name] = _hash_file(p)

    # 2. First N source files (shallow, breadth-first)
    count = 0
    for child in sorted(repo_path.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            for f in sorted(child.iterdir()):
                if f.is_file() and f.suffix in (
                    ".py",
                    ".rs",
                    ".go",
                    ".js",
                    ".ts",
                    ".cpp",
                    ".c",
                    ".h",
                ):
                    rel = str(f.relative_to(repo_path))
                    hashes[rel] = _hash_file(f)
                    count += 1
                    if count >= max_source_files:
                        break
            if count >= max_source_files:
                break

    return hashes


def _repo_similarity(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Jaccard-like similarity of two repo fingerprint dicts.

    Returns 0.0–1.0 where 1.0 means identical files and hashes.
    """
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    if not keys_a or not keys_b:
        return 0.0

    shared = keys_a & keys_b
    if not shared:
        return 0.0

    identical = sum(1 for k in shared if a[k] == b[k])
    union = len(keys_a | keys_b)
    return identical / union if union else 0.0


@dataclass(frozen=True)
class RepoDuplicatePair:
    """A pair of repos flagged as potential duplicates."""

    repo_a: str
    repo_b: str
    similarity: float
    shared_files: List[str]
    identical_files: List[str]


class RepoDuplicateDetector:
    """Scan a workspace and find duplicate/near-duplicate repos."""

    def __init__(self, threshold: float = 0.85, max_source_files: int = 10) -> None:
        self.threshold = threshold
        self.max_source_files = max_source_files

    def scan(self, workspace: str | Path) -> List[RepoDuplicatePair]:
        """Scan all git repos in workspace, return flagged pairs.

        Args:
            workspace: Directory containing git repos.

        Returns:
            Sorted list of RepoDuplicatePair (highest similarity first).
        """
        root = Path(workspace)
        repos: Dict[str, Dict[str, str]] = {}

        for entry in os.scandir(root):
            if (Path(entry.path) / ".git").is_dir():
                name = Path(entry.path).name
                repos[name] = _hash_repo(Path(entry.path), self.max_source_files)

        names = sorted(repos.keys())
        pairs: List[RepoDuplicatePair] = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = _repo_similarity(repos[names[i]], repos[names[j]])
                if sim >= self.threshold:
                    shared = sorted(
                        set(repos[names[i]].keys()) & set(repos[names[j]].keys())
                    )
                    identical = sorted(
                        k for k in shared if repos[names[i]][k] == repos[names[j]][k]
                    )
                    pairs.append(
                        RepoDuplicatePair(
                            repo_a=names[i],
                            repo_b=names[j],
                            similarity=round(sim, 3),
                            shared_files=shared,
                            identical_files=identical,
                        )
                    )

        pairs.sort(key=lambda p: p.similarity, reverse=True)
        return pairs


def find_repo_duplicates(
    workspace: str | Path,
    threshold: float = 0.85,
    max_source_files: int = 10,
) -> List[RepoDuplicatePair]:
    """One-liner entrypoint."""
    return RepoDuplicateDetector(threshold, max_source_files).scan(workspace)
