"""Survey the codebase: structure, patterns, debt, and recent changes."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


__all__ = ["CodebaseState", "survey_codebase"]

# Language extensions we recognise
_LANG_MAP: Dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".rs": "Rust",
    ".go": "Go",
    ".rb": "Ruby",
    ".java": "Java",
    ".kt": "Kotlin",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".swift": "Swift",
    ".sh": "Shell",
    ".bash": "Shell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".md": "Markdown",
}

_DEBT_PATTERNS = {
    "TODO": re.compile(r"\bTODO\b", re.IGNORECASE),
    "FIXME": re.compile(r"\bFIXME\b", re.IGNORECASE),
    "HACK": re.compile(r"\bHACK\b", re.IGNORECASE),
    "XXX": re.compile(r"\bXXX\b"),
    "DEPRECATED": re.compile(r"\bDEPRECATED\b", re.IGNORECASE),
}


@dataclass
class CodebaseState:
    """A snapshot of a codebase at a point in time."""

    root: str
    file_count: int = 0
    total_lines: int = 0
    language_breakdown: Dict[str, int] = field(default_factory=dict)
    language_lines: Dict[str, int] = field(default_factory=dict)
    test_count: int = 0
    test_collected: List[str] = field(default_factory=list)
    architecture_patterns: Dict[str, List[str]] = field(default_factory=dict)
    technical_debt: Dict[str, List[str]] = field(default_factory=dict)
    recent_commits: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"CodebaseState(root={self.root!r}, files={self.file_count}, "
            f"lines={self.total_lines}, languages={len(self.language_breakdown)})"
        )


def _run(cmd: List[str], cwd: str, timeout: int = 30) -> Optional[str]:
    """Run a subprocess, return stdout or None on failure."""
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _count_files_and_lines(root: Path) -> tuple:
    """Walk the tree counting files and lines by language."""
    file_count = 0
    total_lines = 0
    lang_files: Dict[str, int] = {}
    lang_lines: Dict[str, int] = {}

    # dirs to skip
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".tox", "dist", "build", ".egg-info"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            lang = _LANG_MAP.get(ext)
            file_count += 1
            if lang:
                lang_files[lang] = lang_files.get(lang, 0) + 1
            try:
                with open(fpath, errors="ignore") as fh:
                    line_count = sum(1 for _ in fh)
            except OSError:
                continue
            total_lines += line_count
            if lang:
                lang_lines[lang] = lang_lines.get(lang, 0) + line_count

    return file_count, total_lines, lang_files, lang_lines


def _detect_patterns(root: Path) -> Dict[str, List[str]]:
    """Detect architecture patterns from imports and module structure."""
    patterns: Dict[str, List[str]] = {
        "imported_packages": [],
        "module_dirs": [],
        "entry_points": [],
    }

    # Module dirs (contain __init__.py)
    for dirpath, dirnames, filenames in os.walk(root):
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv"}
        dirnames[:] = [d for d in dirnames if d not in skip]
        if "__init__.py" in filenames:
            rel = Path(dirpath).relative_to(root)
            patterns["module_dirs"].append(str(rel))

    # Common imported packages from Python files (top-level only)
    imports: Dict[str, int] = {}
    import_re = re.compile(r"^(?:from|import)\s+(\w+)")
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            try:
                with open(Path(dirpath) / fname, errors="ignore") as f:
                    for line in f:
                        m = import_re.match(line)
                        if m:
                            pkg = m.group(1)
                            imports[pkg] = imports.get(pkg, 0) + 1
            except OSError:
                continue

    # Top 15 most-imported packages
    patterns["imported_packages"] = sorted(imports, key=imports.get, reverse=True)[:15]  # type: ignore[arg-type]

    # Entry points (main.py, app.py, manage.py, etc.)
    entry_names = {"main.py", "app.py", "manage.py", "cli.py", "server.py", "run.py", "wsgi.py", "asgi.py"}
    for name in entry_names:
        if (root / name).exists():
            patterns["entry_points"].append(name)

    return patterns


def _scan_debt(root: Path) -> Dict[str, List[str]]:
    """Scan for TODO, FIXME, HACK, etc."""
    debt: Dict[str, List[str]] = {k: [] for k in _DEBT_PATTERNS}
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in _LANG_MAP and fname not in ("Makefile", "Dockerfile", "Cargo.toml"):
                continue
            fpath = Path(dirpath) / fname
            try:
                with open(fpath, errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        for label, pat in _DEBT_PATTERNS.items():
                            if pat.search(line):
                                rel = fpath.relative_to(root)
                                debt[label].append(f"{rel}:{i}: {line.strip()[:120]}")
            except OSError:
                continue

    return debt


def _get_recent_commits(root: str) -> List[str]:
    """Get last 20 git commits."""
    out = _run(["git", "log", "--oneline", "-20"], root)
    if out:
        return [l.strip() for l in out.strip().splitlines() if l.strip()]
    return []


def _get_test_info(root: str) -> tuple:
    """Try to collect tests via pytest --co."""
    out = _run(["python3", "-m", "pytest", "--co", "-q", "--no-header"], root, timeout=60)
    if out is None:
        return 0, []
    tests = [l.strip() for l in out.strip().splitlines() if l.strip() and "test" in l.lower()]
    return len(tests), tests


def survey_codebase(root: Optional[str] = None) -> CodebaseState:
    """Survey a codebase and return a CodebaseState snapshot.

    Args:
        root: Path to the codebase root. Defaults to current working directory.

    Returns:
        A CodebaseState dataclass with survey results.
    """
    root = root or os.getcwd()
    root_path = Path(root).resolve()

    if not root_path.is_dir():
        return CodebaseState(root=root, errors=[f"Not a directory: {root}"])

    state = CodebaseState(root=str(root_path))
    errors: List[str] = []

    # File and line counts
    file_count, total_lines, lang_files, lang_lines = _count_files_and_lines(root_path)
    state.file_count = file_count
    state.total_lines = total_lines
    state.language_breakdown = lang_files
    state.language_lines = lang_lines

    # Architecture patterns
    state.architecture_patterns = _detect_patterns(root_path)

    # Technical debt
    state.technical_debt = _scan_debt(root_path)

    # Recent commits
    state.recent_commits = _get_recent_commits(str(root_path))

    # Test info
    test_count, test_collected = _get_test_info(str(root_path))
    state.test_count = test_count
    state.test_collected = test_collected

    state.errors = errors
    return state
