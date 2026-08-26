"""
Auto-Documentation Pipeline

Watches fleet code for changes and automatically regenerates
documentation, keeping README, API reference, and guides in sync.

Usage:
    from fleet.auto_doc_pipeline import AutoDocPipeline
    pipeline = AutoDocPipeline()
    pipeline.scan_all()
    pipeline.regenerate()
    pipeline.write_to_disk()
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fleet.doc_generator import DocGenerator


@dataclass
class DocFile:
    """A tracked documentation file."""

    path: str
    content: str = ""
    last_hash: str = ""
    last_generated: float = 0.0

    def compute_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def is_stale(self) -> bool:
        return self.compute_hash() != self.last_hash


class AutoDocPipeline:
    """
    Automated documentation pipeline.

    Scans code, detects changes, regenerates docs, writes to disk.
    """

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.doc_generator = DocGenerator()
        self.doc_files: Dict[str, DocFile] = {}
        self.code_hashes: Dict[str, str] = {}
        self.changes_detected: List[str] = []

    def scan_all(self, code_dirs: Optional[List[str]] = None):
        """Scan all code and documentation."""
        code_dirs = code_dirs or ["fleet/", "swarm/", "nerve/", "nexus/", "logos/"]
        self.doc_generator.scan_modules(code_dirs, self.project_root)
        self._compute_code_hashes(code_dirs)

    def _compute_code_hashes(self, code_dirs: List[str]):
        """Compute hashes of all source files."""
        for directory in code_dirs:
            dir_path = os.path.join(self.project_root, directory)
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".py"):
                        path = os.path.join(root, file)
                        with open(path, "r") as f:
                            content = f.read()
                        self.code_hashes[path] = hashlib.sha256(
                            content.encode()
                        ).hexdigest()[:16]

    def detect_changes(self, code_dirs: Optional[List[str]] = None) -> List[str]:
        """Detect which files changed since last scan."""
        changed = []
        code_dirs = code_dirs or ["fleet/", "swarm/", "nerve/", "nexus/", "logos/"]
        for directory in code_dirs:
            dir_path = os.path.join(self.project_root, directory)
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".py"):
                        path = os.path.join(root, file)
                        with open(path, "r") as f:
                            content = f.read()
                        new_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                        if (
                            path in self.code_hashes
                            and self.code_hashes[path] != new_hash
                        ):
                            changed.append(path)
        self.changes_detected = changed
        return changed

    def regenerate(self) -> Dict[str, str]:
        """Regenerate all documentation."""
        docs = {
            "README.md": self.doc_generator.generate_readme(),
            "API_REFERENCE.md": self.doc_generator.generate_api_reference(),
            "MODULE_INDEX.md": self.doc_generator.generate_module_index(),
        }

        for path, content in docs.items():
            if path not in self.doc_files:
                self.doc_files[path] = DocFile(path)
            self.doc_files[path].content = content
            self.doc_files[path].last_generated = time.time()

        return docs

    def write_to_disk(self, docs_dir: str = "docs/auto") -> List[str]:
        """Write all generated docs to disk."""
        os.makedirs(docs_dir, exist_ok=True)
        written = []
        for doc_file in self.doc_files.values():
            if not doc_file.content:
                continue
            path = os.path.join(docs_dir, doc_file.path)
            with open(path, "w") as f:
                f.write(doc_file.content)
            doc_file.last_hash = doc_file.compute_hash()
            written.append(path)
        return written

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "modules": len(self.doc_generator.modules),
            "tests": self.doc_generator.total_tests,
            "doc_files": len(self.doc_files),
            "changes_detected": len(self.changes_detected),
            "stale_docs": sum(1 for d in self.doc_files.values() if d.is_stale()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.get_status(),
            "code_files": len(self.code_hashes),
            "doc_files": {k: v.last_generated for k, v in self.doc_files.items()},
        }
