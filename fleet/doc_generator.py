"""
Fleet Documentation Generator

Auto-generates educational documentation from fleet code, tests, and schemas.
Produces both human-readable guides and agent-specification documents.

Usage:
    from fleet.doc_generator import DocGenerator
    gen = DocGenerator("sunset-ecosystem")
    gen.scan_modules(["swarm/", "fleet/"])
    gen.generate_readme()
    gen.generate_api_reference()
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModuleDoc:
    """Documentation for a single module."""

    name: str
    path: str
    docstring: str = ""
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    test_count: int = 0
    test_status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "docstring": self.docstring,
            "classes": self.classes,
            "functions": self.functions,
            "test_count": self.test_count,
            "test_status": self.test_status,
        }


class DocGenerator:
    """
    Auto-generates documentation from fleet code.

    Scans Python modules, extracts docstrings, signatures, and test info.
    """

    def __init__(self, project_name: str = "sunset-ecosystem"):
        self.project_name = project_name
        self.modules: List[ModuleDoc] = []
        self.total_tests = 0
        self.total_lines = 0

    def scan_modules(self, directories: List[str], base_path: str = "."):
        """Scan directories for Python modules."""
        for directory in directories:
            dir_path = os.path.join(base_path, directory)
            if not os.path.exists(dir_path):
                continue

            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".py") and not file.startswith("test_"):
                        path = os.path.join(root, file)
                        self._scan_file(path)

    def _scan_file(self, path: str):
        """Scan a single Python file."""
        try:
            with open(path, "r") as f:
                source = f.read()
                tree = ast.parse(source)
        except Exception:
            return

        module_doc = ModuleDoc(
            name=os.path.basename(path),
            path=path,
        )

        # Count lines
        self.total_lines += len(source.splitlines())

        # Extract docstring
        if ast.get_docstring(tree):
            module_doc.docstring = ast.get_docstring(tree)

        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "methods": [
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                }
                module_doc.classes.append(class_info)
            elif isinstance(node, ast.FunctionDef) and not isinstance(
                node, ast.ClassDef
            ):
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "args": [arg.arg for arg in node.args.args],
                }
                module_doc.functions.append(func_info)

        # Count tests
        test_path = path.replace(".py", "_test.py").replace("/", "/test_")
        test_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "tests")
        test_file = os.path.join(test_dir, f"test_{os.path.basename(path)}")
        if os.path.exists(test_file):
            module_doc.test_count = self._count_tests(test_file)
            module_doc.test_status = "present"
            self.total_tests += module_doc.test_count

        self.modules.append(module_doc)

    def _count_tests(self, path: str) -> int:
        """Count test functions in a test file."""
        try:
            with open(path, "r") as f:
                tree = ast.parse(f.read())
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        count += 1
            return count
        except Exception:
            return 0

    def generate_module_index(self) -> str:
        """Generate module index markdown."""
        lines = [
            f"# {self.project_name} Module Index",
            "",
            f"**Total Modules:** {len(self.modules)}",
            f"**Total Tests:** {self.total_tests}",
            f"**Total Lines:** {self.total_lines}",
            "",
            "| Module | Classes | Functions | Tests | Status |",
            "|--------|---------|-----------|-------|--------|",
        ]

        for mod in self.modules:
            lines.append(
                f"| {mod.name} | {len(mod.classes)} | {len(mod.functions)} | "
                f"{mod.test_count} | {mod.test_status} |"
            )

        return "\n".join(lines)

    def generate_api_reference(self) -> str:
        """Generate API reference markdown."""
        lines = [f"# {self.project_name} API Reference", ""]

        for mod in self.modules:
            lines.append(f"## {mod.name}")
            if mod.docstring:
                lines.append(mod.docstring)
                lines.append("")

            for cls in mod.classes:
                lines.append(f"### class {cls['name']}")
                if cls["docstring"]:
                    lines.append(cls["docstring"])
                if cls["methods"]:
                    lines.append(f"**Methods:** {', '.join(cls['methods'])}")
                lines.append("")

            for func in mod.functions:
                lines.append(f"### {func['name']}({', '.join(func['args'])})")
                if func["docstring"]:
                    lines.append(func["docstring"])
                lines.append("")

        return "\n".join(lines)

    def generate_readme(self, extra_sections: Optional[Dict[str, str]] = None) -> str:
        """Generate a comprehensive README."""
        sections = {
            "overview": self._generate_overview(),
            "modules": self.generate_module_index(),
            "api": self.generate_api_reference(),
            "tests": self._generate_test_summary(),
        }
        if extra_sections:
            sections.update(extra_sections)

        lines = [
            f"# {self.project_name}",
            "",
            sections.get("overview", ""),
            "",
            "## Quick Start",
            "",
            "```python",
            "from fleet.openconstruct_shell import OpenConstructShell",
            "shell = OpenConstructShell()",
            "shell.spawn('my-agent', {'task': 'optimize'})",
            "```",
            "",
            sections.get("modules", ""),
            "",
            sections.get("tests", ""),
            "",
            sections.get("api", ""),
            "",
        ]

        return "\n".join(lines)

    def _generate_overview(self) -> str:
        """Generate project overview."""
        return (
            f"{self.project_name} is a fleet-scale breeding and orchestration ecosystem.\n\n"
            f"- **{len(self.modules)}** modules\n"
            f"- **{self.total_tests}** tests\n"
            f"- **{self.total_lines}** lines of code\n"
            "\n"
            "## Features\n"
            "- Multi-paradigm breeding (genetic, swarm, GNN, meta-learning)\n"
            "- Spatial awareness and navigation\n"
            "- Fleet consensus and coordination\n"
            "- Agent-native interfaces and A2A communication\n"
        )

    def _generate_test_summary(self) -> str:
        """Generate test summary."""
        tested = [m for m in self.modules if m.test_count > 0]
        untested = [m for m in self.modules if m.test_count == 0]

        lines = [
            "## Test Summary",
            "",
            f"**Tested Modules:** {len(tested)}/{len(self.modules)}",
            f"**Total Tests:** {self.total_tests}",
            "",
        ]

        if untested:
            lines.append("### Modules Without Tests")
            for mod in untested:
                lines.append(f"- {mod.name}")
            lines.append("")

        return "\n".join(lines)

    def generate_agent_spec(self) -> Dict[str, Any]:
        """Generate agent specification document."""
        return {
            "project": self.project_name,
            "version": "1.0",
            "modules": [m.to_dict() for m in self.modules],
            "total_tests": self.total_tests,
            "total_lines": self.total_lines,
            "entry_points": [
                "fleet.openconstruct_shell.OpenConstructShell",
                "fleet.openconstruct_bridge.OpenConstructBridge",
                "swarm.breeder_daemon_v2.BreederDaemonV2",
            ],
        }

    def export_json(self, path: str = "docs/auto_generated.json"):
        """Export documentation as JSON."""
        data = {
            "project": self.project_name,
            "modules": [m.to_dict() for m in self.modules],
            "readme": self.generate_readme(),
            "api_reference": self.generate_api_reference(),
            "agent_spec": self.generate_agent_spec(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def get_stats(self) -> Dict[str, Any]:
        return {
            "modules": len(self.modules),
            "tests": self.total_tests,
            "lines": self.total_lines,
            "classes": sum(len(m.classes) for m in self.modules),
            "functions": sum(len(m.functions) for m in self.modules),
        }
