"""FleetDoc — auto-documentation generator for the fleet ecosystem.

Reads all fleet module docstrings, builds cross-referenced API docs,
generates ASCII architecture diagrams, and produces comprehensive
integration guides.

Usage
-----
    from fleet.fleet_doc import FleetDoc

    doc = FleetDoc()
    doc.generate_api_docs("docs/API_INDEX.md")
    doc.generate_architecture_diagram("docs/ARCHITECTURE.md")
    doc.generate_integration_guide("docs/INTEGRATION_GUIDE.md")
"""

from __future__ import annotations

__all__ = [
    "FleetDoc",
    "ModuleDoc",
    "FunctionDoc",
    "ClassDoc",
    "ArchitectureNode",
    "ArchitectureEdge",
]

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleet.harbor import Harbor
from fleet.ternary_types import TernaryValue


@dataclass
class FunctionDoc:
    """Documentation for a single function/method."""

    name: str
    signature: str
    docstring: str
    is_method: bool = False
    source_file: str = ""
    line_number: int = 0


@dataclass
class ClassDoc:
    """Documentation for a single class."""

    name: str
    docstring: str
    methods: list[FunctionDoc] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0


@dataclass
class ModuleDoc:
    """Documentation for a fleet module."""

    name: str
    docstring: str
    classes: list[ClassDoc] = field(default_factory=list)
    functions: list[FunctionDoc] = field(default_factory=list)
    test_count: int = 0
    test_passed: int = 0
    source_file: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ArchitectureNode:
    """Node in an architecture diagram."""

    name: str
    layer: str
    description: str
    status: str
    health_emoji: str


@dataclass
class ArchitectureEdge:
    """Edge in an architecture diagram."""

    source: str
    target: str
    label: str
    direction: str = "->"


class FleetDoc:
    """Fleet documentation generator.

    Parameters
    ----------
    workspace : str
        Path to sunset-ecosystem workspace.
    """

    MODULE_PREFIXES = {
        "fleet": "Fleet Layer",
        "swarm": "Swarm Layer",
        "nerve": "Nerve Layer",
        "nexus": "Nexus Layer",
    }

    LAYER_ORDER = [
        "Nerve Layer",
        "Swarm Layer",
        "Fleet Layer",
        "Nexus Layer",
    ]

    def __init__(self, workspace: str = ".") -> None:
        self.workspace = Path(workspace)
        self._harbor: Harbor | None = None
        self._docs: dict[str, ModuleDoc] = {}

    def _ensure_harbor(self) -> None:
        if self._harbor is None:
            self._harbor = Harbor(str(self.workspace))
            self._harbor.bootstrap_fleet()

    # ── Module Parsing ────────────────────────────────────────

    def parse_module(self, module_path: str) -> ModuleDoc:
        """Parse a Python module for docstrings and signatures."""
        path = Path(module_path)
        if not path.exists():
            return ModuleDoc(name=path.stem, docstring="", source_file=str(path))

        content = path.read_text()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ModuleDoc(
                name=path.stem,
                docstring="File has syntax errors",
                source_file=str(path),
            )

        docstring = ast.get_docstring(tree) or ""
        classes: list[ClassDoc] = []
        functions: list[FunctionDoc] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_doc = self._parse_class(node, str(path))
                classes.append(cls_doc)
            elif isinstance(node, ast.FunctionDef) and node.name != "__init__":
                func_doc = self._parse_function(node, str(path), is_method=False)
                functions.append(func_doc)

        return ModuleDoc(
            name=path.stem,
            docstring=docstring,
            classes=classes,
            functions=functions,
            source_file=str(path),
        )

    def _parse_class(self, node: ast.ClassDef, source_file: str) -> ClassDoc:
        docstring = ast.get_docstring(node) or ""
        methods: list[FunctionDoc] = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._parse_function(item, source_file, is_method=True))

        return ClassDoc(
            name=node.name,
            docstring=docstring,
            methods=methods,
            source_file=source_file,
            line_number=node.lineno,
        )

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_file: str,
        is_method: bool,
    ) -> FunctionDoc:
        docstring = ast.get_docstring(node) or ""
        args = []
        for arg in node.args.args:
            if arg.arg != "self" and arg.arg != "cls":
                args.append(arg.arg)
        signature = f"{node.name}({', '.join(args)})"

        return FunctionDoc(
            name=node.name,
            signature=signature,
            docstring=docstring,
            is_method=is_method,
            source_file=source_file,
            line_number=node.lineno,
        )

    def parse_all_modules(self) -> dict[str, ModuleDoc]:
        """Parse all fleet and swarm modules."""
        self._ensure_harbor()
        if not self._harbor:
            return {}

        for mod in self._harbor.modules.values():
            # Try to find the source file
            candidates = [
                self.workspace / mod.path,
                self.workspace / "fleet" / f"{mod.name.lower().replace('_', '')}.py",
                self.workspace / "swarm" / f"{mod.name.lower().replace('_', '')}.py",
                self.workspace / "nerve" / f"{mod.name.lower().replace('_', '')}.py",
                self.workspace / "nexus" / f"{mod.name.lower().replace('_', '')}.py",
            ]

            # Also try exact name matches
            for prefix in ["fleet", "swarm", "nerve", "nexus"]:
                candidates.append(self.workspace / prefix / f"{mod.name.lower()}.py")
                candidates.append(
                    self.workspace / prefix / f"{mod.name.lower().replace('_', '_')}.py"
                )

            source_path = None
            for candidate in candidates:
                if candidate.exists():
                    source_path = str(candidate)
                    break

            if not source_path:
                continue

            doc = self.parse_module(source_path)
            doc.test_count = mod.test_count
            doc.test_passed = mod.test_passed
            doc.dependencies = mod.dependencies
            self._docs[mod.name] = doc

        return self._docs

    # ── API Documentation ─────────────────────────────────────

    def generate_api_docs(self, output_path: str | Path) -> str:
        """Generate comprehensive API documentation in Markdown.

        Parameters
        ----------
        output_path : str | Path
            Path to write the documentation.

        Returns
        -------
        str
            The Markdown content.
        """
        self.parse_all_modules()

        lines: list[str] = []
        lines.append("# 📚 Sunset Ecosystem API Documentation")
        lines.append("")
        lines.append(f"*Auto-generated from {len(self._docs)} modules*")
        lines.append("")

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for name in sorted(self._docs.keys()):
            doc = self._docs[name]
            lines.append(f"- [{name}](#{name.lower().replace(' ', '-')})")
        lines.append("")

        # Module details
        for name in sorted(self._docs.keys()):
            doc = self._docs[name]
            lines.append(f"## {name}")
            lines.append("")

            if doc.docstring:
                lines.append(textwrap.dedent(doc.docstring))
                lines.append("")

            # Source info
            lines.append(f"**Source:** `{doc.source_file}`")
            lines.append(f"**Tests:** {doc.test_passed}/{doc.test_count}")
            if doc.dependencies:
                lines.append(f"**Dependencies:** {', '.join(doc.dependencies)}")
            lines.append("")

            # Classes
            for cls in doc.classes:
                lines.append(f"### `class {cls.name}`")
                lines.append("")
                if cls.docstring:
                    lines.append(textwrap.dedent(cls.docstring))
                    lines.append("")

                for method in cls.methods:
                    lines.append(f"#### `{method.signature}`")
                    lines.append("")
                    if method.docstring:
                        lines.append(textwrap.dedent(method.docstring))
                        lines.append("")
                    lines.append(
                        f"*Source: `{method.source_file}:{method.line_number}`*"
                    )
                    lines.append("")

            # Functions
            for func in doc.functions:
                lines.append(f"### `{func.signature}`")
                lines.append("")
                if func.docstring:
                    lines.append(textwrap.dedent(func.docstring))
                    lines.append("")
                lines.append(f"*Source: `{func.source_file}:{func.line_number}`*")
                lines.append("")

            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        Path(output_path).write_text(content)
        return content

    # ── Architecture Diagram ─────────────────────────────────

    def generate_architecture_diagram(self, output_path: str | Path) -> str:
        """Generate an ASCII architecture diagram.

        Parameters
        ----------
        output_path : str | Path
            Path to write the diagram.

        Returns
        -------
        str
            The Markdown content.
        """
        self._ensure_harbor()
        if not self._harbor:
            return ""

        nodes = self._build_architecture_nodes()
        edges = self._build_architecture_edges()

        lines: list[str] = []
        lines.append("# 🏗️ Sunset Ecosystem Architecture")
        lines.append("")
        lines.append("```")
        lines.append("")

        # Build layer sections
        for layer in self.LAYER_ORDER:
            layer_nodes = [n for n in nodes if n.layer == layer]
            if not layer_nodes:
                continue

            lines.append(f"┌{'─' * 50}┐")
            lines.append(f"│ {layer:48} │")
            lines.append(f"├{'─' * 50}┤")

            for node in layer_nodes:
                status = (
                    node.status[:8] if len(node.status) <= 8 else node.status[:7] + "…"
                )
                lines.append(f"│ {node.health_emoji} {node.name:30} {status:15} │")

            lines.append(f"└{'─' * 50}┘")
            lines.append("")

        # Integration lines
        if edges:
            lines.append("Integration Flows:")
            lines.append("")
            for edge in edges:
                lines.append(
                    f"  {edge.source} {edge.direction} {edge.target} [{edge.label}]"
                )
            lines.append("")

        lines.append("```")
        lines.append("")

        # Legend
        lines.append("## Legend")
        lines.append("")
        lines.append("| Emoji | Status |")
        lines.append("|-------|--------|")
        lines.append("| 🟢 | Healthy |")
        lines.append("| 🟡 | Warning |")
        lines.append("| 🔴 | Critical |")
        lines.append("")

        content = "\n".join(lines)
        Path(output_path).write_text(content)
        return content

    def _build_architecture_nodes(self) -> list[ArchitectureNode]:
        nodes: list[ArchitectureNode] = []
        if not self._harbor:
            return nodes

        for mod in self._harbor.modules.values():
            layer = "Unknown"
            for prefix, layer_name in self.MODULE_PREFIXES.items():
                if prefix in mod.path.lower():
                    layer = layer_name
                    break

            # Determine health emoji
            if mod.health_ternary == TernaryValue.POS:
                emoji = "🟢"
                status = "healthy"
            elif mod.health_ternary == TernaryValue.NEG:
                emoji = "🔴"
                status = "critical"
            else:
                emoji = "🟡"
                status = "warning"

            nodes.append(
                ArchitectureNode(
                    name=mod.name,
                    layer=layer,
                    description=mod.description,
                    status=status,
                    health_emoji=emoji,
                )
            )

        return nodes

    def _build_architecture_edges(self) -> list[ArchitectureEdge]:
        edges: list[ArchitectureEdge] = []
        if not self._harbor:
            return edges

        for path in self._harbor.integrations:
            edges.append(
                ArchitectureEdge(
                    source=path.source,
                    target=path.target,
                    label=path.status,
                )
            )

        return edges

    # ── Integration Guide ─────────────────────────────────────

    def generate_integration_guide(self, output_path: str | Path) -> str:
        """Generate a comprehensive integration guide.

        Parameters
        ----------
        output_path : str | Path
            Path to write the guide.

        Returns
        -------
        str
            The Markdown content.
        """
        self._ensure_harbor()
        if not self._harbor:
            return ""

        lines: list[str] = []
        lines.append("# 🔗 Integration Guide")
        lines.append("")
        lines.append("How to integrate new modules into the Sunset Ecosystem fleet.")
        lines.append("")

        # Prerequisites
        lines.append("## Prerequisites")
        lines.append("")
        lines.append("Before adding a new module, ensure:")
        lines.append("")
        lines.append(
            "1. **Tests**: Every module must have a comprehensive pytest suite"
        )
        lines.append(
            "2. **Docstrings**: All public classes and functions must have docstrings"
        )
        lines.append(
            "3. **Integration**: Identify at least one existing module to connect with"
        )
        lines.append("4. **Documentation**: Add a description of what the module does")
        lines.append("")

        # Step-by-step
        lines.append("## Step-by-Step Integration")
        lines.append("")
        lines.append("### 1. Register the Module")
        lines.append("")
        lines.append("```python")
        lines.append("from fleet.harbor import Harbor")
        lines.append("")
        lines.append("harbor = Harbor()")
        lines.append(
            'harbor.register_module("MyModule", "swarm/my_module.py", ["VectorSwarm"])'
        )
        lines.append("```")
        lines.append("")

        lines.append("### 2. Add Integration Paths")
        lines.append("")
        lines.append("```python")
        lines.append(
            'harbor.add_integration("MyModule", "VectorSwarm", "MyModule uses VectorSwarm for search")'
        )
        lines.append("```")
        lines.append("")

        lines.append("### 3. Verify Health")
        lines.append("")
        lines.append("```python")
        lines.append("health = harbor.check_fleet_health()")
        lines.append('print(health["healthy"], "/", health["total"])')
        lines.append("```")
        lines.append("")

        lines.append("### 4. Run Tests")
        lines.append("")
        lines.append("```bash")
        lines.append("python3 -m pytest tests/test_my_module.py -v")
        lines.append("```")
        lines.append("")

        # Existing integrations
        lines.append("## Existing Integration Patterns")
        lines.append("")
        lines.append("| Source | Target | Pattern |")
        lines.append("|--------|--------|---------|")

        for path in self._harbor.integrations:
            lines.append(f"| {path.source} | {path.target} | {path.status} |")

        lines.append("")

        # Gaps
        gaps = self._harbor.find_integration_gaps()
        if gaps:
            lines.append("## Known Integration Gaps")
            lines.append("")
            for gap in gaps:
                lines.append(f"- **{gap['source']} → {gap['target']}**: {gap['issue']}")
            lines.append("")

        # Dependency order
        lines.append("## Dependency Order")
        lines.append("")
        lines.append("Modules should be initialized in this order:")
        lines.append("")
        for i, name in enumerate(self._harbor.get_dependency_order(), 1):
            lines.append(f"{i}. {name}")
        lines.append("")

        content = "\n".join(lines)
        Path(output_path).write_text(content)
        return content

    # ── Console Output ────────────────────────────────────────

    def print_module_summary(self) -> None:
        """Print a console summary of all modules."""
        self.parse_all_modules()

        print("═" * 60)
        print(" 📚 FLEET DOCUMENTATION SUMMARY")
        print("═" * 60)
        print(f"  Modules documented: {len(self._docs)}")
        print(
            f"  Total classes:      {sum(len(d.classes) for d in self._docs.values())}"
        )
        print(
            f"  Total functions:    {sum(len(d.functions) for d in self._docs.values())}"
        )
        print(
            f"  Total methods:      {sum(sum(len(c.methods) for c in d.classes) for d in self._docs.values())}"
        )
        print("═" * 60)

        for name, doc in sorted(self._docs.items()):
            cls_count = len(doc.classes)
            func_count = len(doc.functions)
            method_count = sum(len(c.methods) for c in doc.classes)
            print(
                f"  {name:25} {cls_count:2d} classes, {func_count:2d} funcs, {method_count:2d} methods"
            )

        print("═" * 60)
