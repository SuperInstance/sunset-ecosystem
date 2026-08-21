"""Tests for FleetDoc — auto-documentation generator for the fleet.

Reference: fleet/fleet_doc.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.fleet_doc import (
    ArchitectureEdge,
    ArchitectureNode,
    ClassDoc,
    FleetDoc,
    FunctionDoc,
    ModuleDoc,
)


class TestModuleDoc:
    def test_defaults(self) -> None:
        d = ModuleDoc(name="Test", docstring="A test module")
        assert d.classes == []
        assert d.functions == []
        assert d.dependencies == []

    def test_fields(self) -> None:
        d = ModuleDoc(
            name="Test",
            docstring="A test module",
            test_count=10,
            test_passed=8,
            dependencies=["A", "B"],
        )
        assert d.test_count == 10
        assert d.test_passed == 8
        assert d.dependencies == ["A", "B"]


class TestClassDoc:
    def test_defaults(self) -> None:
        c = ClassDoc(name="TestClass", docstring="A class")
        assert c.methods == []

    def test_methods(self) -> None:
        c = ClassDoc(
            name="TestClass",
            docstring="A class",
            methods=[FunctionDoc(name="run", signature="run()", docstring="Run")],
        )
        assert len(c.methods) == 1
        assert c.methods[0].name == "run"


class TestFunctionDoc:
    def test_fields(self) -> None:
        f = FunctionDoc(
            name="test_func",
            signature="test_func(a, b)",
            docstring="Test function",
            is_method=False,
            source_file="test.py",
            line_number=10,
        )
        assert f.name == "test_func"
        assert f.signature == "test_func(a, b)"
        assert f.is_method is False


class TestArchitectureNode:
    def test_fields(self) -> None:
        n = ArchitectureNode(
            name="TestModule",
            layer="Fleet Layer",
            description="A module",
            status="healthy",
            health_emoji="🟢",
        )
        assert n.layer == "Fleet Layer"
        assert n.health_emoji == "🟢"


class TestArchitectureEdge:
    def test_fields(self) -> None:
        e = ArchitectureEdge(source="A", target="B", label="uses")
        assert e.direction == "->"


class TestFleetDoc:
    def test_init(self) -> None:
        doc = FleetDoc()
        assert doc._harbor is None
        assert doc._docs == {}

    def test_ensure_harbor(self) -> None:
        doc = FleetDoc()
        doc._ensure_harbor()
        assert doc._harbor is not None

    def test_parse_module(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        test_file = tmp_path / "test_module.py"
        test_file.write_text('''"""A test module."""

class TestClass:
    """A test class."""
    
    def run(self, x: int) -> int:
        """Run the thing."""
        return x + 1

def helper():
    """A helper function."""
    pass
''')
        result = doc.parse_module(str(test_file))
        assert result.name == "test_module"
        assert "A test module" in result.docstring
        assert len(result.classes) == 1
        assert result.classes[0].name == "TestClass"
        assert len(result.classes[0].methods) == 1
        assert result.classes[0].methods[0].name == "run"
        assert len(result.functions) == 1
        assert result.functions[0].name == "helper"

    def test_parse_module_syntax_error(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        test_file = tmp_path / "bad.py"
        test_file.write_text("def bad(")
        result = doc.parse_module(str(test_file))
        assert "syntax errors" in result.docstring

    def test_parse_module_missing(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        result = doc.parse_module(str(tmp_path / "nonexistent.py"))
        assert result.name == "nonexistent"

    def test_parse_all_modules(self) -> None:
        doc = FleetDoc()
        docs = doc.parse_all_modules()
        assert len(docs) > 0
        # Should have at least some known modules
        assert any(
            name in docs for name in ["Harbor", "FleetOrchestrator", "TernaryTypes"]
        )

    def test_generate_api_docs(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        output = tmp_path / "api.md"
        content = doc.generate_api_docs(output)
        assert output.exists()
        assert "# 📚 Sunset Ecosystem API Documentation" in content
        assert "Table of Contents" in content

    def test_api_docs_has_modules(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        output = tmp_path / "api.md"
        content = doc.generate_api_docs(output)
        assert len(doc._docs) > 0
        for name in doc._docs:
            assert f"## {name}" in content

    def test_generate_architecture_diagram(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        output = tmp_path / "arch.md"
        content = doc.generate_architecture_diagram(output)
        assert output.exists()
        assert "# 🏗️ Sunset Ecosystem Architecture" in content
        assert "Fleet Layer" in content or "Swarm Layer" in content

    def test_architecture_nodes(self) -> None:
        doc = FleetDoc()
        doc._ensure_harbor()
        nodes = doc._build_architecture_nodes()
        assert len(nodes) > 0
        assert all(n.layer in doc.LAYER_ORDER for n in nodes)

    def test_architecture_edges(self) -> None:
        doc = FleetDoc()
        doc._ensure_harbor()
        edges = doc._build_architecture_edges()
        assert len(edges) > 0

    def test_generate_integration_guide(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        output = tmp_path / "guide.md"
        content = doc.generate_integration_guide(output)
        assert output.exists()
        assert "# 🔗 Integration Guide" in content
        assert "Prerequisites" in content
        assert "Step-by-Step Integration" in content

    def test_integration_guide_has_patterns(self, tmp_path: Path) -> None:
        doc = FleetDoc()
        output = tmp_path / "guide.md"
        content = doc.generate_integration_guide(output)
        assert "Existing Integration Patterns" in content
        assert "Dependency Order" in content

    def test_print_module_summary(self, capsys) -> None:
        doc = FleetDoc()
        doc.print_module_summary()
        captured = capsys.readouterr()
        assert "FLEET DOCUMENTATION SUMMARY" in captured.out
        assert "Modules documented:" in captured.out

    def test_layer_prefixes(self) -> None:
        assert "fleet" in FleetDoc.MODULE_PREFIXES
        assert "swarm" in FleetDoc.MODULE_PREFIXES

    def test_layer_order(self) -> None:
        assert FleetDoc.LAYER_ORDER[0] == "Nerve Layer"
        assert FleetDoc.LAYER_ORDER[-1] == "Nexus Layer"
